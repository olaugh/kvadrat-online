use chrono::{DateTime, SecondsFormat, Utc};
use flate2::{Compression, write::GzEncoder};
use kvadrat_strategy::native::{
    BoardEvaluation, FragmentModelPair, FragmentRerank, HEIGHT, PackedBoard, Piece, RootRankModel,
    RootRerank, SearchOutcome, Strategy, WIDTH,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::env;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

const PIECE_NAMES: [&str; 7] = ["I", "J", "L", "O", "S", "T", "Z"];
const MAX_LINES: u32 = 40;
const MAX_POSITIONS: usize = 400;

type AnyError = Box<dyn Error + Send + Sync>;
type Result<T> = std::result::Result<T, AnyError>;

#[derive(Clone, Debug)]
struct Options {
    hours: f64,
    max_games: u64,
    output: PathBuf,
    seed: u32,
    shard_records: usize,
    depths: Vec<usize>,
    beam_width: Option<usize>,
    threads: usize,
    fragment_full: Option<PathBuf>,
    fragment_context: Option<PathBuf>,
    fragment_weight: f64,
    fragment_candidates: usize,
    root_candidate_beam: Option<usize>,
    root_candidates: usize,
    root_ranker: Option<PathBuf>,
    root_ranker_candidate_beam: usize,
    root_ranker_candidates: usize,
    root_ranker_weight: f64,
}

#[derive(Clone)]
struct LexiconAssets {
    name: &'static str,
    id: u8,
    strategy: Arc<Strategy>,
    word_bags: Arc<Vec<Vec<[u8; 4]>>>,
}

struct Assets {
    csw24: LexiconAssets,
    nwl23: LexiconAssets,
    fragment_model: Option<Arc<FragmentModelPair>>,
    root_ranker: Option<Arc<RootRankModel>>,
}

impl Assets {
    fn for_game(&self, game_index: u64) -> &LexiconAssets {
        if game_index.is_multiple_of(2) {
            &self.csw24
        } else {
            &self.nwl23
        }
    }
}

#[derive(Clone, Copy)]
struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0x9e37_79b9 } else { seed },
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }

    fn index(&mut self, length: usize) -> usize {
        let unit = self.next_u32() as f64 / 4_294_967_296.0;
        (unit * length as f64).floor() as usize
    }
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct CurrentMetrics {
    score: i32,
    lines: u32,
    pieces: u32,
    words: u32,
    total_word_length: u32,
}

#[derive(Clone, Serialize)]
struct BoardView {
    letters: Vec<String>,
    pieces: Vec<String>,
}

#[derive(Clone, Serialize)]
struct PieceView {
    piece: String,
    letters: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Features {
    heights: Vec<i32>,
    holes: i32,
    buried_depth: i32,
    aggregate_height: i32,
    maximum_height: i32,
    bumpiness: i32,
    wells: i32,
    word_potential: f64,
    setup_words: Vec<String>,
    heuristic_value: f64,
}

#[derive(Clone, Serialize)]
struct Position {
    board: BoardView,
    active: PieceView,
    next: Vec<PieceView>,
    current: CurrentMetrics,
    features: Features,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Plan {
    piece: String,
    source_letters: Vec<String>,
    letters: Vec<String>,
    letter_shift: u8,
    rotation: u8,
    row: i8,
    col: i8,
    immediate_score: i32,
    immediate_lines: u8,
    immediate_words: Vec<String>,
    projected_score: i32,
    projected_lines: u16,
    setup_words: Vec<String>,
    depth: u8,
    nodes: u32,
    evaluation: i32,
    reason: String,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct Policy {
    depth: usize,
    beam_width: usize,
}

#[derive(Clone)]
struct UnlabelledRecord {
    episode_id: String,
    step: usize,
    lexicon: String,
    seed: u32,
    policy: Policy,
    position: Position,
    action: Plan,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct Delta {
    score: i32,
    lines: i32,
    words: i32,
    word_length: i32,
}

#[derive(Serialize)]
struct Horizons {
    #[serde(rename = "4")]
    four: Delta,
    #[serde(rename = "8")]
    eight: Delta,
    #[serde(rename = "16")]
    sixteen: Delta,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Target {
    completed: bool,
    topped_out: bool,
    terminal_score: i32,
    terminal_lines: u32,
    score_to_go: i32,
    lines_to_go: i32,
    words_to_go: i32,
    word_length_to_go: i32,
    score_per_line_to_go: f64,
    horizons: Horizons,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TrainingRecord {
    schema_version: u8,
    episode_id: String,
    step: usize,
    lexicon: String,
    seed: u32,
    policy: Policy,
    position: Position,
    action: Plan,
    target: Target,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct EpisodeSummary {
    episode_id: String,
    lexicon: String,
    seed: u32,
    depth: usize,
    beam_width: usize,
    positions: usize,
    score: i32,
    lines: u32,
    words: u32,
    average_word_length: f64,
    phase: String,
    search_nodes: u64,
    elapsed_ms: u64,
}

#[derive(Default, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Aggregate {
    episodes: u64,
    positions: u64,
    completed: u64,
    topped_out: u64,
    score: i64,
    lines: u64,
    words: u64,
    search_nodes: u64,
}

impl Aggregate {
    fn add(&mut self, summary: &EpisodeSummary) {
        self.episodes += 1;
        self.positions += summary.positions as u64;
        self.completed += u64::from(summary.phase == "complete");
        self.topped_out += u64::from(summary.phase == "over");
        self.score += summary.score as i64;
        self.lines += summary.lines as u64;
        self.words += summary.words as u64;
        self.search_nodes += summary.search_nodes;
    }
}

#[derive(Clone, Serialize)]
struct ShardSummary {
    file: String,
    records: usize,
    bytes: u64,
}

struct EpisodeResult {
    records: Vec<TrainingRecord>,
    summary: EpisodeSummary,
}

struct WorkerMessage {
    game_index: u64,
    result: Result<EpisodeResult>,
}

struct GameState {
    board: PackedBoard,
    piece_queue: VecDeque<u8>,
    letter_queue: VecDeque<[u8; 4]>,
    active: Option<Piece>,
    random: XorShift32,
    current: CurrentMetrics,
    phase: &'static str,
}

impl GameState {
    fn new(seed: u32, assets: &LexiconAssets) -> Self {
        let mut game = Self {
            board: [0; WIDTH * HEIGHT],
            piece_queue: VecDeque::new(),
            letter_queue: VecDeque::new(),
            active: None,
            random: XorShift32::new(seed),
            current: CurrentMetrics {
                score: 0,
                lines: 0,
                pieces: 0,
                words: 0,
                total_word_length: 0,
            },
            phase: "playing",
        };
        game.spawn_piece(assets);
        game
    }

    fn ensure_queues(&mut self, assets: &LexiconAssets) {
        while self.piece_queue.len() < 14 {
            let mut bag = [0u8, 1, 2, 3, 4, 5, 6];
            for index in (1..bag.len()).rev() {
                let swap = self.random.index(index + 1);
                bag.swap(index, swap);
            }
            self.piece_queue.extend(bag);
        }
        while self.letter_queue.len() < 56 {
            let bag_index = self.random.index(assets.word_bags.len());
            self.letter_queue
                .extend(assets.word_bags[bag_index].iter().copied());
        }
    }

    fn spawn_piece(&mut self, assets: &LexiconAssets) {
        self.ensure_queues(assets);
        let piece = Piece {
            kind: self.piece_queue.pop_front().expect("piece queue"),
            letters: self.letter_queue.pop_front().expect("letter queue"),
        };
        if !assets.strategy.can_spawn(&self.board, piece.kind) {
            self.active = None;
            self.phase = "over";
            return;
        }
        self.active = Some(piece);
        self.current.pieces += 1;
    }

    fn sequence(&mut self, assets: &LexiconAssets, depth: usize) -> Vec<Piece> {
        self.ensure_queues(assets);
        let mut sequence = Vec::with_capacity(depth);
        sequence.push(self.active.expect("active piece"));
        sequence.extend(
            self.piece_queue
                .iter()
                .zip(self.letter_queue.iter())
                .take(depth.saturating_sub(1))
                .map(|(&kind, &letters)| Piece { kind, letters }),
        );
        sequence
    }

    fn leaf_visible(&mut self, assets: &LexiconAssets, depth: usize) -> Vec<Piece> {
        self.ensure_queues(assets);
        self.piece_queue
            .iter()
            .zip(self.letter_queue.iter())
            .skip(depth.saturating_sub(1))
            .take(5)
            .map(|(&kind, &letters)| Piece { kind, letters })
            .collect()
    }

    fn root_visible(&mut self, assets: &LexiconAssets) -> Vec<Piece> {
        self.ensure_queues(assets);
        self.piece_queue
            .iter()
            .zip(self.letter_queue.iter())
            .take(5)
            .map(|(&kind, &letters)| Piece { kind, letters })
            .collect()
    }

    fn position(&mut self, assets: &LexiconAssets) -> Result<Position> {
        self.ensure_queues(assets);
        let active = self
            .active
            .ok_or("position requested without an active piece")?;
        let evaluation = assets
            .strategy
            .evaluate(&self.board)
            .ok_or("native board evaluation rejected a valid game board")?;
        Ok(Position {
            board: board_view(&self.board),
            active: piece_view(active),
            next: self
                .piece_queue
                .iter()
                .zip(self.letter_queue.iter())
                .take(4)
                .map(|(&kind, &letters)| piece_view(Piece { kind, letters }))
                .collect(),
            current: self.current,
            features: features(evaluation),
        })
    }

    fn apply(&mut self, assets: &LexiconAssets, outcome: &SearchOutcome) {
        self.board = outcome.board;
        self.current.score += outcome.immediate_score;
        self.current.lines += outcome.immediate_lines as u32;
        self.current.words += outcome.immediate_words.len() as u32;
        self.current.total_word_length += outcome
            .immediate_words
            .iter()
            .map(|word| word.text.len() as u32)
            .sum::<u32>();
        self.active = None;
        if self.current.lines >= MAX_LINES {
            self.phase = "complete";
        } else {
            self.spawn_piece(assets);
        }
    }
}

fn parse_options(root: &Path) -> Result<Options> {
    let mut values = HashMap::new();
    let args: Vec<String> = env::args().skip(1).collect();
    if !args.len().is_multiple_of(2) {
        return Err("expected --name value arguments".into());
    }
    for pair in args.chunks_exact(2) {
        let key = pair[0]
            .strip_prefix("--")
            .ok_or("expected every option to begin with --")?;
        values.insert(key.to_string(), pair[1].clone());
    }

    let hardware_threads = thread::available_parallelism().map_or(1, usize::from);
    let default_threads = hardware_threads.saturating_sub(1).max(1);
    let depths: Vec<usize> = values
        .get("depths")
        .map(String::as_str)
        .unwrap_or("2,3,3,3")
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if depths.is_empty() || depths.iter().any(|depth| !(1..=5).contains(depth)) {
        return Err("--depths must contain comma-separated integers from 1 through 5".into());
    }
    let hours = values
        .get("hours")
        .map(String::as_str)
        .unwrap_or("8")
        .parse()?;
    if !matches!(hours, value if value > 0.0 && f64::is_finite(value)) {
        return Err("--hours must be positive and finite".into());
    }
    let shard_records = values
        .get("shard-records")
        .map(String::as_str)
        .unwrap_or("2000")
        .parse()?;
    if shard_records < 100 {
        return Err("--shard-records must be at least 100".into());
    }
    let threads = values
        .get("threads")
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(default_threads);
    if threads == 0 {
        return Err("--threads must be at least 1".into());
    }
    let beam_width = values
        .get("beam-width")
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if beam_width.is_some_and(|width| !(12..=160).contains(&width)) {
        return Err("--beam-width must be from 12 through 160".into());
    }
    let stamp = Utc::now().format("%Y-%m-%dT%H-%M-%S%.3fZ");
    let output = values.get("output").map_or_else(
        || root.join(format!("training-data/selfplay-{stamp}")),
        |value| {
            let path = PathBuf::from(value);
            if path.is_absolute() {
                path
            } else {
                env::current_dir()
                    .unwrap_or_else(|_| root.to_path_buf())
                    .join(path)
            }
        },
    );
    let optional_path = |name: &str| -> Result<Option<PathBuf>> {
        values
            .get(name)
            .map(|value| {
                let path = PathBuf::from(value);
                Ok(if path.is_absolute() {
                    path
                } else {
                    env::current_dir()?.join(path)
                })
            })
            .transpose()
    };
    let fragment_full = optional_path("fragment-full")?;
    let fragment_context = optional_path("fragment-context")?;
    if fragment_full.is_some() != fragment_context.is_some() {
        return Err("--fragment-full and --fragment-context must be supplied together".into());
    }
    let fragment_weight = values
        .get("fragment-weight")
        .map(String::as_str)
        .unwrap_or("1")
        .parse()?;
    if !matches!(fragment_weight, value if f64::is_finite(value) && value >= 0.0) {
        return Err("--fragment-weight must be finite and nonnegative".into());
    }
    let fragment_candidates = values
        .get("fragment-candidates")
        .map(String::as_str)
        .unwrap_or("12")
        .parse()?;
    if !(1..=160).contains(&fragment_candidates) {
        return Err("--fragment-candidates must be from 1 through 160".into());
    }
    let root_candidate_beam = values
        .get("root-candidate-beam")
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if root_candidate_beam.is_some_and(|width| !(12..=160).contains(&width)) {
        return Err("--root-candidate-beam must be from 12 through 160".into());
    }
    let root_candidates = values
        .get("root-candidates")
        .map(String::as_str)
        .unwrap_or("12")
        .parse()?;
    if !(1..=160).contains(&root_candidates) {
        return Err("--root-candidates must be from 1 through 160".into());
    }
    let root_ranker = optional_path("root-ranker")?;
    if root_ranker.is_some() && fragment_full.is_some() {
        return Err("--root-ranker cannot be combined with the fragment model options".into());
    }
    if root_candidate_beam.is_some() && (root_ranker.is_some() || fragment_full.is_some()) {
        return Err("--root-candidate-beam cannot be combined with learned model options".into());
    }
    if root_ranker.is_some() && depths.iter().any(|&depth| depth != 3) {
        return Err("--root-ranker currently requires --depths 3".into());
    }
    let root_ranker_candidate_beam = values
        .get("root-ranker-candidate-beam")
        .map(String::as_str)
        .unwrap_or("160")
        .parse()?;
    if !(12..=160).contains(&root_ranker_candidate_beam) {
        return Err("--root-ranker-candidate-beam must be from 12 through 160".into());
    }
    let root_ranker_candidates = values
        .get("root-ranker-candidates")
        .map(String::as_str)
        .unwrap_or("12")
        .parse()?;
    if !(1..=160).contains(&root_ranker_candidates) {
        return Err("--root-ranker-candidates must be from 1 through 160".into());
    }
    let root_ranker_weight = values
        .get("root-ranker-weight")
        .map(String::as_str)
        .unwrap_or("1")
        .parse()?;
    if !matches!(root_ranker_weight, value if f64::is_finite(value) && value >= 0.0) {
        return Err("--root-ranker-weight must be finite and nonnegative".into());
    }
    Ok(Options {
        hours,
        max_games: values
            .get("games")
            .map(String::as_str)
            .unwrap_or("9007199254740991")
            .parse()?,
        output,
        seed: values
            .get("seed")
            .map(String::as_str)
            .unwrap_or("12628309")
            .parse()?,
        shard_records,
        depths,
        beam_width,
        threads,
        fragment_full,
        fragment_context,
        fragment_weight,
        fragment_candidates,
        root_candidate_beam,
        root_candidates,
        root_ranker,
        root_ranker_candidate_beam,
        root_ranker_candidates,
        root_ranker_weight,
    })
}

fn load_assets(root: &Path, options: &Options) -> Result<Assets> {
    let fragment_model = options
        .fragment_full
        .as_ref()
        .zip(options.fragment_context.as_ref())
        .map(|(full, context)| -> Result<Arc<FragmentModelPair>> {
            Ok(Arc::new(FragmentModelPair::from_bytes(
                &fs::read(full)?,
                &fs::read(context)?,
            )?))
        })
        .transpose()?;
    let root_ranker = options
        .root_ranker
        .as_ref()
        .map(|path| -> Result<Arc<RootRankModel>> {
            Ok(Arc::new(RootRankModel::from_bytes(&fs::read(path)?)?))
        })
        .transpose()?;
    Ok(Assets {
        csw24: load_lexicon(root, "CSW24", 0)?,
        nwl23: load_lexicon(root, "NWL23", 1)?,
        fragment_model,
        root_ranker,
    })
}

fn load_lexicon(root: &Path, name: &'static str, id: u8) -> Result<LexiconAssets> {
    let data = root.join("public/data");
    let kwg = fs::read(data.join(format!("{name}.kwg")))?;
    let bags = fs::read_to_string(data.join(format!("{}-bags.txt", name.to_lowercase())))?;
    let word_bags: Vec<Vec<[u8; 4]>> = bags
        .lines()
        .map(|line| {
            line.split_whitespace()
                .filter_map(|text| parse_letters(text).ok())
                .take(28)
                .collect::<Vec<_>>()
        })
        .filter(|bag| bag.len() >= 28)
        .collect();
    if word_bags.is_empty() {
        return Err(format!("{name} has no valid 28-piece letter bags").into());
    }
    Ok(LexiconAssets {
        name,
        id,
        strategy: Arc::new(Strategy::from_kwg_bytes(&kwg)?),
        word_bags: Arc::new(word_bags),
    })
}

fn parse_letters(text: &str) -> Result<[u8; 4]> {
    let bytes = text.as_bytes();
    if bytes.len() != 4 || bytes.iter().any(|byte| !byte.is_ascii_uppercase()) {
        return Err(format!("invalid four-letter piece {text:?}").into());
    }
    let letters: [u8; 4] = bytes.try_into().expect("validated four-letter piece");
    Ok(letters.map(|byte| byte - b'A' + 1))
}

fn piece_name(kind: u8) -> &'static str {
    PIECE_NAMES[kind as usize]
}

fn letters_string(letters: [u8; 4]) -> String {
    letters
        .into_iter()
        .map(|letter| char::from(b'A' + letter - 1))
        .collect()
}

fn letters_vec(letters: [u8; 4]) -> Vec<String> {
    letters
        .into_iter()
        .map(|letter| char::from(b'A' + letter - 1).to_string())
        .collect()
}

fn shifted_letters(letters: [u8; 4], shift: u8) -> [u8; 4] {
    std::array::from_fn(|index| letters[(index + shift as usize) % 4])
}

fn piece_view(piece: Piece) -> PieceView {
    PieceView {
        piece: piece_name(piece.kind).to_string(),
        letters: letters_string(piece.letters),
    }
}

fn board_view(board: &PackedBoard) -> BoardView {
    let mut letters = Vec::with_capacity(HEIGHT);
    let mut pieces = Vec::with_capacity(HEIGHT);
    for row in board.chunks_exact(WIDTH) {
        letters.push(
            row.iter()
                .map(|cell| {
                    let letter = cell & 0x1f;
                    if letter == 0 {
                        '.'
                    } else {
                        char::from(b'A' + letter - 1)
                    }
                })
                .collect(),
        );
        pieces.push(
            row.iter()
                .map(|cell| {
                    let piece = cell >> 5;
                    if piece == 0 {
                        '.'
                    } else {
                        piece_name(piece - 1).chars().next().expect("piece name")
                    }
                })
                .collect(),
        );
    }
    BoardView { letters, pieces }
}

fn features(evaluation: BoardEvaluation) -> Features {
    Features {
        heights: evaluation.heights.to_vec(),
        holes: evaluation.holes,
        buried_depth: evaluation.buried_depth,
        aggregate_height: evaluation.aggregate_height,
        maximum_height: evaluation.maximum_height,
        bumpiness: evaluation.bumpiness,
        wells: evaluation.wells,
        word_potential: evaluation.word_potential,
        setup_words: evaluation.setup_words,
        heuristic_value: evaluation.heuristic_value,
    }
}

fn beam_width(depth: usize) -> usize {
    match depth {
        1 | 2 => 48,
        3 => 64,
        _ => 72,
    }
}

fn game_seed(base: u32, game_index: u64) -> u32 {
    base.wrapping_add(
        (game_index as u32)
            .wrapping_add(1)
            .wrapping_mul(0x9e37_79b1),
    )
}

fn delta(from: CurrentMetrics, to: CurrentMetrics) -> Delta {
    Delta {
        score: to.score - from.score,
        lines: to.lines as i32 - from.lines as i32,
        words: to.words as i32 - from.words as i32,
        word_length: to.total_word_length as i32 - from.total_word_length as i32,
    }
}

fn plan(active: Piece, outcome: &SearchOutcome) -> Plan {
    let shifted = shifted_letters(active.letters, outcome.letter_shift);
    let immediate_words: Vec<String> = outcome
        .immediate_words
        .iter()
        .map(|word| word.text.clone())
        .collect();
    let reason = if immediate_words.is_empty() {
        format!(
            "Native depth-{} search selected column {} at {}°.",
            outcome.depth,
            outcome.col + 1,
            outcome.rotation as u16 * 90
        )
    } else {
        format!(
            "Bank {} for {} points at column {}, {}°.",
            immediate_words.join(" + "),
            outcome.immediate_score,
            outcome.col + 1,
            outcome.rotation as u16 * 90
        )
    };
    Plan {
        piece: piece_name(active.kind).to_string(),
        source_letters: letters_vec(active.letters),
        letters: letters_vec(shifted),
        letter_shift: outcome.letter_shift,
        rotation: outcome.rotation,
        row: outcome.row,
        col: outcome.col,
        immediate_score: outcome.immediate_score,
        immediate_lines: outcome.immediate_lines,
        immediate_words,
        projected_score: outcome.projected_score,
        projected_lines: outcome.projected_lines,
        setup_words: outcome.setup_words.clone(),
        depth: outcome.depth,
        nodes: outcome.nodes,
        evaluation: outcome.evaluation,
        reason,
    }
}

fn run_game(
    game_index: u64,
    started_at: &str,
    options: &Options,
    assets: &LexiconAssets,
    fragment_model: Option<&FragmentModelPair>,
    root_ranker: Option<&RootRankModel>,
) -> Result<EpisodeResult> {
    let depth = options.depths[game_index as usize % options.depths.len()];
    let width = options.beam_width.unwrap_or_else(|| beam_width(depth));
    let seed = game_seed(options.seed, game_index);
    let episode_id = format!("{started_at}-{game_index:07}");
    let began = Instant::now();
    let mut game = GameState::new(seed, assets);
    let mut records = Vec::new();
    let mut search_nodes = 0u64;

    while game.phase == "playing" && records.len() < MAX_POSITIONS {
        let active = game.active.ok_or("playing game lacks an active piece")?;
        let position = game.position(assets)?;
        let sequence = game.sequence(assets, depth);
        let outcome = if let Some(model) = root_ranker {
            let root_visible = game.root_visible(assets);
            assets.strategy.find_best_move_with_root_ranker(
                &game.board,
                game.current.lines as u8,
                &sequence,
                width,
                RootRerank {
                    visible_after_root: &root_visible,
                    model,
                    lexicon: assets.id,
                    candidate_beam_width: options.root_ranker_candidate_beam,
                    candidates: options.root_ranker_candidates,
                    correction_weight: options.root_ranker_weight,
                },
            )
        } else if let Some(candidate_beam_width) = options.root_candidate_beam {
            assets.strategy.find_best_move_with_root_candidates(
                &game.board,
                game.current.lines as u8,
                &sequence,
                width,
                candidate_beam_width,
                options.root_candidates,
            )
        } else if let Some(model) = fragment_model {
            let leaf_visible = game.leaf_visible(assets, depth);
            assets.strategy.find_best_move_with_fragment_model(
                &game.board,
                game.current.lines as u8,
                &sequence,
                width,
                FragmentRerank {
                    leaf_visible: &leaf_visible,
                    model,
                    lexicon: assets.id,
                    weight: options.fragment_weight,
                    candidates: options.fragment_candidates,
                },
            )
        } else {
            assets
                .strategy
                .find_best_move(&game.board, game.current.lines as u8, &sequence, width)
        };
        let Some(outcome) = outcome else {
            game.phase = "over";
            game.active = None;
            break;
        };
        let action = plan(active, &outcome);
        search_nodes += outcome.nodes as u64;
        records.push(UnlabelledRecord {
            episode_id: episode_id.clone(),
            step: records.len(),
            lexicon: assets.name.to_string(),
            seed,
            policy: Policy {
                depth,
                beam_width: width,
            },
            position,
            action,
        });
        game.apply(assets, &outcome);
    }

    let final_metrics = game.current;
    let completed = game.phase == "complete";
    let currents: Vec<CurrentMetrics> = records
        .iter()
        .map(|record| record.position.current)
        .collect();
    let training_records = records
        .into_iter()
        .enumerate()
        .map(|(index, record)| {
            let current = record.position.current;
            let metric_after = |moves: usize| {
                currents
                    .get(index + moves)
                    .copied()
                    .unwrap_or(final_metrics)
            };
            let outcome = delta(current, final_metrics);
            TrainingRecord {
                schema_version: 1,
                episode_id: record.episode_id,
                step: record.step,
                lexicon: record.lexicon,
                seed: record.seed,
                policy: record.policy,
                position: record.position,
                action: record.action,
                target: Target {
                    completed,
                    topped_out: game.phase == "over",
                    terminal_score: final_metrics.score,
                    terminal_lines: final_metrics.lines,
                    score_to_go: outcome.score,
                    lines_to_go: outcome.lines,
                    words_to_go: outcome.words,
                    word_length_to_go: outcome.word_length,
                    score_per_line_to_go: if outcome.lines > 0 {
                        outcome.score as f64 / outcome.lines as f64
                    } else {
                        0.0
                    },
                    horizons: Horizons {
                        four: delta(current, metric_after(4)),
                        eight: delta(current, metric_after(8)),
                        sixteen: delta(current, metric_after(16)),
                    },
                },
            }
        })
        .collect::<Vec<_>>();
    let summary = EpisodeSummary {
        episode_id,
        lexicon: assets.name.to_string(),
        seed,
        depth,
        beam_width: width,
        positions: training_records.len(),
        score: final_metrics.score,
        lines: final_metrics.lines,
        words: final_metrics.words,
        average_word_length: if final_metrics.words > 0 {
            final_metrics.total_word_length as f64 / final_metrics.words as f64
        } else {
            0.0
        },
        phase: game.phase.to_string(),
        search_nodes,
        elapsed_ms: began.elapsed().as_millis() as u64,
    };
    Ok(EpisodeResult {
        records: training_records,
        summary,
    })
}

struct CorpusWriter<'a> {
    options: &'a Options,
    started_at: DateTime<Utc>,
    deadline: DateTime<Utc>,
    aggregate: Aggregate,
    by_depth: BTreeMap<String, Aggregate>,
    by_lexicon: BTreeMap<String, Aggregate>,
    shards: Vec<ShardSummary>,
    shard_index: usize,
    current_shard_records: usize,
    last_status: Instant,
}

impl<'a> CorpusWriter<'a> {
    fn new(options: &'a Options, started_at: DateTime<Utc>, deadline: DateTime<Utc>) -> Self {
        Self {
            options,
            started_at,
            deadline,
            aggregate: Aggregate::default(),
            by_depth: options
                .depths
                .iter()
                .map(|depth| (depth.to_string(), Aggregate::default()))
                .collect(),
            by_lexicon: [
                ("CSW24".to_string(), Aggregate::default()),
                ("NWL23".to_string(), Aggregate::default()),
            ]
            .into_iter()
            .collect(),
            shards: Vec::new(),
            shard_index: 0,
            current_shard_records: 0,
            last_status: Instant::now(),
        }
    }

    fn shard_file(&self) -> String {
        format!("positions-{:05}.jsonl.gz", self.shard_index)
    }

    fn append_episode(&mut self, episode: EpisodeResult) -> Result<()> {
        if !episode.records.is_empty() {
            let mut raw = Vec::new();
            for record in &episode.records {
                serde_json::to_writer(&mut raw, record)?;
                raw.push(b'\n');
            }
            let mut encoder = GzEncoder::new(Vec::new(), Compression::new(6));
            encoder.write_all(&raw)?;
            let compressed = encoder.finish()?;
            let mut shard = OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.options.output.join(self.shard_file()))?;
            shard.write_all(&compressed)?;
            self.current_shard_records += episode.records.len();
        }
        let mut episodes = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.options.output.join("episodes.jsonl"))?;
        serde_json::to_writer(&mut episodes, &episode.summary)?;
        episodes.write_all(b"\n")?;

        self.aggregate.add(&episode.summary);
        self.by_depth
            .entry(episode.summary.depth.to_string())
            .or_default()
            .add(&episode.summary);
        self.by_lexicon
            .entry(episode.summary.lexicon.clone())
            .or_default()
            .add(&episode.summary);
        if self.current_shard_records >= self.options.shard_records {
            self.finalize_shard()?;
        }
        if self.aggregate.episodes.is_multiple_of(10)
            || self.last_status.elapsed() >= Duration::from_secs(60)
        {
            self.write_manifest("running")?;
            self.print_status("running");
            self.last_status = Instant::now();
        }
        Ok(())
    }

    fn finalize_shard(&mut self) -> Result<()> {
        if self.current_shard_records == 0 {
            return Ok(());
        }
        let file = self.shard_file();
        let bytes = fs::metadata(self.options.output.join(&file))?.len();
        self.shards.push(ShardSummary {
            file,
            records: self.current_shard_records,
            bytes,
        });
        self.shard_index += 1;
        self.current_shard_records = 0;
        Ok(())
    }

    fn manifest(&self, status: &str) -> Value {
        let current_shard = (self.current_shard_records > 0).then(|| {
            json!({
                "file": self.shard_file(),
                "records": self.current_shard_records,
            })
        });
        json!({
            "schemaVersion": 1,
            "status": status,
            "pid": std::process::id(),
            "startedAt": rfc3339(self.started_at),
            "deadline": rfc3339(self.deadline),
            "updatedAt": rfc3339(Utc::now()),
            "completedAt": (status == "complete").then(|| rfc3339(Utc::now())),
            "generator": "native-rust",
            "options": {
                "hours": self.options.hours,
                "maxGames": self.options.max_games,
                "output": self.options.output,
                "seed": self.options.seed,
                "shardRecords": self.options.shard_records,
                "depths": self.options.depths,
                "beamWidth": self.options.beam_width,
                "threads": self.options.threads,
                "fragmentFull": self.options.fragment_full,
                "fragmentContext": self.options.fragment_context,
                "fragmentWeight": self.options.fragment_weight,
                "fragmentCandidates": self.options.fragment_candidates,
                "rootCandidateBeam": self.options.root_candidate_beam,
                "rootCandidates": self.options.root_candidates,
                "rootRanker": self.options.root_ranker,
                "rootRankerCandidateBeam": self.options.root_ranker_candidate_beam,
                "rootRankerCandidates": self.options.root_ranker_candidates,
                "rootRankerWeight": self.options.root_ranker_weight,
            },
            "aggregate": self.aggregate,
            "byDepth": self.by_depth,
            "byLexicon": self.by_lexicon,
            "shards": self.shards,
            "currentShard": current_shard,
        })
    }

    fn write_manifest(&self, status: &str) -> Result<()> {
        let temporary = self.options.output.join("manifest.next.json");
        let destination = self.options.output.join("manifest.json");
        let mut file = File::create(&temporary)?;
        serde_json::to_writer_pretty(&mut file, &self.manifest(status))?;
        file.write_all(b"\n")?;
        fs::rename(temporary, destination)?;
        Ok(())
    }

    fn print_status(&self, status: &str) {
        let elapsed_minutes = (Utc::now() - self.started_at).num_milliseconds() as f64 / 60_000.0;
        let rate = self.aggregate.positions as f64 / elapsed_minutes.max(1.0 / 60.0);
        let mean_score = if self.aggregate.episodes > 0 {
            self.aggregate.score as f64 / self.aggregate.episodes as f64
        } else {
            0.0
        };
        println!(
            "{}",
            json!({
                "status": status,
                "elapsedMinutes": (elapsed_minutes * 100.0).round() / 100.0,
                "episodes": self.aggregate.episodes,
                "positions": self.aggregate.positions,
                "shards": self.shards.len(),
                "meanScore": mean_score.round(),
                "positionsPerMinute": rate.round(),
                "threads": self.options.threads,
                "output": self.options.output,
            })
        );
    }
}

fn rfc3339(date: DateTime<Utc>) -> String {
    date.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn write_schema(output: &Path) -> Result<()> {
    let schema = json!({
        "schemaVersion": 1,
        "format": "gzip-compressed JSON Lines",
        "generator": "native Rust; one independent game per worker",
        "boardEncoding": {
            "letters": "22 strings of 10 characters; . denotes empty and A-Z denotes a tile",
            "pieces": "22 strings of 10 characters; . denotes empty and I/J/L/O/S/T/Z denotes color group",
        },
        "labels": {
            "horizons": "Observed score, line, word, and word-length gain after 4, 8, and 16 placements",
            "scoreToGo": "Observed undiscounted terminal score minus score at this position",
            "scorePerLineToGo": "Observed score-to-go divided by observed remaining cleared lines",
        },
        "caveat": "On-policy Monte Carlo labels generated by the recorded native beam-search depth and width; shard files contain concatenated gzip members",
    });
    let mut file = File::create(output.join("schema.json"))?;
    serde_json::to_writer_pretty(&mut file, &schema)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn git_commit(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn write_provenance(
    root: &Path,
    output: &Path,
    options: &Options,
    started_at: DateTime<Utc>,
    deadline: DateTime<Utc>,
) -> Result<()> {
    let source = root.join("wasm-strategy/src/lib.rs");
    let generator = root.join("wasm-strategy/src/bin/kvadrat-self-play.rs");
    let executable = env::current_exe()?;
    let data = root.join("public/data");
    let fragment_model = options
        .fragment_full
        .as_ref()
        .zip(options.fragment_context.as_ref())
        .map(|(full, context)| -> Result<Value> {
            Ok(json!({
                "full": {
                    "path": full,
                    "sha256": sha256_file(full)?,
                },
                "context": {
                    "path": context,
                    "sha256": sha256_file(context)?,
                },
                "weight": options.fragment_weight,
                "candidates": options.fragment_candidates,
            }))
        })
        .transpose()?;
    let root_ranker = options
        .root_ranker
        .as_ref()
        .map(|path| -> Result<Value> {
            Ok(json!({
                "path": path,
                "sha256": sha256_file(path)?,
                "candidateBeam": options.root_ranker_candidate_beam,
                "candidates": options.root_ranker_candidates,
                "weight": options.root_ranker_weight,
            }))
        })
        .transpose()?;
    let provenance = json!({
        "schemaVersion": 1,
        "run": {
            "startedAt": rfc3339(started_at),
            "deadline": rfc3339(deadline),
            "baseSeed": options.seed,
            "requestedHours": options.hours,
            "beamWidth": options.beam_width,
            "rootCandidateBeam": options.root_candidate_beam,
            "rootCandidates": options.root_candidates,
            "threads": options.threads,
        },
        "generator": {
            "kind": "native-rust",
            "commit": git_commit(root),
            "engineSha256": sha256_file(&source)?,
            "generatorSha256": sha256_file(&generator)?,
            "binarySha256": sha256_file(&executable)?,
        },
        "assets": {
            "CSW24": {
                "kwgSha256": sha256_file(&data.join("CSW24.kwg"))?,
                "bagsSha256": sha256_file(&data.join("csw24-bags.txt"))?,
            },
            "NWL23": {
                "kwgSha256": sha256_file(&data.join("NWL23.kwg"))?,
                "bagsSha256": sha256_file(&data.join("nwl23-bags.txt"))?,
            },
            "fragmentModel": fragment_model,
            "rootRanker": root_ranker,
        },
    });
    let mut file = File::create(output.join("PROVENANCE.json"))?;
    serde_json::to_writer_pretty(&mut file, &provenance)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn run() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("strategy crate has no project parent")?
        .to_path_buf();
    let options = parse_options(&root)?;
    fs::create_dir_all(options.output.parent().ok_or("output has no parent")?)?;
    fs::create_dir(&options.output)?;

    let started_at = Utc::now();
    let duration = Duration::from_secs_f64(options.hours * 3_600.0);
    let deadline = started_at + chrono::Duration::from_std(duration)?;
    let deadline_instant = Instant::now() + duration;
    let started_at_text = rfc3339(started_at);
    let assets = Arc::new(load_assets(&root, &options)?);
    write_schema(&options.output)?;
    write_provenance(&root, &options.output, &options, started_at, deadline)?;
    let mut writer = CorpusWriter::new(&options, started_at, deadline);
    writer.write_manifest("running")?;

    let stop = Arc::new(AtomicBool::new(false));
    let signal_stop = Arc::clone(&stop);
    ctrlc::set_handler(move || signal_stop.store(true, Ordering::Relaxed))?;
    let next_game = Arc::new(AtomicU64::new(0));
    let (sender, receiver) = mpsc::sync_channel::<WorkerMessage>(options.threads * 2);
    let mut handles = Vec::with_capacity(options.threads);

    for _ in 0..options.threads {
        let sender = sender.clone();
        let assets = Arc::clone(&assets);
        let next_game = Arc::clone(&next_game);
        let stop = Arc::clone(&stop);
        let options = options.clone();
        let started_at = started_at_text.clone();
        handles.push(thread::spawn(move || {
            loop {
                if stop.load(Ordering::Relaxed) || Instant::now() >= deadline_instant {
                    break;
                }
                let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                if game_index >= options.max_games {
                    break;
                }
                let lexicon = assets.for_game(game_index);
                let result = run_game(
                    game_index,
                    &started_at,
                    &options,
                    lexicon,
                    assets.fragment_model.as_deref(),
                    assets.root_ranker.as_deref(),
                );
                let failed = result.is_err();
                if sender.send(WorkerMessage { game_index, result }).is_err() {
                    break;
                }
                if failed {
                    stop.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }));
    }
    drop(sender);

    let mut pending = BTreeMap::new();
    let mut next_to_write = 0u64;
    let mut worker_error: Option<AnyError> = None;
    for message in receiver {
        match message.result {
            Ok(episode) => {
                pending.insert(message.game_index, episode);
                while let Some(episode) = pending.remove(&next_to_write) {
                    writer.append_episode(episode)?;
                    next_to_write += 1;
                }
            }
            Err(error) => {
                worker_error = Some(error);
                stop.store(true, Ordering::Relaxed);
            }
        }
    }
    for handle in handles {
        if handle.join().is_err() {
            return Err("native self-play worker panicked".into());
        }
    }
    if let Some(error) = worker_error {
        return Err(error);
    }
    for (_, episode) in pending {
        writer.append_episode(episode)?;
    }
    writer.finalize_shard()?;
    writer.write_manifest("complete")?;
    writer.print_status("complete");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("kvadrat-self-play: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_sequence_matches_the_browser_generator() {
        let mut random = XorShift32::new(2_654_860_003);
        let actual = std::array::from_fn::<_, 6, _>(|_| random.next_u32());
        assert_eq!(
            actual,
            [
                743_561_395,
                4_289_843_472,
                2_011_547_897,
                296_451_844,
                3_073_623_041,
                969_144_536,
            ]
        );
    }

    #[test]
    fn letter_cycles_wrap_in_browser_order() {
        assert_eq!(shifted_letters([17, 9, 14, 19], 1), [9, 14, 19, 17]);
        assert_eq!(shifted_letters([17, 9, 14, 19], 3), [19, 17, 9, 14]);
    }
}
