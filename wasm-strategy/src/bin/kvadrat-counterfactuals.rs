use flate2::read::MultiGzDecoder;
use kvadrat_strategy::native::{PackedBoard, Piece, RootCandidate, Strategy};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

const RECORD_BYTES: usize = 288;
const HEADER_BYTES: usize = 64;
const MAGIC: &[u8; 8] = b"KVCF1\0\0\0";
const PIECES: &str = "IJLOSTZ";
const TOP_OUT_PENALTY: i32 = 2_500;
const CURVE_LIMITS: [usize; 6] = [1, 2, 4, 6, 8, 12];

type AnyError = Box<dyn Error + Send + Sync>;
type Result<T> = std::result::Result<T, AnyError>;

#[derive(Clone)]
struct Options {
    input: PathBuf,
    output: PathBuf,
    positions: usize,
    candidates: usize,
    search_depth: usize,
    rollout_depth: usize,
    horizon: usize,
    candidate_beam_width: usize,
    rollout_beam_width: usize,
    threads: usize,
    sample_modulus: u32,
}

#[derive(Deserialize)]
struct CorpusManifest {
    status: String,
    shards: Vec<CorpusShard>,
}

#[derive(Deserialize)]
struct CorpusShard {
    file: String,
    records: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceRecord {
    episode_id: String,
    step: u16,
    lexicon: String,
    seed: u32,
    position: SourcePosition,
}

#[derive(Deserialize)]
struct SourcePosition {
    board: SourceBoard,
    active: SourcePiece,
    current: SourceCurrent,
}

#[derive(Deserialize)]
struct SourceBoard {
    letters: Vec<String>,
    pieces: Vec<String>,
}

#[derive(Deserialize)]
struct SourcePiece {
    piece: String,
    letters: String,
}

#[derive(Deserialize)]
struct SourceCurrent {
    lines: u8,
}

#[derive(Clone)]
struct Task {
    seed: u32,
    step: u16,
    lexicon: u8,
    board: PackedBoard,
    current_lines: u8,
    future: Vec<Piece>,
}

#[derive(Clone, Copy)]
struct Rollout {
    score: i32,
    lines: u16,
    words: u16,
    word_length: u16,
    placements: u8,
    completed: bool,
    topped_out: bool,
}

#[derive(Clone)]
struct CounterfactualRecord {
    board: PackedBoard,
    visible: [Piece; 5],
    current_lines: u8,
    lexicon: u8,
    rank: u8,
    seed: u32,
    step: u16,
    candidate_count: u16,
    immediate_score: i32,
    immediate_lines: u8,
    immediate_words: u8,
    immediate_word_length: u16,
    projected_score: i32,
    projected_lines: u16,
    heuristic_value: f32,
    rollout: Rollout,
}

impl CounterfactualRecord {
    fn objective(&self) -> i32 {
        self.immediate_score + self.rollout.score
            - i32::from(self.rollout.topped_out) * TOP_OUT_PENALTY
    }

    fn encode(&self) -> [u8; RECORD_BYTES] {
        let mut output = [0u8; RECORD_BYTES];
        output[..220].copy_from_slice(&self.board);
        for (index, piece) in self.visible.iter().enumerate() {
            let start = 220 + index * 5;
            output[start] = piece.kind;
            output[start + 1..start + 5].copy_from_slice(&piece.letters);
        }
        output[245] = self.current_lines;
        output[246] = self.lexicon;
        output[247] = self.rank;
        output[248..252].copy_from_slice(&self.seed.to_le_bytes());
        output[252..254].copy_from_slice(&self.step.to_le_bytes());
        output[254..256].copy_from_slice(&self.candidate_count.to_le_bytes());
        output[256..260].copy_from_slice(&self.immediate_score.to_le_bytes());
        output[260] = self.immediate_lines;
        output[261] = self.immediate_words;
        output[262..264].copy_from_slice(&self.immediate_word_length.to_le_bytes());
        output[264..268].copy_from_slice(&self.projected_score.to_le_bytes());
        output[268..270].copy_from_slice(&self.projected_lines.to_le_bytes());
        output[270..274].copy_from_slice(&self.heuristic_value.to_le_bytes());
        output[274..278].copy_from_slice(&self.rollout.score.to_le_bytes());
        output[278..280].copy_from_slice(&self.rollout.lines.to_le_bytes());
        output[280..282].copy_from_slice(&self.rollout.words.to_le_bytes());
        output[282..284].copy_from_slice(&self.rollout.word_length.to_le_bytes());
        output[284] = self.rollout.placements;
        output[285] = u8::from(self.rollout.completed);
        output[286] = u8::from(self.rollout.topped_out);
        output
    }
}

struct GroupResult {
    task_index: usize,
    records: Vec<CounterfactualRecord>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GroupMeasure {
    lexicon: &'static str,
    candidates: usize,
    baseline_objective: i32,
    oracle_objective: i32,
    oracle_delta: i32,
    oracle_rank: usize,
    baseline_topped_out: bool,
    oracle_topped_out: bool,
    curves: BTreeMap<usize, i32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    groups: usize,
    candidate_records: usize,
    mean_candidates: f64,
    baseline_mean_objective: f64,
    oracle_mean_objective: f64,
    mean_oracle_delta: f64,
    median_oracle_delta: f64,
    oracle_delta_ci95: [f64; 2],
    groups_with_headroom: usize,
    groups_changing_action: usize,
    baseline_top_outs: usize,
    oracle_top_outs: usize,
    mean_uplift_by_candidate_limit: BTreeMap<usize, f64>,
}

fn absolute(value: &str) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    Ok(if path.is_absolute() {
        path
    } else {
        env::current_dir()?.join(path)
    })
}

fn option<T: std::str::FromStr>(
    values: &HashMap<String, String>,
    name: &str,
    default: T,
) -> Result<T>
where
    T::Err: Error + Send + Sync + 'static,
{
    values
        .get(name)
        .map(|value| value.parse())
        .transpose()
        .map(|value| value.unwrap_or(default))
        .map_err(Into::into)
}

fn parse_options() -> Result<Options> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if !arguments.len().is_multiple_of(2) {
        return Err("expected --name value arguments".into());
    }
    let mut values = HashMap::new();
    for pair in arguments.chunks_exact(2) {
        let key = pair[0]
            .strip_prefix("--")
            .ok_or("expected every option to begin with --")?;
        values.insert(key.to_string(), pair[1].clone());
    }
    let hardware_threads = thread::available_parallelism().map_or(1, usize::from);
    let options = Options {
        input: absolute(values.get("input").ok_or("--input is required")?)?,
        output: absolute(values.get("output").ok_or("--output is required")?)?,
        positions: option(&values, "positions", 10_000)?,
        candidates: option(&values, "candidates", 12)?,
        search_depth: option(&values, "search-depth", 3)?,
        rollout_depth: option(&values, "rollout-depth", 3)?,
        horizon: option(&values, "horizon", 8)?,
        candidate_beam_width: option(&values, "candidate-beam-width", 160)?,
        rollout_beam_width: option(&values, "rollout-beam-width", 64)?,
        threads: option(
            &values,
            "threads",
            hardware_threads.saturating_sub(1).max(1),
        )?,
        sample_modulus: option(&values, "sample-modulus", 8)?,
    };
    if options.positions == 0
        || options.candidates == 0
        || options.candidates > u8::MAX as usize
        || !(1..=5).contains(&options.search_depth)
        || !(1..=5).contains(&options.rollout_depth)
        || options.horizon == 0
        || !(12..=160).contains(&options.candidate_beam_width)
        || !(12..=160).contains(&options.rollout_beam_width)
        || options.threads == 0
        || options.sample_modulus == 0
    {
        return Err("counterfactual options are outside their supported ranges".into());
    }
    Ok(options)
}

fn piece_id(text: &str) -> Result<u8> {
    let byte = *text.as_bytes().first().ok_or("empty piece name")?;
    PIECES
        .bytes()
        .position(|candidate| candidate == byte)
        .map(|index| index as u8)
        .ok_or_else(|| format!("invalid piece name {text:?}").into())
}

fn letter_id(byte: u8) -> Result<u8> {
    if byte.is_ascii_uppercase() {
        Ok(byte - b'A' + 1)
    } else {
        Err(format!("invalid letter byte {byte}").into())
    }
}

fn encode_piece(piece: &SourcePiece) -> Result<Piece> {
    if piece.letters.len() != 4 {
        return Err("piece does not contain four letters".into());
    }
    let mut letters = [0u8; 4];
    for (destination, &letter) in letters.iter_mut().zip(piece.letters.as_bytes()) {
        *destination = letter_id(letter)?;
    }
    Ok(Piece {
        kind: piece_id(&piece.piece)?,
        letters,
    })
}

fn encode_board(board: &SourceBoard) -> Result<PackedBoard> {
    if board.letters.len() != 22 || board.pieces.len() != 22 {
        return Err("board does not contain 22 rows".into());
    }
    let mut output = [0u8; 220];
    for row in 0..22 {
        let letters = board.letters[row].as_bytes();
        let pieces = board.pieces[row].as_bytes();
        if letters.len() != 10 || pieces.len() != 10 {
            return Err("board row does not contain ten cells".into());
        }
        for col in 0..10 {
            let cell = row * 10 + col;
            output[cell] = if letters[col] == b'.' {
                if pieces[col] != b'.' {
                    return Err("empty letter has a nonempty piece color".into());
                }
                0
            } else {
                let piece = PIECES
                    .bytes()
                    .position(|candidate| candidate == pieces[col])
                    .ok_or("board contains an invalid piece color")?
                    as u8;
                letter_id(letters[col])? | ((piece + 1) << 5)
            };
        }
    }
    Ok(output)
}

fn mixed_hash(seed: u32, step: u16) -> u32 {
    let mut mixed = seed ^ u32::from(step).wrapping_mul(0x9e37_79b1);
    mixed ^= mixed >> 16;
    mixed = mixed.wrapping_mul(0x7feb_352d);
    mixed ^= mixed >> 15;
    mixed = mixed.wrapping_mul(0x846c_a68b);
    mixed ^ (mixed >> 16)
}

fn add_episode_tasks(
    episode: &[SourceRecord],
    required_future: usize,
    options: &Options,
    tasks: &mut Vec<Task>,
) -> Result<()> {
    if episode.is_empty() || tasks.len() == options.positions {
        return Ok(());
    }
    if episode.len() < required_future {
        return Ok(());
    }
    for pair in episode.windows(2) {
        if pair[0].episode_id != pair[1].episode_id || pair[1].step != pair[0].step + 1 {
            return Err("source episode records are not contiguous".into());
        }
    }
    for index in 0..=episode.len().saturating_sub(required_future) {
        let record = &episode[index];
        if !mixed_hash(record.seed, record.step).is_multiple_of(options.sample_modulus) {
            continue;
        }
        let lexicon = match record.lexicon.as_str() {
            "CSW24" => 0,
            "NWL23" => 1,
            other => return Err(format!("unsupported lexicon {other:?}").into()),
        };
        let future = episode[index..index + required_future]
            .iter()
            .map(|future_record| {
                if future_record.seed != record.seed || future_record.lexicon != record.lexicon {
                    return Err("future sequence crosses an episode boundary".into());
                }
                encode_piece(&future_record.position.active)
            })
            .collect::<Result<Vec<_>>>()?;
        tasks.push(Task {
            seed: record.seed,
            step: record.step,
            lexicon,
            board: encode_board(&record.position.board)?,
            current_lines: record.position.current.lines,
            future,
        });
        if tasks.len() == options.positions {
            break;
        }
    }
    Ok(())
}

fn collect_tasks(options: &Options) -> Result<(Vec<Task>, usize, usize)> {
    let manifest: CorpusManifest =
        serde_json::from_reader(File::open(options.input.join("manifest.json"))?)?;
    if manifest.status != "complete" {
        return Err(format!("source corpus status is {}", manifest.status).into());
    }
    let required_future = options
        .search_depth
        .max(options.horizon + options.rollout_depth)
        .max(6);
    let mut tasks = Vec::with_capacity(options.positions);
    let mut scanned_shards = 0usize;
    let mut scanned_records = 0usize;
    for shard in &manifest.shards {
        let decoder = MultiGzDecoder::new(File::open(options.input.join(&shard.file))?);
        let reader = BufReader::with_capacity(1024 * 1024, decoder);
        let mut episode = Vec::new();
        let mut decoded = 0usize;
        for line in reader.lines() {
            let record: SourceRecord = serde_json::from_str(&line?)?;
            if episode
                .last()
                .is_some_and(|previous: &SourceRecord| previous.episode_id != record.episode_id)
            {
                add_episode_tasks(&episode, required_future, options, &mut tasks)?;
                episode.clear();
            }
            episode.push(record);
            decoded += 1;
            if tasks.len() == options.positions {
                break;
            }
        }
        add_episode_tasks(&episode, required_future, options, &mut tasks)?;
        if tasks.len() < options.positions && decoded != shard.records {
            return Err(format!(
                "{}: expected {} records, decoded {decoded}",
                shard.file, shard.records
            )
            .into());
        }
        scanned_shards += 1;
        scanned_records += decoded;
        if tasks.len() == options.positions {
            break;
        }
    }
    if tasks.len() != options.positions {
        return Err(format!(
            "source corpus supplied only {} eligible positions, requested {}",
            tasks.len(),
            options.positions
        )
        .into());
    }
    Ok((tasks, scanned_shards, scanned_records))
}

fn rollout(
    strategy: &Strategy,
    task: &Task,
    candidate: &RootCandidate,
    options: &Options,
) -> Rollout {
    let mut board = candidate.board;
    let mut current_lines =
        (u16::from(task.current_lines) + u16::from(candidate.immediate_lines)).min(255) as u8;
    let mut result = Rollout {
        score: 0,
        lines: 0,
        words: 0,
        word_length: 0,
        placements: 0,
        completed: current_lines >= 40,
        topped_out: false,
    };
    for placement in 0..options.horizon {
        if result.completed {
            break;
        }
        let start = 1 + placement;
        let sequence = &task.future[start..start + options.rollout_depth];
        let Some(outcome) =
            strategy.find_best_move(&board, current_lines, sequence, options.rollout_beam_width)
        else {
            result.topped_out = true;
            break;
        };
        board = outcome.board;
        result.score += outcome.immediate_score;
        result.lines += u16::from(outcome.immediate_lines);
        result.words += outcome.immediate_words.len() as u16;
        result.word_length += outcome
            .immediate_words
            .iter()
            .map(|word| word.text.len() as u16)
            .sum::<u16>();
        result.placements += 1;
        current_lines = current_lines.saturating_add(outcome.immediate_lines);
        result.completed = current_lines >= 40;
    }
    result
}

fn process_task(
    task_index: usize,
    task: &Task,
    strategies: &[Arc<Strategy>; 2],
    options: &Options,
) -> Result<GroupResult> {
    let strategy = &strategies[task.lexicon as usize];
    let search_sequence = &task.future[..options.search_depth];
    let baseline = strategy
        .find_best_move(
            &task.board,
            task.current_lines,
            search_sequence,
            options.rollout_beam_width,
        )
        .ok_or_else(|| format!("seed {} step {} has no baseline move", task.seed, task.step))?;
    let mut candidates = strategy.root_candidates(
        &task.board,
        task.current_lines,
        search_sequence,
        options.candidate_beam_width,
        options.candidates,
    );
    if candidates.is_empty() {
        return Err(format!(
            "seed {} step {} has no root candidates",
            task.seed, task.step
        )
        .into());
    }
    if let Some(baseline_index) = candidates
        .iter()
        .position(|candidate| candidate.board == baseline.board)
    {
        let baseline_candidate = candidates.remove(baseline_index);
        candidates.insert(0, baseline_candidate);
    } else {
        candidates.insert(
            0,
            RootCandidate {
                board: baseline.board,
                letter_shift: baseline.letter_shift,
                rotation: baseline.rotation,
                row: baseline.row,
                col: baseline.col,
                immediate_score: baseline.immediate_score,
                immediate_lines: baseline.immediate_lines,
                immediate_words: baseline.immediate_words,
                projected_score: baseline.projected_score,
                projected_lines: baseline.projected_lines,
                heuristic_value: f64::from(baseline.evaluation),
            },
        );
        candidates.truncate(options.candidates);
    }
    let visible: [Piece; 5] = task.future[1..6].try_into().expect("five visible pieces");
    let candidate_count = candidates.len() as u16;
    let records = candidates
        .iter()
        .enumerate()
        .map(|(rank, candidate)| {
            let immediate_word_length = candidate
                .immediate_words
                .iter()
                .map(|word| word.text.len() as u16)
                .sum();
            CounterfactualRecord {
                board: candidate.board,
                visible,
                current_lines: task.current_lines.saturating_add(candidate.immediate_lines),
                lexicon: task.lexicon,
                rank: rank as u8,
                seed: task.seed,
                step: task.step,
                candidate_count,
                immediate_score: candidate.immediate_score,
                immediate_lines: candidate.immediate_lines,
                immediate_words: candidate.immediate_words.len() as u8,
                immediate_word_length,
                projected_score: candidate.projected_score,
                projected_lines: candidate.projected_lines,
                heuristic_value: candidate.heuristic_value as f32,
                rollout: rollout(strategy, task, candidate, options),
            }
        })
        .collect();
    Ok(GroupResult {
        task_index,
        records,
    })
}

fn measure(records: &[CounterfactualRecord]) -> GroupMeasure {
    let baseline = &records[0];
    let mut oracle_rank = 0usize;
    for (rank, record) in records.iter().enumerate().skip(1) {
        if record.objective() > records[oracle_rank].objective() {
            oracle_rank = rank;
        }
    }
    let oracle = &records[oracle_rank];
    let mut curves = BTreeMap::new();
    for limit in CURVE_LIMITS {
        let objective = records
            .iter()
            .take(limit)
            .map(CounterfactualRecord::objective)
            .max()
            .expect("nonempty candidate prefix");
        curves.insert(limit, objective);
    }
    GroupMeasure {
        lexicon: if baseline.lexicon == 0 {
            "CSW24"
        } else {
            "NWL23"
        },
        candidates: records.len(),
        baseline_objective: baseline.objective(),
        oracle_objective: oracle.objective(),
        oracle_delta: oracle.objective() - baseline.objective(),
        oracle_rank,
        baseline_topped_out: baseline.rollout.topped_out,
        oracle_topped_out: oracle.rollout.topped_out,
        curves,
    }
}

fn report(measures: &[&GroupMeasure]) -> Report {
    let count = measures.len().max(1) as f64;
    let deltas: Vec<f64> = measures
        .iter()
        .map(|measure| f64::from(measure.oracle_delta))
        .collect();
    let mean_delta = deltas.iter().sum::<f64>() / count;
    let variance = if measures.len() > 1 {
        deltas
            .iter()
            .map(|delta| (delta - mean_delta).powi(2))
            .sum::<f64>()
            / (count - 1.0)
    } else {
        0.0
    };
    let standard_error = (variance / count).sqrt();
    let mut sorted = deltas;
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let mut curves = BTreeMap::new();
    for limit in CURVE_LIMITS {
        curves.insert(
            limit,
            measures
                .iter()
                .map(|measure| f64::from(measure.curves[&limit] - measure.baseline_objective))
                .sum::<f64>()
                / count,
        );
    }
    Report {
        groups: measures.len(),
        candidate_records: measures.iter().map(|measure| measure.candidates).sum(),
        mean_candidates: measures
            .iter()
            .map(|measure| measure.candidates as f64)
            .sum::<f64>()
            / count,
        baseline_mean_objective: measures
            .iter()
            .map(|measure| f64::from(measure.baseline_objective))
            .sum::<f64>()
            / count,
        oracle_mean_objective: measures
            .iter()
            .map(|measure| f64::from(measure.oracle_objective))
            .sum::<f64>()
            / count,
        mean_oracle_delta: mean_delta,
        median_oracle_delta: median,
        oracle_delta_ci95: [
            mean_delta - 1.96 * standard_error,
            mean_delta + 1.96 * standard_error,
        ],
        groups_with_headroom: measures
            .iter()
            .filter(|measure| measure.oracle_delta > 0)
            .count(),
        groups_changing_action: measures
            .iter()
            .filter(|measure| measure.oracle_rank > 0)
            .count(),
        baseline_top_outs: measures
            .iter()
            .filter(|measure| measure.baseline_topped_out)
            .count(),
        oracle_top_outs: measures
            .iter()
            .filter(|measure| measure.oracle_topped_out)
            .count(),
        mean_uplift_by_candidate_limit: curves,
    }
}

fn sha256(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn write_header(file: &mut File, groups: u64, records: u64) -> Result<()> {
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0u8; HEADER_BYTES];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&1u32.to_le_bytes());
    header[12..16].copy_from_slice(&(RECORD_BYTES as u32).to_le_bytes());
    header[16..24].copy_from_slice(&groups.to_le_bytes());
    header[24..32].copy_from_slice(&records.to_le_bytes());
    file.write_all(&header)?;
    Ok(())
}

fn run() -> Result<()> {
    let options = Arc::new(parse_options()?);
    fs::create_dir_all(options.output.parent().ok_or("output has no parent")?)?;
    fs::create_dir(&options.output)?;
    let began = Instant::now();
    let (tasks, scanned_shards, scanned_records) = collect_tasks(&options)?;
    println!(
        "{}",
        json!({
            "status": "selected",
            "positions": tasks.len(),
            "scannedShards": scanned_shards,
            "scannedRecords": scanned_records,
            "elapsedSeconds": began.elapsed().as_secs_f64(),
        })
    );

    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("strategy crate has no project parent")?;
    let strategies = Arc::new([
        Arc::new(Strategy::from_kwg_bytes(&fs::read(
            root.join("public/data/CSW24.kwg"),
        )?)?),
        Arc::new(Strategy::from_kwg_bytes(&fs::read(
            root.join("public/data/NWL23.kwg"),
        )?)?),
    ]);
    let tasks = Arc::new(tasks);
    let next_task = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::sync_channel::<Result<GroupResult>>(options.threads * 2);
    let mut handles = Vec::new();
    for _ in 0..options.threads {
        let options = Arc::clone(&options);
        let strategies = Arc::clone(&strategies);
        let tasks = Arc::clone(&tasks);
        let next_task = Arc::clone(&next_task);
        let sender = sender.clone();
        handles.push(thread::spawn(move || {
            loop {
                let index = next_task.fetch_add(1, Ordering::Relaxed);
                let Some(task) = tasks.get(index) else {
                    break;
                };
                if sender
                    .send(process_task(index, task, &strategies, &options))
                    .is_err()
                {
                    break;
                }
            }
        }));
    }
    drop(sender);

    let data_path = options.output.join("counterfactuals.kvcf");
    let mut data = BufWriter::new(File::create(&data_path)?);
    data.write_all(&[0; HEADER_BYTES])?;
    let mut pending = BTreeMap::new();
    let mut next_write = 0usize;
    let mut candidate_records = 0u64;
    let mut measures = Vec::with_capacity(tasks.len());
    for result in receiver {
        let result = result?;
        pending.insert(result.task_index, result);
        while let Some(group) = pending.remove(&next_write) {
            measures.push(measure(&group.records));
            for record in &group.records {
                data.write_all(&record.encode())?;
                candidate_records += 1;
            }
            next_write += 1;
            if next_write.is_multiple_of(100) || next_write == tasks.len() {
                let elapsed = began.elapsed().as_secs_f64();
                println!(
                    "{}",
                    json!({
                        "status": "rolling-out",
                        "positions": next_write,
                        "candidateRecords": candidate_records,
                        "positionsPerMinute": next_write as f64 / (elapsed / 60.0).max(1.0 / 60.0),
                        "elapsedMinutes": elapsed / 60.0,
                        "threads": options.threads,
                    })
                );
            }
        }
    }
    for handle in handles {
        if handle.join().is_err() {
            return Err("counterfactual worker panicked".into());
        }
    }
    if next_write != tasks.len() {
        return Err("counterfactual workers did not return every task".into());
    }
    data.flush()?;
    let mut data = data.into_inner()?;
    write_header(&mut data, tasks.len() as u64, candidate_records)?;
    data.sync_all()?;

    let all: Vec<&GroupMeasure> = measures.iter().collect();
    let csw24: Vec<&GroupMeasure> = measures
        .iter()
        .filter(|measure| measure.lexicon == "CSW24")
        .collect();
    let nwl23: Vec<&GroupMeasure> = measures
        .iter()
        .filter(|measure| measure.lexicon == "NWL23")
        .collect();
    let reports = json!({
        "all": report(&all),
        "CSW24": report(&csw24),
        "NWL23": report(&nwl23),
    });
    let manifest = json!({
        "schemaVersion": 1,
        "status": "complete",
        "generator": "native-rust-counterfactual-root-actions",
        "source": options.input,
        "sourceManifestSha256": sha256(&options.input.join("manifest.json"))?,
        "data": {
            "file": "counterfactuals.kvcf",
            "bytes": fs::metadata(&data_path)?.len(),
            "sha256": sha256(&data_path)?,
            "recordBytes": RECORD_BYTES,
            "groups": tasks.len(),
            "candidateRecords": candidate_records,
        },
        "selection": {
            "kind": "deterministic mixed hash of episode seed and step",
            "sampleModulus": options.sample_modulus,
            "scannedShards": scanned_shards,
            "scannedRecords": scanned_records,
        },
        "policy": {
            "candidateSource": "unique root actions represented by the heuristic depth frontier",
            "searchDepth": options.search_depth,
            "rolloutDepth": options.rollout_depth,
            "candidateBeamWidth": options.candidate_beam_width,
            "rolloutBeamWidth": options.rollout_beam_width,
            "horizon": options.horizon,
            "candidateLimit": options.candidates,
            "topOutPenalty": TOP_OUT_PENALTY,
            "objective": "immediate root score + same-queue rollout score - top-out penalty",
        },
        "inputs": {
            "board": "post-root-action packed 22x10 letter/color board",
            "visiblePieces": "next active plus four previews after the root action",
            "context": "line count after the root action and lexicon",
            "boundarySemantics": "adjacent cells with the same tetromino color share one boundary class",
        },
        "reports": reports,
        "threads": options.threads,
        "elapsedSeconds": began.elapsed().as_secs_f64(),
    });
    let mut manifest_file = BufWriter::new(File::create(options.output.join("MANIFEST.json"))?);
    serde_json::to_writer_pretty(&mut manifest_file, &manifest)?;
    manifest_file.write_all(b"\n")?;
    println!(
        "{}",
        json!({
            "status": "complete",
            "positions": tasks.len(),
            "candidateRecords": candidate_records,
            "reports": reports,
            "elapsedSeconds": began.elapsed().as_secs_f64(),
            "output": options.output,
        })
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("kvadrat-counterfactuals: {error}");
        std::process::exit(1);
    }
}
