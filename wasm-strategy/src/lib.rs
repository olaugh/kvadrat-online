use std::alloc::{Layout, alloc, dealloc};
use std::cmp::Ordering;
use std::slice;

mod fragment_model;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static FRAGMENT_MODELS: std::cell::RefCell<Vec<Option<fragment_model::FragmentModelPair>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

const BOARD_WIDTH: usize = 10;
const BOARD_HEIGHT: usize = 22;
const BOARD_CELLS: usize = BOARD_WIDTH * BOARD_HEIGHT;
const MAX_LINES: u16 = 40;
const MINIMUM_WORD_SCORE: i32 = 40;
const ARC_MASK: u32 = 0x3fffff;
const IS_END: u32 = 0x400000;
const ACCEPTS: u32 = 0x800000;

type Board = [u8; BOARD_CELLS];

const LETTER_VALUES: [i32; 27] = [
    0, 1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10,
];

#[derive(Clone, Copy)]
struct Block {
    x: i8,
    y: i8,
    letter_index: u8,
}

const EMPTY_BLOCK: Block = Block {
    x: 0,
    y: 0,
    letter_index: 0,
};

const BASE_BLOCKS: [[Block; 4]; 7] = [
    [
        Block {
            x: 0,
            y: 1,
            letter_index: 0,
        },
        Block {
            x: 1,
            y: 1,
            letter_index: 1,
        },
        Block {
            x: 2,
            y: 1,
            letter_index: 2,
        },
        Block {
            x: 3,
            y: 1,
            letter_index: 3,
        },
    ],
    [
        Block {
            x: 0,
            y: 0,
            letter_index: 0,
        },
        Block {
            x: 0,
            y: 1,
            letter_index: 1,
        },
        Block {
            x: 1,
            y: 1,
            letter_index: 2,
        },
        Block {
            x: 2,
            y: 1,
            letter_index: 3,
        },
    ],
    [
        Block {
            x: 2,
            y: 0,
            letter_index: 0,
        },
        Block {
            x: 0,
            y: 1,
            letter_index: 1,
        },
        Block {
            x: 1,
            y: 1,
            letter_index: 2,
        },
        Block {
            x: 2,
            y: 1,
            letter_index: 3,
        },
    ],
    [
        Block {
            x: 1,
            y: 1,
            letter_index: 0,
        },
        Block {
            x: 2,
            y: 1,
            letter_index: 1,
        },
        Block {
            x: 1,
            y: 2,
            letter_index: 2,
        },
        Block {
            x: 2,
            y: 2,
            letter_index: 3,
        },
    ],
    [
        Block {
            x: 1,
            y: 0,
            letter_index: 0,
        },
        Block {
            x: 2,
            y: 0,
            letter_index: 1,
        },
        Block {
            x: 0,
            y: 1,
            letter_index: 2,
        },
        Block {
            x: 1,
            y: 1,
            letter_index: 3,
        },
    ],
    [
        Block {
            x: 1,
            y: 0,
            letter_index: 0,
        },
        Block {
            x: 0,
            y: 1,
            letter_index: 1,
        },
        Block {
            x: 1,
            y: 1,
            letter_index: 2,
        },
        Block {
            x: 2,
            y: 1,
            letter_index: 3,
        },
    ],
    [
        Block {
            x: 0,
            y: 0,
            letter_index: 0,
        },
        Block {
            x: 1,
            y: 0,
            letter_index: 1,
        },
        Block {
            x: 1,
            y: 1,
            letter_index: 2,
        },
        Block {
            x: 2,
            y: 1,
            letter_index: 3,
        },
    ],
];

const PIECE_SIZES: [i8; 7] = [4, 3, 3, 4, 3, 3, 3];

const fn build_rotations() -> [[[Block; 4]; 4]; 7] {
    let mut result = [[[EMPTY_BLOCK; 4]; 4]; 7];
    let mut piece = 0;
    while piece < 7 {
        result[piece][0] = BASE_BLOCKS[piece];
        let mut rotation = 1;
        while rotation < 4 {
            let mut block = 0;
            while block < 4 {
                let previous = result[piece][rotation - 1][block];
                result[piece][rotation][block] = Block {
                    x: PIECE_SIZES[piece] - 1 - previous.y,
                    y: previous.x,
                    letter_index: previous.letter_index,
                };
                block += 1;
            }
            rotation += 1;
        }
        piece += 1;
    }
    result
}

const ROTATIONS: [[[Block; 4]; 4]; 7] = build_rotations();

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Text {
    len: u8,
    letters: [u8; BOARD_WIDTH],
}

impl Text {
    fn from_slice(letters: &[u8]) -> Self {
        let mut text = Self {
            len: letters.len() as u8,
            ..Self::default()
        };
        text.letters[..letters.len()].copy_from_slice(letters);
        text
    }

    fn as_slice(&self) -> &[u8] {
        &self.letters[..self.len as usize]
    }
}

#[derive(Clone, Copy, Default)]
struct Word {
    start: u8,
    end: u8,
    text: Text,
    score: i32,
}

#[derive(Clone, Copy, Default)]
struct WordList {
    len: u8,
    items: [Word; 20],
}

impl WordList {
    fn push(&mut self, word: Word) {
        if (self.len as usize) < self.items.len() {
            self.items[self.len as usize] = word;
            self.len += 1;
        }
    }

    fn extend(&mut self, other: &Self) {
        for word in &other.items[..other.len as usize] {
            self.push(*word);
        }
    }

    fn as_slice(&self) -> &[Word] {
        &self.items[..self.len as usize]
    }
}

#[derive(Clone, Copy)]
struct CandidateList {
    len: u8,
    items: [Word; BOARD_WIDTH],
}

impl Default for CandidateList {
    fn default() -> Self {
        Self {
            len: 0,
            items: [Word::default(); BOARD_WIDTH],
        }
    }
}

impl CandidateList {
    fn push(&mut self, word: Word) {
        self.items[self.len as usize] = word;
        self.len += 1;
    }

    fn as_slice(&self) -> &[Word] {
        &self.items[..self.len as usize]
    }
}

#[derive(Clone, Copy, Default)]
struct TextList {
    len: u8,
    items: [Text; 4],
}

impl TextList {
    fn push_unique(&mut self, text: Text) {
        if self.items[..self.len as usize].contains(&text) || self.len as usize >= self.items.len()
        {
            return;
        }
        self.items[self.len as usize] = text;
        self.len += 1;
    }

    fn as_slice(&self) -> &[Text] {
        &self.items[..self.len as usize]
    }
}

#[derive(Clone, Copy)]
struct SearchPiece {
    piece: u8,
    letters: [u8; 4],
}

#[derive(Clone, Copy)]
struct Placement {
    board: Board,
    letter_shift: u8,
    rotation: u8,
    row: i8,
    col: i8,
    score: i32,
    lines: u8,
    words: WordList,
}

#[derive(Clone, Copy)]
struct Evaluation {
    #[cfg(not(target_arch = "wasm32"))]
    heights: [i32; BOARD_WIDTH],
    #[cfg(not(target_arch = "wasm32"))]
    holes: i32,
    #[cfg(not(target_arch = "wasm32"))]
    buried_depth: i32,
    #[cfg(not(target_arch = "wasm32"))]
    aggregate_height: i32,
    #[cfg(not(target_arch = "wasm32"))]
    maximum_height: i32,
    #[cfg(not(target_arch = "wasm32"))]
    bumpiness: i32,
    #[cfg(not(target_arch = "wasm32"))]
    wells: i32,
    #[cfg(not(target_arch = "wasm32"))]
    word_potential: f64,
    value: f64,
    setup_words: TextList,
}

#[derive(Clone, Copy)]
struct SearchState {
    board: Board,
    score: i32,
    lines: u16,
    root_index: u16,
    value: f64,
    setup_words: TextList,
}

struct SearchResult {
    root: Placement,
    immediate_words: WordList,
    projected_score: i32,
    projected_lines: u16,
    setup_words: TextList,
    depth: u8,
    nodes: u32,
    evaluation: i32,
}

type LeafReranker<'a> = &'a dyn Fn(&Board, u16) -> f64;

fn cell_letter(cell: u8) -> u8 {
    cell & 0x1f
}

fn cell_piece(cell: u8) -> u8 {
    cell >> 5
}

fn valid_cell(cell: u8) -> bool {
    cell == 0 || ((1..=26).contains(&cell_letter(cell)) && (1..=7).contains(&cell_piece(cell)))
}

fn is_word(lexicon: &[u32], text: &[u8]) -> bool {
    if text.len() < 2 || lexicon.is_empty() {
        return false;
    }
    let mut node_index = (lexicon[0] & ARC_MASK) as usize;
    let mut accepts = false;
    for (letter_index, &tile) in text.iter().enumerate() {
        let mut found = false;
        let mut index = node_index;
        while index < lexicon.len() {
            let node = lexicon[index];
            if (node >> 24) as u8 == tile {
                found = true;
                accepts = node & ACCEPTS != 0;
                node_index = (node & ARC_MASK) as usize;
                break;
            }
            if node & IS_END != 0 {
                break;
            }
            index += 1;
        }
        if !found || (letter_index + 1 < text.len() && node_index == 0) {
            return false;
        }
    }
    accepts
}

fn score_word(text: &[u8]) -> i32 {
    let raw_score: i32 = text.iter().map(|&tile| LETTER_VALUES[tile as usize]).sum();
    raw_score * text.len() as i32 * text.len() as i32
}

fn analyze_row(lexicon: &[u32], row: &[u8; BOARD_WIDTH]) -> WordList {
    let mut candidates = [CandidateList::default(); BOARD_WIDTH];
    for start in 0..BOARD_WIDTH {
        if row[start] == 0 {
            continue;
        }
        let mut changed_piece = false;
        let mut letters = [0u8; BOARD_WIDTH];
        for end in start..BOARD_WIDTH {
            if row[end] == 0 {
                break;
            }
            letters[end - start] = cell_letter(row[end]);
            if end > start && cell_piece(row[end]) != cell_piece(row[end - 1]) {
                changed_piece = true;
            }
            let length = end - start + 1;
            if length < 2 || !changed_piece || !is_word(lexicon, &letters[..length]) {
                continue;
            }
            let score = score_word(&letters[..length]);
            if score < MINIMUM_WORD_SCORE {
                continue;
            }
            candidates[start].push(Word {
                start: start as u8,
                end: end as u8,
                text: Text::from_slice(&letters[..length]),
                score,
            });
        }
    }

    let mut best_score = [0i32; BOARD_WIDTH + 1];
    let mut choice = [None; BOARD_WIDTH];
    for position in (0..BOARD_WIDTH).rev() {
        best_score[position] = best_score[position + 1];
        for &candidate in candidates[position].as_slice() {
            let total = candidate.score + best_score[candidate.end as usize + 1];
            if total > best_score[position] {
                best_score[position] = total;
                choice[position] = Some(candidate);
            }
        }
    }

    let mut words = WordList::default();
    let mut position = 0;
    while position < BOARD_WIDTH {
        if let Some(candidate) = choice[position]
            && candidate.score + best_score[candidate.end as usize + 1] == best_score[position]
        {
            words.push(candidate);
            position = candidate.end as usize + 1;
        } else {
            position += 1;
        }
    }
    words
}

fn collides(board: &Board, piece: u8, rotation: u8, row: i8, col: i8) -> bool {
    for block in &ROTATIONS[piece as usize][rotation as usize] {
        let board_row = row + block.y;
        let board_col = col + block.x;
        if board_row < 0
            || board_row >= BOARD_HEIGHT as i8
            || board_col < 0
            || board_col >= BOARD_WIDTH as i8
            || board[board_row as usize * BOARD_WIDTH + board_col as usize] != 0
        {
            return true;
        }
    }
    false
}

fn unique_cycles(letters: [u8; 4]) -> ([([u8; 4], u8); 4], usize) {
    let mut cycles = [([0u8; 4], 0u8); 4];
    let mut count = 0;
    for shift in 0..4 {
        let mut shifted = [0u8; 4];
        for index in 0..4 {
            shifted[index] = letters[(index + shift) % 4];
        }
        if cycles[..count].iter().any(|(cycle, _)| *cycle == shifted) {
            continue;
        }
        cycles[count] = (shifted, shift as u8);
        count += 1;
    }
    (cycles, count)
}

fn simulate_placements(lexicon: &[u32], board: &Board, piece: SearchPiece) -> Vec<Placement> {
    let mut placements = Vec::with_capacity(144);
    let (cycles, cycle_count) = unique_cycles(piece.letters);
    for &(letters, letter_shift) in &cycles[..cycle_count] {
        for rotation in 0..4u8 {
            let blocks = &ROTATIONS[piece.piece as usize][rotation as usize];
            let mut min_x = i8::MAX;
            let mut max_x = i8::MIN;
            let mut min_y = i8::MAX;
            for block in blocks {
                min_x = min_x.min(block.x);
                max_x = max_x.max(block.x);
                min_y = min_y.min(block.y);
            }
            for col in -min_x..BOARD_WIDTH as i8 - max_x {
                let mut row = -min_y;
                if collides(board, piece.piece, rotation, row, col) {
                    continue;
                }
                while !collides(board, piece.piece, rotation, row + 1, col) {
                    row += 1;
                }

                let mut placed_board = *board;
                for block in blocks {
                    let board_row = (row + block.y) as usize;
                    let board_col = (col + block.x) as usize;
                    placed_board[board_row * BOARD_WIDTH + board_col] =
                        letters[block.letter_index as usize] | ((piece.piece + 1) << 5);
                }

                let mut cleared = [false; BOARD_HEIGHT];
                let mut clearing_count = 0u8;
                let mut words = WordList::default();
                for (board_row, is_cleared) in cleared.iter_mut().enumerate() {
                    let start = board_row * BOARD_WIDTH;
                    if placed_board[start..start + BOARD_WIDTH]
                        .iter()
                        .all(|&cell| cell != 0)
                    {
                        *is_cleared = true;
                        clearing_count += 1;
                        let row_cells: &[u8; BOARD_WIDTH] = placed_board
                            [start..start + BOARD_WIDTH]
                            .try_into()
                            .expect("row width");
                        words.extend(&analyze_row(lexicon, row_cells));
                    }
                }
                let score = words.as_slice().iter().map(|word| word.score).sum();

                let mut collapsed = [0u8; BOARD_CELLS];
                let mut destination_row = clearing_count as usize;
                for (source_row, &is_cleared) in cleared.iter().enumerate() {
                    if is_cleared {
                        continue;
                    }
                    let source = source_row * BOARD_WIDTH;
                    let destination = destination_row * BOARD_WIDTH;
                    collapsed[destination..destination + BOARD_WIDTH]
                        .copy_from_slice(&placed_board[source..source + BOARD_WIDTH]);
                    destination_row += 1;
                }

                placements.push(Placement {
                    board: collapsed,
                    letter_shift,
                    rotation,
                    row,
                    col,
                    score,
                    lines: clearing_count,
                    words,
                });
            }
        }
    }
    placements
}

fn evaluate_board(lexicon: &[u32], board: &Board) -> Evaluation {
    let mut heights = [0i32; BOARD_WIDTH];
    let mut holes = 0i32;
    let mut buried_depth = 0i32;
    for col in 0..BOARD_WIDTH {
        let mut first_filled = BOARD_HEIGHT;
        let mut blocks_above = 0i32;
        for row in 0..BOARD_HEIGHT {
            if board[row * BOARD_WIDTH + col] != 0 {
                if first_filled == BOARD_HEIGHT {
                    first_filled = row;
                }
                blocks_above += 1;
            } else if first_filled != BOARD_HEIGHT {
                holes += 1;
                buried_depth += blocks_above;
            }
        }
        heights[col] = (BOARD_HEIGHT - first_filled) as i32;
    }

    let aggregate_height: i32 = heights.iter().sum();
    let maximum_height = *heights.iter().max().unwrap_or(&0);
    let bumpiness: i32 = (1..BOARD_WIDTH)
        .map(|index| (heights[index] - heights[index - 1]).abs())
        .sum();
    let mut wells = 0i32;
    for col in 0..BOARD_WIDTH {
        let left = if col == 0 {
            BOARD_HEIGHT as i32
        } else {
            heights[col - 1]
        };
        let right = if col + 1 == BOARD_WIDTH {
            BOARD_HEIGHT as i32
        } else {
            heights[col + 1]
        };
        wells += 0.max(left.min(right) - heights[col]);
    }

    let mut word_potential = 0f64;
    let mut word_candidates: Vec<(Text, f64)> = Vec::with_capacity(64);
    for row in 0..BOARD_HEIGHT {
        let start = row * BOARD_WIDTH;
        let row_cells: &[u8; BOARD_WIDTH] = board[start..start + BOARD_WIDTH]
            .try_into()
            .expect("row width");
        let fullness =
            row_cells.iter().filter(|&&cell| cell != 0).count() as f64 / BOARD_WIDTH as f64;
        if fullness < 0.2 || fullness == 1.0 {
            continue;
        }
        for word in analyze_row(lexicon, row_cells).as_slice() {
            let value = word.score as f64 * (0.18 + 1.12 * fullness.powi(4));
            word_potential += value;
            word_candidates.push((word.text, value));
        }
    }
    word_candidates.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    let mut setup_words = TextList::default();
    for (text, _) in word_candidates {
        setup_words.push_unique(text);
        if setup_words.len == 4 {
            break;
        }
    }

    let value = word_potential
        - holes as f64 * 285.0
        - buried_depth as f64 * 34.0
        - aggregate_height as f64 * 4.5
        - (maximum_height * maximum_height) as f64 * 1.7
        - bumpiness as f64 * 8.0
        - wells as f64 * 3.0;
    Evaluation {
        #[cfg(not(target_arch = "wasm32"))]
        heights,
        #[cfg(not(target_arch = "wasm32"))]
        holes,
        #[cfg(not(target_arch = "wasm32"))]
        buried_depth,
        #[cfg(not(target_arch = "wasm32"))]
        aggregate_height,
        #[cfg(not(target_arch = "wasm32"))]
        maximum_height,
        #[cfg(not(target_arch = "wasm32"))]
        bumpiness,
        #[cfg(not(target_arch = "wasm32"))]
        wells,
        #[cfg(not(target_arch = "wasm32"))]
        word_potential,
        value,
        setup_words,
    }
}

fn compare_states(left: &SearchState, right: &SearchState) -> Ordering {
    right
        .value
        .partial_cmp(&left.value)
        .unwrap_or(Ordering::Equal)
}

fn js_round(value: f64) -> i32 {
    (value + 0.5).floor() as i32
}

fn search(
    lexicon: &[u32],
    board: Board,
    current_lines: u8,
    sequence: &[SearchPiece],
    beam_width: usize,
    leaf_reranker: Option<LeafReranker<'_>>,
    leaf_rerank_candidates: usize,
) -> Option<SearchResult> {
    let root_placements = simulate_placements(lexicon, &board, sequence[0]);
    if root_placements.is_empty() {
        return None;
    }
    let mut nodes = root_placements.len() as u32;
    let mut reached_depth = 1u8;
    let mut frontier = Vec::with_capacity(root_placements.len());
    for (root_index, placement) in root_placements.iter().enumerate() {
        let evaluation = evaluate_board(lexicon, &placement.board);
        frontier.push(SearchState {
            board: placement.board,
            score: placement.score,
            lines: placement.lines as u16,
            root_index: root_index as u16,
            value: placement.score as f64 * 1.25 - placement.lines as f64 * 90.0 + evaluation.value,
            setup_words: evaluation.setup_words,
        });
    }
    frontier.sort_by(compare_states);
    frontier.truncate(beam_width);
    let mut finished: Vec<SearchState> = Vec::new();

    for (ply, &piece) in sequence.iter().enumerate().skip(1) {
        if frontier.is_empty() {
            break;
        }
        let mut next: Vec<SearchState> = Vec::with_capacity(frontier.len() * 120);
        for state in &frontier {
            if current_lines as u16 + state.lines >= MAX_LINES {
                finished.push(*state);
                continue;
            }
            for placement in simulate_placements(lexicon, &state.board, piece) {
                nodes += 1;
                let score = state.score + placement.score;
                let lines = state.lines + placement.lines as u16;
                let evaluation = evaluate_board(lexicon, &placement.board);
                next.push(SearchState {
                    board: placement.board,
                    score,
                    lines,
                    root_index: state.root_index,
                    value: score as f64 * 1.25 - lines as f64 * 90.0 + evaluation.value,
                    setup_words: evaluation.setup_words,
                });
            }
        }
        if next.is_empty() {
            break;
        }
        reached_depth = (ply + 1) as u8;
        next.sort_by(compare_states);
        next.truncate(beam_width);
        frontier = next;
    }

    frontier.extend(finished);
    frontier.sort_by(compare_states);
    if let Some(rerank) = leaf_reranker {
        frontier.truncate(leaf_rerank_candidates.clamp(1, beam_width));
        for state in &mut frontier {
            state.value += rerank(&state.board, state.lines);
        }
        frontier.sort_by(compare_states);
    }
    let best = *frontier.first()?;
    let root = root_placements[best.root_index as usize];
    let root_evaluation = evaluate_board(lexicon, &root.board);
    let setup_words = if root_evaluation.setup_words.len > 0 {
        root_evaluation.setup_words
    } else {
        best.setup_words
    };
    Some(SearchResult {
        root,
        immediate_words: root.words,
        projected_score: best.score,
        projected_lines: best.lines,
        setup_words,
        depth: reached_depth,
        nodes,
        evaluation: js_round(best.value),
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    use super::*;

    pub use crate::fragment_model::{FragmentModelPair, FragmentPrediction, FragmentResidual};

    pub const WIDTH: usize = BOARD_WIDTH;
    pub const HEIGHT: usize = BOARD_HEIGHT;
    pub const CELL_COUNT: usize = BOARD_CELLS;
    pub type PackedBoard = [u8; CELL_COUNT];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Piece {
        pub kind: u8,
        pub letters: [u8; 4],
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ScoredWord {
        pub start: u8,
        pub end: u8,
        pub text: String,
        pub score: i32,
    }

    #[derive(Clone, Debug)]
    pub struct BoardEvaluation {
        pub heights: [i32; WIDTH],
        pub holes: i32,
        pub buried_depth: i32,
        pub aggregate_height: i32,
        pub maximum_height: i32,
        pub bumpiness: i32,
        pub wells: i32,
        pub word_potential: f64,
        pub setup_words: Vec<String>,
        pub heuristic_value: f64,
    }

    #[derive(Clone, Debug)]
    pub struct SearchOutcome {
        pub board: PackedBoard,
        pub letter_shift: u8,
        pub rotation: u8,
        pub row: i8,
        pub col: i8,
        pub immediate_score: i32,
        pub immediate_lines: u8,
        pub immediate_words: Vec<ScoredWord>,
        pub projected_score: i32,
        pub projected_lines: u16,
        pub setup_words: Vec<String>,
        pub depth: u8,
        pub nodes: u32,
        pub evaluation: i32,
    }

    pub struct FragmentRerank<'a> {
        pub leaf_visible: &'a [Piece],
        pub model: &'a FragmentModelPair,
        pub lexicon: u8,
        pub weight: f64,
        pub candidates: usize,
    }

    pub struct Strategy {
        lexicon: Vec<u32>,
    }

    fn text_to_string(text: &Text) -> String {
        text.as_slice()
            .iter()
            .map(|&letter| char::from(b'A' + letter - 1))
            .collect()
    }

    fn words_to_vec(words: &WordList) -> Vec<ScoredWord> {
        words
            .as_slice()
            .iter()
            .map(|word| ScoredWord {
                start: word.start,
                end: word.end,
                text: text_to_string(&word.text),
                score: word.score,
            })
            .collect()
    }

    fn texts_to_vec(texts: &TextList) -> Vec<String> {
        texts.as_slice().iter().map(text_to_string).collect()
    }

    fn result_to_outcome(result: SearchResult) -> SearchOutcome {
        SearchOutcome {
            board: result.root.board,
            letter_shift: result.root.letter_shift,
            rotation: result.root.rotation,
            row: result.root.row,
            col: result.root.col,
            immediate_score: result.root.score,
            immediate_lines: result.root.lines,
            immediate_words: words_to_vec(&result.immediate_words),
            projected_score: result.projected_score,
            projected_lines: result.projected_lines,
            setup_words: texts_to_vec(&result.setup_words),
            depth: result.depth,
            nodes: result.nodes,
            evaluation: result.evaluation,
        }
    }

    fn valid_sequence(sequence: &[Piece]) -> bool {
        !sequence.is_empty()
            && sequence.len() <= 5
            && sequence.iter().all(|piece| {
                piece.kind < 7 && piece.letters.iter().all(|letter| (1..=26).contains(letter))
            })
    }

    fn to_search_pieces(sequence: &[Piece]) -> Vec<SearchPiece> {
        sequence
            .iter()
            .map(|piece| SearchPiece {
                piece: piece.kind,
                letters: piece.letters,
            })
            .collect()
    }

    impl Strategy {
        pub fn from_kwg_bytes(bytes: &[u8]) -> Result<Self, String> {
            if bytes.len() < 8 || !bytes.len().is_multiple_of(4) {
                return Err("KWG must contain at least two little-endian u32 nodes".to_string());
            }
            let lexicon = bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte KWG node")))
                .collect();
            Ok(Self { lexicon })
        }

        pub fn is_word(&self, letters: &[u8]) -> bool {
            super::is_word(&self.lexicon, letters)
        }

        pub fn can_spawn(&self, board: &PackedBoard, piece: u8) -> bool {
            piece < 7
                && !board.iter().any(|&cell| !valid_cell(cell))
                && (!collides(board, piece, 0, 1, 3) || !collides(board, piece, 0, 0, 3))
        }

        pub fn evaluate(&self, board: &PackedBoard) -> Option<BoardEvaluation> {
            if board.iter().any(|&cell| !valid_cell(cell)) {
                return None;
            }
            let evaluation = evaluate_board(&self.lexicon, board);
            Some(BoardEvaluation {
                heights: evaluation.heights,
                holes: evaluation.holes,
                buried_depth: evaluation.buried_depth,
                aggregate_height: evaluation.aggregate_height,
                maximum_height: evaluation.maximum_height,
                bumpiness: evaluation.bumpiness,
                wells: evaluation.wells,
                word_potential: evaluation.word_potential,
                setup_words: texts_to_vec(&evaluation.setup_words),
                heuristic_value: evaluation.value,
            })
        }

        pub fn find_best_move(
            &self,
            board: &PackedBoard,
            current_lines: u8,
            sequence: &[Piece],
            beam_width: usize,
        ) -> Option<SearchOutcome> {
            if board.iter().any(|&cell| !valid_cell(cell)) || !valid_sequence(sequence) {
                return None;
            }
            let search_pieces = to_search_pieces(sequence);
            let result = search(
                &self.lexicon,
                *board,
                current_lines,
                &search_pieces,
                beam_width.clamp(12, 160),
                None,
                0,
            )?;
            Some(result_to_outcome(result))
        }

        pub fn find_best_move_with_fragment_model(
            &self,
            board: &PackedBoard,
            current_lines: u8,
            sequence: &[Piece],
            beam_width: usize,
            reranker: FragmentRerank<'_>,
        ) -> Option<SearchOutcome> {
            if board.iter().any(|&cell| !valid_cell(cell))
                || !valid_sequence(sequence)
                || reranker.leaf_visible.len() != 5
                || !valid_sequence(reranker.leaf_visible)
                || reranker.lexicon > 1
                || !reranker.weight.is_finite()
            {
                return None;
            }
            let search_pieces = to_search_pieces(sequence);
            let leaf_pieces = to_search_pieces(reranker.leaf_visible);
            let rerank = |leaf_board: &Board, projected_lines: u16| {
                let leaf_lines =
                    (u16::from(current_lines) + projected_lines).min(u16::from(u8::MAX)) as u8;
                reranker
                    .model
                    .residual(leaf_board, &leaf_pieces, leaf_lines, reranker.lexicon)
                    .value()
                    * reranker.weight
            };
            let result = search(
                &self.lexicon,
                *board,
                current_lines,
                &search_pieces,
                beam_width.clamp(12, 160),
                Some(&rerank),
                reranker.candidates,
            )?;
            Some(result_to_outcome(result))
        }
    }
}

struct Writer<'a> {
    output: &'a mut [u8],
    position: usize,
    failed: bool,
}

impl<'a> Writer<'a> {
    fn new(output: &'a mut [u8]) -> Self {
        Self {
            output,
            position: 0,
            failed: false,
        }
    }

    fn bytes(&mut self, bytes: &[u8]) {
        if self.position + bytes.len() > self.output.len() {
            self.failed = true;
            return;
        }
        self.output[self.position..self.position + bytes.len()].copy_from_slice(bytes);
        self.position += bytes.len();
    }

    fn u8(&mut self, value: u8) {
        self.bytes(&[value]);
    }

    fn i8(&mut self, value: i8) {
        self.u8(value as u8);
    }

    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn i32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }

    fn text(&mut self, text: &Text) {
        self.u8(text.len);
        self.bytes(text.as_slice());
    }
}

fn lexicon_from_raw<'a>(pointer: u32, nodes: u32) -> Option<&'a [u32]> {
    if pointer == 0 || nodes < 2 {
        return None;
    }
    Some(unsafe { slice::from_raw_parts(pointer as *const u32, nodes as usize) })
}

fn bytes_from_raw<'a>(pointer: u32, length: u32) -> Option<&'a [u8]> {
    if pointer == 0 || length == 0 {
        return None;
    }
    Some(unsafe { slice::from_raw_parts(pointer as *const u8, length as usize) })
}

fn output_from_raw<'a>(pointer: u32, capacity: u32) -> Option<&'a mut [u8]> {
    if pointer == 0 || capacity == 0 {
        return None;
    }
    Some(unsafe { slice::from_raw_parts_mut(pointer as *mut u8, capacity as usize) })
}

fn write_search_result(result: &SearchResult, output: &mut [u8]) -> u32 {
    let mut writer = Writer::new(output);
    writer.u8(1);
    writer.u8(result.root.letter_shift);
    writer.u8(result.root.rotation);
    writer.i8(result.root.row);
    writer.i8(result.root.col);
    writer.u8(result.root.lines);
    writer.u8(result.depth);
    writer.u8(result.immediate_words.len);
    writer.u8(result.setup_words.len);
    writer.u8(0);
    writer.i32(result.root.score);
    writer.i32(result.projected_score);
    writer.u16(result.projected_lines);
    writer.u32(result.nodes);
    writer.i32(result.evaluation);
    for word in result.immediate_words.as_slice() {
        writer.text(&word.text);
    }
    for text in result.setup_words.as_slice() {
        writer.text(text);
    }
    if writer.failed {
        0
    } else {
        writer.position as u32
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kv_alloc(size: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    let layout = match Layout::from_size_align(size as usize, 8) {
        Ok(layout) => layout,
        Err(_) => return 0,
    };
    unsafe { alloc(layout) as u32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn kv_dealloc(pointer: u32, size: u32) {
    if pointer == 0 || size == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size as usize, 8) {
        unsafe { dealloc(pointer as *mut u8, layout) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kv_is_word(
    lexicon_pointer: u32,
    lexicon_nodes: u32,
    word_pointer: u32,
    word_length: u32,
) -> u32 {
    let Some(lexicon) = lexicon_from_raw(lexicon_pointer, lexicon_nodes) else {
        return 0;
    };
    let Some(word) = bytes_from_raw(word_pointer, word_length) else {
        return 0;
    };
    is_word(lexicon, word) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn kv_analyze_row(
    lexicon_pointer: u32,
    lexicon_nodes: u32,
    row_pointer: u32,
    output_pointer: u32,
    output_capacity: u32,
) -> u32 {
    let Some(lexicon) = lexicon_from_raw(lexicon_pointer, lexicon_nodes) else {
        return 0;
    };
    let Some(row_bytes) = bytes_from_raw(row_pointer, BOARD_WIDTH as u32) else {
        return 0;
    };
    let Ok(row) = <&[u8; BOARD_WIDTH]>::try_from(row_bytes) else {
        return 0;
    };
    if row.iter().any(|&cell| !valid_cell(cell)) {
        return 0;
    }
    let Some(output) = output_from_raw(output_pointer, output_capacity) else {
        return 0;
    };
    let words = analyze_row(lexicon, row);
    let mut writer = Writer::new(output);
    writer.u8(words.len);
    for word in words.as_slice() {
        writer.u8(word.start);
        writer.u8(word.end);
        writer.i32(word.score);
        writer.text(&word.text);
    }
    if writer.failed {
        0
    } else {
        writer.position as u32
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kv_find_best_move(
    lexicon_pointer: u32,
    lexicon_nodes: u32,
    input_pointer: u32,
    input_length: u32,
    output_pointer: u32,
    output_capacity: u32,
) -> u32 {
    let Some(lexicon) = lexicon_from_raw(lexicon_pointer, lexicon_nodes) else {
        return 0;
    };
    let Some(input) = bytes_from_raw(input_pointer, input_length) else {
        return 0;
    };
    if input.len() < 226 || input[0] != 1 {
        return 0;
    }
    let depth = input[1].clamp(1, 5) as usize;
    let beam_width = input[2].clamp(12, 160) as usize;
    let current_lines = input[3];
    let sequence_length = input[4] as usize;
    if sequence_length != depth || input.len() < 226 + sequence_length * 5 {
        return 0;
    }
    let mut board = [0u8; BOARD_CELLS];
    board.copy_from_slice(&input[6..226]);
    if board.iter().any(|&cell| !valid_cell(cell)) {
        return 0;
    }
    let mut sequence = Vec::with_capacity(sequence_length);
    for index in 0..sequence_length {
        let offset = 226 + index * 5;
        let piece = input[offset];
        if piece >= 7 {
            return 0;
        }
        let mut letters = [0u8; 4];
        letters.copy_from_slice(&input[offset + 1..offset + 5]);
        if letters.iter().any(|&letter| !(1..=26).contains(&letter)) {
            return 0;
        }
        sequence.push(SearchPiece { piece, letters });
    }

    let Some(result) = search(
        lexicon,
        board,
        current_lines,
        &sequence,
        beam_width,
        None,
        0,
    ) else {
        return 0;
    };
    let Some(output) = output_from_raw(output_pointer, output_capacity) else {
        return 0;
    };
    write_search_result(&result, output)
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn kv_fragment_model_pair_new(
    full_pointer: u32,
    full_length: u32,
    context_pointer: u32,
    context_length: u32,
) -> u32 {
    let Some(full) = bytes_from_raw(full_pointer, full_length) else {
        return 0;
    };
    let Some(context) = bytes_from_raw(context_pointer, context_length) else {
        return 0;
    };
    let Ok(model) = fragment_model::FragmentModelPair::from_bytes(full, context) else {
        return 0;
    };
    FRAGMENT_MODELS.with(|models| {
        let mut models = models.borrow_mut();
        if let Some(index) = models.iter().position(Option::is_none) {
            models[index] = Some(model);
            (index + 1) as u32
        } else {
            models.push(Some(model));
            models.len() as u32
        }
    })
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn kv_fragment_model_pair_free(handle: u32) {
    let Some(index) = handle.checked_sub(1).map(|value| value as usize) else {
        return;
    };
    FRAGMENT_MODELS.with(|models| {
        if let Some(model) = models.borrow_mut().get_mut(index) {
            *model = None;
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn kv_find_best_move_with_fragment_model(
    lexicon_pointer: u32,
    lexicon_nodes: u32,
    model_handle: u32,
    input_pointer: u32,
    input_length: u32,
    output_pointer: u32,
    output_capacity: u32,
) -> u32 {
    const DEPTH: usize = 3;
    const LEAF_VISIBLE: usize = 5;
    const FRAGMENT_WEIGHT: f64 = 0.25;
    const FRAGMENT_CANDIDATES: usize = 6;

    let Some(lexicon) = lexicon_from_raw(lexicon_pointer, lexicon_nodes) else {
        return 0;
    };
    let Some(model_index) = model_handle.checked_sub(1).map(|value| value as usize) else {
        return 0;
    };
    let Some(input) = bytes_from_raw(input_pointer, input_length) else {
        return 0;
    };
    if input.len() < 226 + (DEPTH + LEAF_VISIBLE) * 5
        || input[0] != 2
        || input[1] as usize != DEPTH
        || input[4] as usize != DEPTH
        || input[5] > 1
    {
        return 0;
    }
    let beam_width = input[2].clamp(12, 160) as usize;
    let current_lines = input[3];
    let lexicon_id = input[5];
    let mut board = [0u8; BOARD_CELLS];
    board.copy_from_slice(&input[6..226]);
    if board.iter().any(|&cell| !valid_cell(cell)) {
        return 0;
    }

    let mut pieces = Vec::with_capacity(DEPTH + LEAF_VISIBLE);
    for index in 0..DEPTH + LEAF_VISIBLE {
        let offset = 226 + index * 5;
        let piece = input[offset];
        if piece >= 7 {
            return 0;
        }
        let mut letters = [0u8; 4];
        letters.copy_from_slice(&input[offset + 1..offset + 5]);
        if letters.iter().any(|&letter| !(1..=26).contains(&letter)) {
            return 0;
        }
        pieces.push(SearchPiece { piece, letters });
    }
    let (sequence, leaf_visible) = pieces.split_at(DEPTH);
    FRAGMENT_MODELS.with(|models| {
        let models = models.borrow();
        let Some(model) = models.get(model_index).and_then(Option::as_ref) else {
            return 0;
        };
        let rerank = |leaf_board: &Board, projected_lines: u16| {
            let leaf_lines =
                (u16::from(current_lines) + projected_lines).min(u16::from(u8::MAX)) as u8;
            model
                .residual(leaf_board, leaf_visible, leaf_lines, lexicon_id)
                .value()
                * FRAGMENT_WEIGHT
        };
        let Some(result) = search(
            lexicon,
            board,
            current_lines,
            sequence,
            beam_width,
            Some(&rerank),
            FRAGMENT_CANDIDATES,
        ) else {
            return 0;
        };
        let Some(output) = output_from_raw(output_pointer, output_capacity) else {
            return 0;
        };
        write_search_result(&result, output)
    })
}
