use kvadrat_strategy::native::{
    FragmentModelPair, FragmentRerank, HEIGHT, PackedBoard, Piece, RootRankModel, RootRerank,
    Strategy, WIDTH,
};
use std::collections::HashSet;

const CSW24: &[u8] = include_bytes!("../../public/data/CSW24.kwg");
const FRAGMENT_FULL: &[u8] = include_bytes!("../../public/data/models/fragment-full-v4.kfm");
const FRAGMENT_CONTEXT: &[u8] = include_bytes!("../../public/data/models/fragment-context-v4.kfm");

fn strategy() -> Strategy {
    Strategy::from_kwg_bytes(CSW24).expect("checked CSW24 fixture")
}

fn cell(letter: u8, piece: u8) -> u8 {
    (letter - b'A' + 1) | ((piece + 1) << 5)
}

fn piece(kind: u8, letters: &[u8; 4]) -> Piece {
    Piece {
        kind,
        letters: letters.map(|letter| letter - b'A' + 1),
    }
}

fn sequence() -> [Piece; 3] {
    [piece(0, b"FAVE"), piece(3, b"RATE"), piece(5, b"LION")]
}

fn visible() -> [Piece; 5] {
    [
        piece(2, b"STAR"),
        piece(4, b"EONS"),
        piece(6, b"TILE"),
        piece(1, b"WORD"),
        piece(3, b"GAME"),
    ]
}

fn zero_root_ranker() -> RootRankModel {
    const HEADER_BYTES: usize = 32;
    const FLOATS: usize = 15_349;
    let mut bytes = vec![0; HEADER_BYTES + FLOATS * 4];
    bytes[..8].copy_from_slice(b"KVRK1\0\0\0");
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&(FLOATS as u32).to_le_bytes());
    bytes[16..20].copy_from_slice(&4u32.to_le_bytes());
    bytes[20..24].copy_from_slice(&10u32.to_le_bytes());
    RootRankModel::from_bytes(&bytes).expect("valid zero ranker")
}

#[test]
fn words_require_a_visible_piece_boundary() {
    let strategy = strategy();
    let mut row = [0u8; WIDTH];
    row[0] = cell(b'F', 0);
    row[1] = cell(b'A', 0);
    row[2] = cell(b'V', 1);

    let words = strategy.analyze_row(&row).expect("valid row");
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].text, "FAV");
    assert_eq!(words[0].score, 81);

    row[2] = cell(b'V', 0);
    assert!(strategy.analyze_row(&row).expect("valid row").is_empty());

    row[2] = 31 | (1 << 5);
    assert!(strategy.analyze_row(&row).is_none());
}

#[test]
fn row_scoring_selects_non_overlapping_words() {
    let strategy = strategy();
    let mut row = [0u8; WIDTH];
    for (offset, &letter) in b"FAV".iter().enumerate() {
        row[offset] = cell(letter, u8::from(offset == 2));
        row[offset + 4] = cell(letter, 2 + u8::from(offset == 2));
    }
    let words = strategy.analyze_row(&row).expect("valid row");
    assert_eq!(
        words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>(),
        ["FAV", "FAV"]
    );
    assert_eq!(words.iter().map(|word| word.score).sum::<i32>(), 162);
}

#[test]
fn placements_clear_and_collapse_a_completed_line() {
    let strategy = strategy();
    let mut board = [0u8; WIDTH * HEIGHT];
    for col in 0..6 {
        board[(HEIGHT - 1) * WIDTH + col] = cell(b'A', 1);
    }
    let candidates = strategy.root_candidates(&board, 0, &[piece(0, b"FAVE")], 160, 160);
    let clearing = candidates
        .iter()
        .find(|candidate| candidate.immediate_lines == 1)
        .expect("horizontal I placement should complete the row");
    assert!(clearing.board.iter().all(|&value| value == 0));
}

#[test]
fn board_evaluation_counts_holes_and_rejects_invalid_cells() {
    let strategy = strategy();
    let mut board = [0u8; WIDTH * HEIGHT];
    board[(HEIGHT - 2) * WIDTH] = cell(b'A', 0);
    let evaluation = strategy.evaluate(&board).expect("valid board");
    assert_eq!(evaluation.heights[0], 2);
    assert_eq!(evaluation.holes, 1);
    assert_eq!(evaluation.buried_depth, 1);
    assert_eq!(evaluation.aggregate_height, 2);
    assert_eq!(evaluation.maximum_height, 2);
    assert!(evaluation.heuristic_value < 0.0);

    board[0] = 31 | (1 << 5);
    assert!(strategy.evaluate(&board).is_none());
    assert!(!strategy.can_spawn(&board, 0));
}

#[test]
fn native_search_is_deterministic_and_returns_unique_roots() {
    let strategy = strategy();
    let board = [0u8; WIDTH * HEIGHT];
    let sequence = sequence();
    let first = strategy
        .find_best_move(&board, 0, &sequence, 24)
        .expect("legal opening move");
    let second = strategy
        .find_best_move(&board, 0, &sequence, 24)
        .expect("repeat opening move");
    assert_eq!(
        (
            first.board,
            first.letter_shift,
            first.rotation,
            first.row,
            first.col
        ),
        (
            second.board,
            second.letter_shift,
            second.rotation,
            second.row,
            second.col
        ),
    );
    assert_eq!(first.depth, 3);
    assert!(first.nodes > 1_000);

    let candidates = strategy.root_candidates(&board, 0, &sequence, 64, 12);
    assert!(!candidates.is_empty());
    assert!(candidates.len() <= 12);
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.board)
            .collect::<HashSet<_>>()
            .len(),
        candidates.len(),
    );

    let pooled = strategy
        .find_best_move_with_root_candidates(&board, 0, &sequence, 24, 64, 12)
        .expect("pooled search move");
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.board == pooled.board)
            || pooled.board == first.board
    );
}

#[test]
fn learned_evaluators_run_on_native_search_candidates() {
    let strategy = strategy();
    let board: PackedBoard = [0; WIDTH * HEIGHT];
    let sequence = sequence();
    let visible = visible();
    let fragments = FragmentModelPair::from_bytes(FRAGMENT_FULL, FRAGMENT_CONTEXT)
        .expect("checked fragment fixtures");
    let fragment = strategy
        .find_best_move_with_fragment_model(
            &board,
            0,
            &sequence,
            12,
            FragmentRerank {
                leaf_visible: &visible,
                model: &fragments,
                lexicon: 0,
                weight: 0.25,
                candidates: 6,
            },
        )
        .expect("fragment-reranked move");
    assert_eq!(fragment.depth, 3);
    assert!(fragment.evaluation > i32::MIN);

    let ranker = zero_root_ranker();
    let ranked = strategy
        .find_best_move_with_root_ranker(
            &board,
            0,
            &sequence,
            12,
            RootRerank {
                visible_after_root: &visible,
                model: &ranker,
                lexicon: 0,
                candidate_beam_width: 24,
                candidates: 6,
                correction_weight: 1.0,
            },
        )
        .expect("root-reranked move");
    assert_eq!(ranked.depth, 3);
}

#[test]
fn public_native_api_rejects_malformed_inputs() {
    assert!(Strategy::from_kwg_bytes(&[]).is_err());
    assert!(Strategy::from_kwg_bytes(&[0; 9]).is_err());
    let strategy = strategy();
    let mut board = [0u8; WIDTH * HEIGHT];
    assert!(!strategy.is_word(&[]));
    assert!(!strategy.is_word(&[26, 26, 26, 26]));
    assert!(!strategy.can_spawn(&board, 7));
    assert!(strategy.find_best_move(&board, 0, &[], 24).is_none());
    assert!(
        strategy
            .root_candidates(&board, 0, &sequence(), 24, 0)
            .is_empty()
    );
    board[0] = 31 | (1 << 5);
    assert!(
        strategy
            .find_best_move(&board, 0, &sequence(), 24)
            .is_none()
    );
}
