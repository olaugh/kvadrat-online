use crate::{BOARD_HEIGHT, BOARD_WIDTH, Board, SearchPiece};

const MAGIC: &[u8; 8] = b"KVFM1\0\0\0";
const HEADER_BYTES: usize = 32;
const LETTER_EMBED: usize = 8;
const BOUNDARY_EMBED: usize = 3;
const ROW_NUMERIC: usize = 38;
const ROW_INPUT: usize =
    BOARD_WIDTH * LETTER_EMBED + (BOARD_WIDTH - 1) * BOUNDARY_EMBED + ROW_NUMERIC;
const ROW_HIDDEN: usize = 64;
const ROW_OUTPUT: usize = 16;
const GLOBAL_INPUT: usize = 37;
const HEAD_INPUT: usize = ROW_OUTPUT * 2 + GLOBAL_INPUT;
const HEAD_HIDDEN: usize = 64;
const OUTPUTS: usize = 7;
const FLOAT_COUNT: usize = 15_544;

#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub struct FragmentPrediction {
    pub score_4: f32,
    pub score_8: f32,
    pub score_16: f32,
    pub words_8: f32,
    pub word_length_8: f32,
    pub lines_8: f32,
    pub score_per_line_8: f32,
}

#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub struct FragmentResidual {
    pub score_8: f32,
    pub words_8: f32,
    pub word_length_8: f32,
    pub score_per_line_8: f32,
}

impl FragmentResidual {
    pub fn value(self) -> f64 {
        // Score-per-line is the cleanest learned measure of lexical quality.
        // Word count and raw length remain useful diagnostics, but rewarding
        // them directly produced more low-value words and extra top-outs.
        f64::from(self.score_8 * 0.5 + self.score_per_line_8 * 1.5)
    }
}

#[derive(Clone)]
pub struct FragmentModel {
    letter_embedding: Vec<f32>,
    boundary_embedding: Vec<f32>,
    row_1_weight: Vec<f32>,
    row_1_bias: Vec<f32>,
    row_2_weight: Vec<f32>,
    row_2_bias: Vec<f32>,
    head_1_weight: Vec<f32>,
    head_1_bias: Vec<f32>,
    head_2_weight: Vec<f32>,
    head_2_bias: Vec<f32>,
}

#[derive(Clone)]
pub struct FragmentModelPair {
    full: FragmentModel,
    context: FragmentModel,
}

#[derive(Clone, Copy)]
enum InputMode {
    Full,
    MaskWordInputs,
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let field = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "fragment model header is truncated".to_string())?;
    Ok(u32::from_le_bytes(
        field.try_into().expect("four-byte model field"),
    ))
}

fn take_floats(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    let end = cursor
        .checked_add(count * 4)
        .ok_or_else(|| "fragment model size overflow".to_string())?;
    let source = bytes
        .get(*cursor..end)
        .ok_or_else(|| "fragment model parameters are truncated".to_string())?;
    let values: Vec<f32> = source
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte model weight")))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("fragment model contains a non-finite weight".to_string());
    }
    *cursor = end;
    Ok(values)
}

fn dense(input: &[f32], weight: &[f32], bias: &[f32], output: &mut [f32], relu: bool) {
    for (row, destination) in output.iter_mut().enumerate() {
        let weights = &weight[row * input.len()..(row + 1) * input.len()];
        let value = input
            .iter()
            .zip(weights)
            .fold(bias[row], |sum, (left, right)| sum + left * right);
        *destination = if relu { value.max(0.0) } else { value };
    }
}

fn inverse(value: f32, scale: f32) -> f32 {
    value.clamp(0.0, 8.0).exp_m1() * scale
}

impl FragmentModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != HEADER_BYTES + FLOAT_COUNT * 4 || bytes.get(..8) != Some(MAGIC) {
            return Err("fragment model has an invalid magic or length".to_string());
        }
        if read_u32(bytes, 8)? != 1
            || read_u32(bytes, 12)? as usize != FLOAT_COUNT
            || read_u32(bytes, 16)? as usize != OUTPUTS
            || read_u32(bytes, 20)? != 10
        {
            return Err("fragment model architecture is unsupported".to_string());
        }
        let mut cursor = HEADER_BYTES;
        let model = Self {
            letter_embedding: take_floats(bytes, &mut cursor, 27 * LETTER_EMBED)?,
            boundary_embedding: take_floats(bytes, &mut cursor, 3 * BOUNDARY_EMBED)?,
            row_1_weight: take_floats(bytes, &mut cursor, ROW_HIDDEN * ROW_INPUT)?,
            row_1_bias: take_floats(bytes, &mut cursor, ROW_HIDDEN)?,
            row_2_weight: take_floats(bytes, &mut cursor, ROW_OUTPUT * ROW_HIDDEN)?,
            row_2_bias: take_floats(bytes, &mut cursor, ROW_OUTPUT)?,
            head_1_weight: take_floats(bytes, &mut cursor, HEAD_HIDDEN * HEAD_INPUT)?,
            head_1_bias: take_floats(bytes, &mut cursor, HEAD_HIDDEN)?,
            head_2_weight: take_floats(bytes, &mut cursor, OUTPUTS * HEAD_HIDDEN)?,
            head_2_bias: take_floats(bytes, &mut cursor, OUTPUTS)?,
        };
        if cursor != bytes.len() {
            return Err("fragment model has trailing parameters".to_string());
        }
        Ok(model)
    }

    fn predict(
        &self,
        board: &Board,
        visible: &[SearchPiece],
        current_lines: u8,
        lexicon: u8,
        input_mode: InputMode,
    ) -> FragmentPrediction {
        let mut first_filled = [BOARD_HEIGHT; BOARD_WIDTH];
        for row in 0..BOARD_HEIGHT {
            for col in 0..BOARD_WIDTH {
                if board[row * BOARD_WIDTH + col] != 0 && first_filled[col] == BOARD_HEIGHT {
                    first_filled[col] = row;
                }
            }
        }
        let mut queue_counts = [0u8; 26];
        if matches!(input_mode, InputMode::Full) {
            for piece in visible.iter().take(5) {
                for &letter in &piece.letters {
                    queue_counts[letter as usize - 1] += 1;
                }
            }
        }
        let mut row_sum = [0f32; ROW_OUTPUT];
        let mut row_max = [f32::NEG_INFINITY; ROW_OUTPUT];
        for row in 0..BOARD_HEIGHT {
            let cells = &board[row * BOARD_WIDTH..(row + 1) * BOARD_WIDTH];
            let mut input = [0f32; ROW_INPUT];
            let mut cursor = 0usize;
            for &cell in cells {
                let letter = if cell == 0 {
                    0
                } else if matches!(input_mode, InputMode::MaskWordInputs) {
                    1
                } else {
                    (cell & 0x1f) as usize
                };
                input[cursor..cursor + LETTER_EMBED].copy_from_slice(
                    &self.letter_embedding[letter * LETTER_EMBED..(letter + 1) * LETTER_EMBED],
                );
                cursor += LETTER_EMBED;
            }
            for pair in cells.windows(2) {
                let boundary = if pair[0] == 0 || pair[1] == 0 {
                    0
                } else if pair[0] >> 5 == pair[1] >> 5 {
                    1
                } else {
                    2
                };
                input[cursor..cursor + BOUNDARY_EMBED].copy_from_slice(
                    &self.boundary_embedding
                        [boundary * BOUNDARY_EMBED..(boundary + 1) * BOUNDARY_EMBED],
                );
                cursor += BOUNDARY_EMBED;
            }
            for col in 0..BOARD_WIDTH {
                input[cursor] = f32::from(cells[col] == 0 && row < first_filled[col]);
                cursor += 1;
            }
            for count in queue_counts {
                input[cursor] = f32::from(count) / 4.0;
                cursor += 1;
            }
            input[cursor] = row as f32 / (BOARD_HEIGHT - 1) as f32;
            cursor += 1;
            input[cursor] = f32::from(lexicon);

            let mut hidden = [0f32; ROW_HIDDEN];
            let mut encoded = [0f32; ROW_OUTPUT];
            dense(
                &input,
                &self.row_1_weight,
                &self.row_1_bias,
                &mut hidden,
                true,
            );
            dense(
                &hidden,
                &self.row_2_weight,
                &self.row_2_bias,
                &mut encoded,
                true,
            );
            for index in 0..ROW_OUTPUT {
                row_sum[index] += encoded[index];
                row_max[index] = row_max[index].max(encoded[index]);
            }
        }

        let mut head_input = [0f32; HEAD_INPUT];
        head_input[..ROW_OUTPUT].copy_from_slice(&row_sum);
        head_input[ROW_OUTPUT..ROW_OUTPUT * 2].copy_from_slice(&row_max);
        let mut cursor = ROW_OUTPUT * 2;
        for position in 0..5 {
            if let Some(piece) = visible.get(position) {
                head_input[cursor + piece.piece as usize] = 1.0;
            }
            cursor += 7;
        }
        head_input[cursor] = (40.0 - f32::from(current_lines)).max(0.0) / 40.0;
        cursor += 1;
        head_input[cursor] = f32::from(lexicon);

        let mut hidden = [0f32; HEAD_HIDDEN];
        let mut output = [0f32; OUTPUTS];
        dense(
            &head_input,
            &self.head_1_weight,
            &self.head_1_bias,
            &mut hidden,
            true,
        );
        dense(
            &hidden,
            &self.head_2_weight,
            &self.head_2_bias,
            &mut output,
            false,
        );
        FragmentPrediction {
            score_4: inverse(output[0], 100.0),
            score_8: inverse(output[1], 100.0),
            score_16: inverse(output[2], 100.0),
            words_8: inverse(output[3], 1.0),
            word_length_8: inverse(output[4], 1.0),
            lines_8: inverse(output[5], 1.0),
            score_per_line_8: inverse(output[6], 100.0),
        }
    }
}

impl FragmentModelPair {
    pub fn from_bytes(full: &[u8], context: &[u8]) -> Result<Self, String> {
        Ok(Self {
            full: FragmentModel::from_bytes(full)?,
            context: FragmentModel::from_bytes(context)?,
        })
    }

    pub(crate) fn residual(
        &self,
        board: &Board,
        visible: &[SearchPiece],
        current_lines: u8,
        lexicon: u8,
    ) -> FragmentResidual {
        let full = self
            .full
            .predict(board, visible, current_lines, lexicon, InputMode::Full);
        let context = self.context.predict(
            board,
            visible,
            current_lines,
            lexicon,
            InputMode::MaskWordInputs,
        );
        FragmentResidual {
            score_8: full.score_8 - context.score_8,
            words_8: full.words_8 - context.words_8,
            word_length_8: full.word_length_8 - context.word_length_8,
            score_per_line_8: full.score_per_line_8 - context.score_per_line_8,
        }
    }
}
