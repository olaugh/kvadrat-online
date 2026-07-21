use crate::{BOARD_HEIGHT, BOARD_WIDTH, Board, SearchPiece};

const MAGIC: &[u8; 8] = b"KVRK1\0\0\0";
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
const OUTPUTS: usize = 4;
const FLOAT_COUNT: usize = 15_349;

#[derive(Clone)]
pub struct RootRankModel {
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

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let field = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "root ranker header is truncated".to_string())?;
    Ok(u32::from_le_bytes(
        field.try_into().expect("four-byte model field"),
    ))
}

fn take_floats(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    let end = cursor
        .checked_add(count * 4)
        .ok_or_else(|| "root ranker size overflow".to_string())?;
    let source = bytes
        .get(*cursor..end)
        .ok_or_else(|| "root ranker parameters are truncated".to_string())?;
    let values: Vec<f32> = source
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte model weight")))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("root ranker contains a non-finite weight".to_string());
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

impl RootRankModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != HEADER_BYTES + FLOAT_COUNT * 4 || bytes.get(..8) != Some(MAGIC) {
            return Err("root ranker has an invalid magic or length".to_string());
        }
        if read_u32(bytes, 8)? != 1
            || read_u32(bytes, 12)? as usize != FLOAT_COUNT
            || read_u32(bytes, 16)? as usize != OUTPUTS
            || read_u32(bytes, 20)? != 10
        {
            return Err("root ranker architecture is unsupported".to_string());
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
            return Err("root ranker has trailing parameters".to_string());
        }
        Ok(model)
    }

    pub(crate) fn correction(
        &self,
        board: &Board,
        visible: &[SearchPiece],
        current_lines: u8,
        lexicon: u8,
    ) -> f32 {
        let mut first_filled = [BOARD_HEIGHT; BOARD_WIDTH];
        for row in 0..BOARD_HEIGHT {
            for col in 0..BOARD_WIDTH {
                if board[row * BOARD_WIDTH + col] != 0 && first_filled[col] == BOARD_HEIGHT {
                    first_filled[col] = row;
                }
            }
        }
        let mut queue_counts = [0u8; 26];
        for piece in visible.iter().take(5) {
            for &letter in &piece.letters {
                queue_counts[letter as usize - 1] += 1;
            }
        }
        let mut row_sum = [0f32; ROW_OUTPUT];
        let mut row_max = [f32::NEG_INFINITY; ROW_OUTPUT];
        for row in 0..BOARD_HEIGHT {
            let cells = &board[row * BOARD_WIDTH..(row + 1) * BOARD_WIDTH];
            let mut input = [0f32; ROW_INPUT];
            let mut cursor = 0usize;
            for &cell in cells {
                let letter = (cell & 0x1f) as usize;
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
        output[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_model() -> Vec<u8> {
        let mut bytes = vec![0; HEADER_BYTES + FLOAT_COUNT * 4];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&(FLOAT_COUNT as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&(OUTPUTS as u32).to_le_bytes());
        bytes[20..24].copy_from_slice(&10u32.to_le_bytes());
        bytes
    }

    #[test]
    fn rejects_truncated_models() {
        assert!(RootRankModel::from_bytes(&zero_model()[..HEADER_BYTES]).is_err());
    }

    #[test]
    fn zero_model_has_zero_correction() {
        let model = RootRankModel::from_bytes(&zero_model()).expect("valid zero model");
        let visible = [SearchPiece {
            piece: 0,
            letters: [1, 2, 3, 4],
        }; 5];
        assert_eq!(model.correction(&[0; 220], &visible, 0, 0), 0.0);
    }
}
