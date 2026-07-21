use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{
    AdamW, Embedding, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding, linear,
};
use memmap2::Mmap;
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const HEADER_BYTES: usize = 64;
const RECORD_BYTES: usize = 288;
const MAGIC: &[u8; 8] = b"KVCF1\0\0\0";
const ROWS: usize = 22;
const COLS: usize = 10;
const LETTER_EMBED: usize = 8;
const BOUNDARY_EMBED: usize = 3;
const ROW_NUMERIC: usize = 38;
const ROW_INPUT: usize = COLS * LETTER_EMBED + (COLS - 1) * BOUNDARY_EMBED + ROW_NUMERIC;
const ROW_HIDDEN: usize = 64;
const ROW_OUTPUT: usize = 16;
const GLOBAL_INPUT: usize = 37;
const HEAD_INPUT: usize = ROW_OUTPUT * 2 + GLOBAL_INPUT;
const HEAD_HIDDEN: usize = 64;
const OUTPUTS: usize = 4;
const TOP_OUT_PENALTY: i32 = 2_500;

type AnyError = Box<dyn Error + Send + Sync>;
type Result<T> = std::result::Result<T, AnyError>;

#[derive(Clone)]
struct Options {
    data: PathBuf,
    output: PathBuf,
    epochs: usize,
    group_batch_size: usize,
    learning_rate: f64,
    device: String,
    seed: u64,
}

#[derive(Clone, Copy)]
enum InputMode {
    Full,
    MaskWordInputs,
}

#[derive(Clone, Copy)]
struct Group {
    start: usize,
    len: usize,
    seed: u32,
    lexicon: u8,
}

struct Dataset {
    mmap: Mmap,
    records: usize,
    groups: Vec<Group>,
}

impl Dataset {
    fn open(path: &Path) -> Result<Self> {
        let mmap = unsafe { Mmap::map(&File::open(path)?)? };
        if mmap.len() < HEADER_BYTES || &mmap[..8] != MAGIC {
            return Err(format!("{} is not a counterfactual dataset", path.display()).into());
        }
        let version = read_u32(&mmap, 8);
        let record_bytes = read_u32(&mmap, 12) as usize;
        let expected_groups = read_u64(&mmap, 16) as usize;
        let records = read_u64(&mmap, 24) as usize;
        if version != 1 || record_bytes != RECORD_BYTES {
            return Err("counterfactual dataset version or record width is unsupported".into());
        }
        if mmap.len() != HEADER_BYTES + records * RECORD_BYTES {
            return Err("counterfactual dataset length does not match its header".into());
        }
        let mut dataset = Self {
            mmap,
            records,
            groups: Vec::with_capacity(expected_groups),
        };
        let mut index = 0usize;
        while index < records {
            let record = dataset.record(index);
            let len = read_u16(record, 254) as usize;
            if len == 0 || index + len > records || record[247] != 0 {
                return Err("counterfactual candidate group is malformed".into());
            }
            let seed = read_u32(record, 248);
            let step = read_u16(record, 252);
            let lexicon = record[246];
            for rank in 0..len {
                let candidate = dataset.record(index + rank);
                if candidate[247] as usize != rank
                    || read_u16(candidate, 254) as usize != len
                    || read_u32(candidate, 248) != seed
                    || read_u16(candidate, 252) != step
                    || candidate[246] != lexicon
                {
                    return Err("counterfactual sibling metadata is inconsistent".into());
                }
            }
            dataset.groups.push(Group {
                start: index,
                len,
                seed,
                lexicon,
            });
            index += len;
        }
        if dataset.groups.len() != expected_groups {
            return Err("counterfactual group count does not match its header".into());
        }
        Ok(dataset)
    }

    fn record(&self, index: usize) -> &[u8] {
        let start = HEADER_BYTES + index * RECORD_BYTES;
        &self.mmap[start..start + RECORD_BYTES]
    }
}

struct RankModel {
    letter_embedding: Embedding,
    boundary_embedding: Embedding,
    row_1: Linear,
    row_2: Linear,
    head_1: Linear,
    head_2: Linear,
}

impl RankModel {
    fn new(vb: VarBuilder<'_>) -> CandleResult<Self> {
        Ok(Self {
            letter_embedding: embedding(27, LETTER_EMBED, vb.pp("letter_embedding"))?,
            boundary_embedding: embedding(3, BOUNDARY_EMBED, vb.pp("boundary_embedding"))?,
            row_1: linear(ROW_INPUT, ROW_HIDDEN, vb.pp("row_1"))?,
            row_2: linear(ROW_HIDDEN, ROW_OUTPUT, vb.pp("row_2"))?,
            head_1: linear(HEAD_INPUT, HEAD_HIDDEN, vb.pp("head_1"))?,
            head_2: linear(HEAD_HIDDEN, OUTPUTS, vb.pp("head_2"))?,
        })
    }

    fn forward(&self, batch: &DeviceBatch) -> CandleResult<Tensor> {
        let row_count = batch.candidates * ROWS;
        let letters = self
            .letter_embedding
            .forward(&batch.letters)?
            .reshape((row_count, COLS * LETTER_EMBED))?;
        let boundaries = self
            .boundary_embedding
            .forward(&batch.boundaries)?
            .reshape((row_count, (COLS - 1) * BOUNDARY_EMBED))?;
        let row_input = Tensor::cat(&[&letters, &boundaries, &batch.row_numeric], 1)?;
        let rows = self
            .row_2
            .forward(&self.row_1.forward(&row_input)?.relu()?)?
            .relu()?
            .reshape((batch.candidates, ROWS, ROW_OUTPUT))?;
        let row_sum = rows.sum(1)?;
        let row_max = rows.max(1)?;
        let head_input = Tensor::cat(&[&row_sum, &row_max, &batch.global], 1)?;
        self.head_2
            .forward(&self.head_1.forward(&head_input)?.relu()?)
    }
}

struct HostBatch {
    letters: Vec<u32>,
    boundaries: Vec<u32>,
    row_numeric: Vec<f32>,
    global: Vec<f32>,
    heuristics: Vec<f32>,
    auxiliary_targets: Vec<f32>,
    pair_left: Vec<u32>,
    pair_right: Vec<u32>,
    pair_sign: Vec<f32>,
    pair_margin: Vec<f32>,
    objectives: Vec<i32>,
    topped_out: Vec<bool>,
    group_sizes: Vec<usize>,
    group_lexica: Vec<u8>,
    candidates: usize,
}

impl HostBatch {
    fn from_groups(
        dataset: &Dataset,
        group_indices: &[u32],
        input_mode: InputMode,
    ) -> Result<Self> {
        let candidate_capacity: usize = group_indices
            .iter()
            .map(|&index| dataset.groups[index as usize].len)
            .sum();
        let mut batch = Self {
            letters: Vec::with_capacity(candidate_capacity * ROWS * COLS),
            boundaries: Vec::with_capacity(candidate_capacity * ROWS * (COLS - 1)),
            row_numeric: Vec::with_capacity(candidate_capacity * ROWS * ROW_NUMERIC),
            global: Vec::with_capacity(candidate_capacity * GLOBAL_INPUT),
            heuristics: Vec::with_capacity(candidate_capacity),
            auxiliary_targets: Vec::with_capacity(candidate_capacity * 3),
            pair_left: Vec::new(),
            pair_right: Vec::new(),
            pair_sign: Vec::new(),
            pair_margin: Vec::new(),
            objectives: Vec::with_capacity(candidate_capacity),
            topped_out: Vec::with_capacity(candidate_capacity),
            group_sizes: Vec::with_capacity(group_indices.len()),
            group_lexica: Vec::with_capacity(group_indices.len()),
            candidates: 0,
        };
        for &group_index in group_indices {
            let group = dataset.groups[group_index as usize];
            let group_start = batch.candidates;
            let mut group_objectives = Vec::with_capacity(group.len);
            for candidate in 0..group.len {
                let record = dataset.record(group.start + candidate);
                let objective = objective(record);
                group_objectives.push(objective);
                batch.push_record(record, input_mode, objective)?;
            }
            for left in 0..group.len {
                for right in left + 1..group.len {
                    let difference = group_objectives[left] - group_objectives[right];
                    if difference == 0 {
                        continue;
                    }
                    batch.pair_left.push((group_start + left) as u32);
                    batch.pair_right.push((group_start + right) as u32);
                    batch
                        .pair_sign
                        .push(if difference > 0 { 1.0 } else { -1.0 });
                    batch
                        .pair_margin
                        .push((difference.unsigned_abs() as f32 / 100.0).clamp(0.1, 3.0));
                }
            }
            batch.group_sizes.push(group.len);
            batch.group_lexica.push(group.lexicon);
        }
        if batch.pair_left.is_empty() {
            return Err("training batch contains no unequal candidate pairs".into());
        }
        Ok(batch)
    }

    fn push_record(&mut self, record: &[u8], input_mode: InputMode, objective: i32) -> Result<()> {
        let board = &record[..220];
        let mut queue_counts = [0u8; 26];
        for piece in 0..5 {
            let start = 220 + piece * 5;
            for &letter in &record[start + 1..start + 5] {
                if !(1..=26).contains(&letter) {
                    return Err("visible piece contains an invalid letter".into());
                }
                if matches!(input_mode, InputMode::Full) {
                    queue_counts[letter as usize - 1] += 1;
                }
            }
        }
        let mut first_filled = [ROWS; COLS];
        for row in 0..ROWS {
            for col in 0..COLS {
                if board[row * COLS + col] != 0 && first_filled[col] == ROWS {
                    first_filled[col] = row;
                }
            }
        }
        for row in 0..ROWS {
            let cells = &board[row * COLS..(row + 1) * COLS];
            self.letters.extend(cells.iter().map(|cell| {
                if matches!(input_mode, InputMode::MaskWordInputs) && *cell != 0 {
                    1
                } else {
                    u32::from(cell & 0x1f)
                }
            }));
            self.boundaries.extend(cells.windows(2).map(|pair| {
                if pair[0] == 0 || pair[1] == 0 {
                    0
                } else if pair[0] >> 5 == pair[1] >> 5 {
                    1
                } else {
                    2
                }
            }));
            for col in 0..COLS {
                self.row_numeric
                    .push(f32::from(cells[col] == 0 && row < first_filled[col]));
            }
            self.row_numeric
                .extend(queue_counts.iter().map(|&count| f32::from(count) / 4.0));
            self.row_numeric.push(row as f32 / (ROWS - 1) as f32);
            self.row_numeric.push(f32::from(record[246]));
        }
        for piece in 0..5 {
            let shape = record[220 + piece * 5] as usize;
            if shape >= 7 {
                return Err("visible piece contains an invalid shape".into());
            }
            for candidate in 0..7 {
                self.global.push(f32::from(shape == candidate));
            }
        }
        self.global
            .push((40.0 - f32::from(record[245])).max(0.0) / 40.0);
        self.global.push(f32::from(record[246]));
        self.heuristics.push(read_f32(record, 270) / 100.0);
        let rollout_score = read_i32(record, 274).max(0) as f32;
        let rollout_lines = f32::from(read_u16(record, 278));
        let rollout_word_length = f32::from(read_u16(record, 282));
        let score_per_line = if rollout_lines > 0.0 {
            rollout_score / rollout_lines
        } else {
            0.0
        };
        self.auxiliary_targets.extend([
            (rollout_score / 100.0).ln_1p(),
            rollout_word_length.ln_1p(),
            (score_per_line / 100.0).ln_1p(),
        ]);
        self.objectives.push(objective);
        self.topped_out.push(record[286] != 0);
        self.candidates += 1;
        Ok(())
    }

    fn to_device(&self, device: &Device) -> CandleResult<DeviceBatch> {
        let rows = self.candidates * ROWS;
        Ok(DeviceBatch {
            letters: Tensor::from_vec(self.letters.clone(), (rows, COLS), device)?,
            boundaries: Tensor::from_vec(self.boundaries.clone(), (rows, COLS - 1), device)?,
            row_numeric: Tensor::from_vec(self.row_numeric.clone(), (rows, ROW_NUMERIC), device)?,
            global: Tensor::from_vec(self.global.clone(), (self.candidates, GLOBAL_INPUT), device)?,
            heuristics: Tensor::from_vec(self.heuristics.clone(), self.candidates, device)?,
            auxiliary_targets: Tensor::from_vec(
                self.auxiliary_targets.clone(),
                (self.candidates, 3),
                device,
            )?,
            pair_left: Tensor::from_vec(self.pair_left.clone(), self.pair_left.len(), device)?,
            pair_right: Tensor::from_vec(self.pair_right.clone(), self.pair_right.len(), device)?,
            pair_sign: Tensor::from_vec(self.pair_sign.clone(), self.pair_sign.len(), device)?,
            pair_margin: Tensor::from_vec(
                self.pair_margin.clone(),
                self.pair_margin.len(),
                device,
            )?,
            candidates: self.candidates,
        })
    }
}

struct DeviceBatch {
    letters: Tensor,
    boundaries: Tensor,
    row_numeric: Tensor,
    global: Tensor,
    heuristics: Tensor,
    auxiliary_targets: Tensor,
    pair_left: Tensor,
    pair_right: Tensor,
    pair_sign: Tensor,
    pair_margin: Tensor,
    candidates: usize,
}

fn batch_loss(model: &RankModel, batch: &DeviceBatch) -> CandleResult<Tensor> {
    let predictions = model.forward(batch)?;
    let correction = predictions.narrow(1, 0, 1)?.flatten_all()?;
    let combined = correction.add(&batch.heuristics)?;
    let left = combined.index_select(&batch.pair_left, 0)?;
    let right = combined.index_select(&batch.pair_right, 0)?;
    let signed_difference = left.sub(&right)?.mul(&batch.pair_sign)?;
    let ranking = batch
        .pair_margin
        .sub(&signed_difference)?
        .relu()?
        .sqr()?
        .mean_all()?;
    let auxiliary = predictions
        .narrow(1, 1, 3)?
        .sub(&batch.auxiliary_targets)?
        .sqr()?
        .mean_all()?;
    ranking.affine(0.9, 0.0)?.add(&auxiliary.affine(0.1, 0.0)?)
}

#[derive(Clone)]
struct Selection {
    lexicon: u8,
    baseline: i32,
    heuristic: i32,
    model: i32,
    oracle: i32,
    selected_rank: usize,
    selected_topped_out: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SelectionReport {
    groups: usize,
    baseline_mean_objective: f64,
    heuristic_mean_objective: f64,
    model_mean_objective: f64,
    oracle_mean_objective: f64,
    model_mean_uplift: f64,
    heuristic_mean_uplift: f64,
    oracle_mean_uplift: f64,
    oracle_headroom_recovered: f64,
    wins: usize,
    ties: usize,
    losses: usize,
    changed_actions: usize,
    selected_top_outs: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EvaluationReport {
    all: SelectionReport,
    csw24: SelectionReport,
    nwl23: SelectionReport,
    pair_accuracy: f64,
    elapsed_seconds: f64,
}

fn selection_report(selections: &[&Selection]) -> SelectionReport {
    let count = selections.len().max(1) as f64;
    let baseline = selections
        .iter()
        .map(|selection| f64::from(selection.baseline))
        .sum::<f64>()
        / count;
    let model = selections
        .iter()
        .map(|selection| f64::from(selection.model))
        .sum::<f64>()
        / count;
    let heuristic = selections
        .iter()
        .map(|selection| f64::from(selection.heuristic))
        .sum::<f64>()
        / count;
    let oracle = selections
        .iter()
        .map(|selection| f64::from(selection.oracle))
        .sum::<f64>()
        / count;
    SelectionReport {
        groups: selections.len(),
        baseline_mean_objective: baseline,
        heuristic_mean_objective: heuristic,
        model_mean_objective: model,
        oracle_mean_objective: oracle,
        model_mean_uplift: model - baseline,
        heuristic_mean_uplift: heuristic - baseline,
        oracle_mean_uplift: oracle - baseline,
        oracle_headroom_recovered: if oracle > baseline {
            (model - baseline) / (oracle - baseline)
        } else {
            0.0
        },
        wins: selections
            .iter()
            .filter(|selection| selection.model > selection.baseline)
            .count(),
        ties: selections
            .iter()
            .filter(|selection| selection.model == selection.baseline)
            .count(),
        losses: selections
            .iter()
            .filter(|selection| selection.model < selection.baseline)
            .count(),
        changed_actions: selections
            .iter()
            .filter(|selection| selection.selected_rank > 0)
            .count(),
        selected_top_outs: selections
            .iter()
            .filter(|selection| selection.selected_topped_out)
            .count(),
    }
}

fn evaluate(
    model: &RankModel,
    dataset: &Dataset,
    groups: &[u32],
    group_batch_size: usize,
    device: &Device,
    input_mode: InputMode,
) -> Result<EvaluationReport> {
    let began = Instant::now();
    let mut selections = Vec::with_capacity(groups.len());
    let mut correct_pairs = 0u64;
    let mut total_pairs = 0u64;
    for group_batch in groups.chunks(group_batch_size) {
        let host = HostBatch::from_groups(dataset, group_batch, input_mode)?;
        let device_batch = host.to_device(device)?;
        let predictions = model.forward(&device_batch)?.to_vec2::<f32>()?;
        let combined: Vec<f32> = predictions
            .iter()
            .zip(&host.heuristics)
            .map(|(prediction, heuristic)| prediction[0] + heuristic)
            .collect();
        for pair in 0..host.pair_left.len() {
            let left = host.pair_left[pair] as usize;
            let right = host.pair_right[pair] as usize;
            correct_pairs +=
                u64::from((combined[left] - combined[right]) * host.pair_sign[pair] > 0.0);
            total_pairs += 1;
        }
        let mut offset = 0usize;
        for (&size, &lexicon) in host.group_sizes.iter().zip(&host.group_lexica) {
            let end = offset + size;
            let selected = (offset..end)
                .max_by(|&left, &right| combined[left].total_cmp(&combined[right]))
                .expect("nonempty group");
            let heuristic_selected = (offset..end)
                .max_by(|&left, &right| host.heuristics[left].total_cmp(&host.heuristics[right]))
                .expect("nonempty group");
            let oracle = *host.objectives[offset..end]
                .iter()
                .max()
                .expect("nonempty group");
            selections.push(Selection {
                lexicon,
                baseline: host.objectives[offset],
                heuristic: host.objectives[heuristic_selected],
                model: host.objectives[selected],
                oracle,
                selected_rank: selected - offset,
                selected_topped_out: host.topped_out[selected],
            });
            offset = end;
        }
    }
    let all: Vec<&Selection> = selections.iter().collect();
    let csw24: Vec<&Selection> = selections
        .iter()
        .filter(|selection| selection.lexicon == 0)
        .collect();
    let nwl23: Vec<&Selection> = selections
        .iter()
        .filter(|selection| selection.lexicon == 1)
        .collect();
    Ok(EvaluationReport {
        all: selection_report(&all),
        csw24: selection_report(&csw24),
        nwl23: selection_report(&nwl23),
        pair_accuracy: correct_pairs as f64 / total_pairs.max(1) as f64,
        elapsed_seconds: began.elapsed().as_secs_f64(),
    })
}

#[derive(Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    fn shuffle(&mut self, values: &mut [u32]) {
        for index in (1..values.len()).rev() {
            values.swap(index, self.next() as usize % (index + 1));
        }
    }
}

fn initialize_model_parameters(
    varmap: &mut VarMap,
    device: &Device,
    seed: u64,
) -> CandleResult<()> {
    let mut parameters: Vec<(String, Vec<usize>)> = {
        let data = varmap.data().lock().expect("model parameter lock");
        data.iter()
            .map(|(name, value)| (name.clone(), value.as_tensor().dims().to_vec()))
            .collect()
    };
    parameters.sort_by(|left, right| left.0.cmp(&right.0));
    let mut random = XorShift64::new(seed ^ 0x5241_4e4b_494e_4701);
    for (name, dimensions) in parameters {
        let count: usize = dimensions.iter().product();
        let values = if name.ends_with("bias") {
            vec![0.0; count]
        } else {
            let bound = if name.contains("embedding") {
                0.25
            } else {
                (6.0 / dimensions.get(1).copied().unwrap_or(1) as f32).sqrt()
            };
            (0..count)
                .map(|_| (random.unit_f32() * 2.0 - 1.0) * bound)
                .collect()
        };
        varmap.set_one(name, Tensor::from_vec(values, dimensions, device)?)?;
    }
    Ok(())
}

fn split_index(seed: u32) -> usize {
    let mut mixed = seed;
    mixed ^= mixed >> 16;
    mixed = mixed.wrapping_mul(0x7feb_352d);
    mixed ^= mixed >> 15;
    mixed = mixed.wrapping_mul(0x846c_a68b);
    mixed ^= mixed >> 16;
    match mixed % 10 {
        9 => 2,
        8 => 1,
        _ => 0,
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("u16 field"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

fn read_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("i32 field"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("u64 field"))
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("f32 field"))
}

fn objective(record: &[u8]) -> i32 {
    read_i32(record, 256) + read_i32(record, 274) - i32::from(record[286] != 0) * TOP_OUT_PENALTY
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
    Ok(Options {
        data: absolute(values.get("data").ok_or("--data is required")?)?,
        output: absolute(values.get("output").ok_or("--output is required")?)?,
        epochs: option(&values, "epochs", 10)?,
        group_batch_size: option(&values, "group-batch-size", 64)?,
        learning_rate: option(&values, "learning-rate", 0.001)?,
        device: option(&values, "device", "cpu".to_string())?,
        seed: option(&values, "seed", 0x4b56_5241_4e4b_0001)?,
    })
}

fn run() -> Result<()> {
    let options = parse_options()?;
    if options.epochs == 0 || options.group_batch_size == 0 {
        return Err("epochs and group batch size must be positive".into());
    }
    fs::create_dir_all(options.output.parent().ok_or("output has no parent")?)?;
    fs::create_dir(&options.output)?;
    let dataset = Dataset::open(&options.data)?;
    let mut splits = [Vec::new(), Vec::new(), Vec::new()];
    for (index, group) in dataset.groups.iter().enumerate() {
        splits[split_index(group.seed)].push(index as u32);
    }
    let [mut train, validation, test] = splits;
    let device = match options.device.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        other => return Err(format!("unsupported device {other:?}").into()),
    };
    let mut varmap = VarMap::new();
    let model = RankModel::new(VarBuilder::from_varmap(&varmap, DType::F32, &device))?;
    initialize_model_parameters(&mut varmap, &device, options.seed)?;
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: options.learning_rate,
            weight_decay: 0.0001,
            ..Default::default()
        },
    )?;
    let began = Instant::now();
    let mut random = XorShift64::new(options.seed);
    let mut best_validation_uplift = f64::NEG_INFINITY;
    let best_path = options.output.join("model.safetensors");
    let mut epoch_reports = Vec::new();
    for epoch in 1..=options.epochs {
        random.shuffle(&mut train);
        let epoch_began = Instant::now();
        let mut loss_sum = 0f64;
        let mut batches = 0u64;
        for group_batch in train.chunks(options.group_batch_size) {
            let host = HostBatch::from_groups(&dataset, group_batch, InputMode::Full)?;
            let device_batch = host.to_device(&device)?;
            let loss = batch_loss(&model, &device_batch)?;
            optimizer.backward_step(&loss)?;
            loss_sum += f64::from(loss.to_vec0::<f32>()?);
            batches += 1;
        }
        let validation_report = evaluate(
            &model,
            &dataset,
            &validation,
            options.group_batch_size,
            &device,
            InputMode::Full,
        )?;
        if validation_report.all.model_mean_uplift > best_validation_uplift {
            best_validation_uplift = validation_report.all.model_mean_uplift;
            varmap.save(&best_path)?;
        }
        let epoch_report = json!({
            "epoch": epoch,
            "trainingLoss": loss_sum / batches.max(1) as f64,
            "validation": validation_report,
            "elapsedSeconds": epoch_began.elapsed().as_secs_f64(),
        });
        println!("{epoch_report}");
        epoch_reports.push(epoch_report);
    }
    varmap.load(&best_path)?;
    let test_report = evaluate(
        &model,
        &dataset,
        &test,
        options.group_batch_size,
        &device,
        InputMode::Full,
    )?;
    let lexical_ablation = evaluate(
        &model,
        &dataset,
        &test,
        options.group_batch_size,
        &device,
        InputMode::MaskWordInputs,
    )?;
    let report = json!({
        "schemaVersion": 1,
        "architecture": {
            "kind": "candidate-relative row encoder",
            "letterEmbedding": LETTER_EMBED,
            "boundaryEmbedding": BOUNDARY_EMBED,
            "rowHidden": ROW_HIDDEN,
            "rowOutput": ROW_OUTPUT,
            "headHidden": HEAD_HIDDEN,
            "outputs": [
                "heuristic correction",
                "log1p(rolloutScore/100)",
                "log1p(rolloutWordLength)",
                "log1p(rolloutScorePerLine/100)"
            ],
            "loss": "90% all-pairs margin ranking over heuristic + learned correction; 10% lexical rollout auxiliaries",
        },
        "options": {
            "data": options.data,
            "epochs": options.epochs,
            "groupBatchSize": options.group_batch_size,
            "learningRate": options.learning_rate,
            "device": options.device,
            "seed": options.seed,
        },
        "splits": {
            "kind": "hashed episode seed 80/10/10",
            "trainGroups": train.len(),
            "validationGroups": validation.len(),
            "testGroups": test.len(),
            "sourceRecords": dataset.records,
        },
        "epochs": epoch_reports,
        "bestValidationUplift": best_validation_uplift,
        "test": test_report,
        "lexicalAblation": lexical_ablation,
        "elapsedSeconds": began.elapsed().as_secs_f64(),
    });
    let mut file = File::create(options.output.join("REPORT.json"))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    println!(
        "{}",
        json!({
            "status": "complete",
            "bestValidationUplift": best_validation_uplift,
            "testModelUplift": test_report.all.model_mean_uplift,
            "testOracleUplift": test_report.all.oracle_mean_uplift,
            "testHeadroomRecovered": test_report.all.oracle_headroom_recovered,
            "lexicalAblationUplift": lexical_ablation.all.model_mean_uplift,
            "elapsedSeconds": began.elapsed().as_secs_f64(),
            "output": options.output,
        })
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("kvadrat-counterfactual-train: {error}");
        std::process::exit(1);
    }
}
