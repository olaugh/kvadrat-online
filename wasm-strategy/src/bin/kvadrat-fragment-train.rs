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
const RECORD_BYTES: usize = 272;
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
const OUTPUTS: usize = 7;
const SCORE_4: usize = 0;
const SCORE_8: usize = 1;
const SCORE_16: usize = 2;
const WORDS_8: usize = 3;
const WORD_LENGTH_8: usize = 4;
const LINES_8: usize = 5;
const SCORE_PER_LINE_8: usize = 6;
const MAGIC: &[u8; 8] = b"KVFRAG1\0";

type AnyError = Box<dyn Error + Send + Sync>;
type Result<T> = std::result::Result<T, AnyError>;

#[derive(Clone)]
struct Options {
    data: PathBuf,
    output: PathBuf,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    max_train: usize,
    max_validation: usize,
    max_test: usize,
    device: String,
    input_mode: InputMode,
    lexicon: LexiconFilter,
    initial_model: Option<PathBuf>,
    seed: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
enum LexiconFilter {
    All,
    Csw24,
    Nwl23,
}

impl LexiconFilter {
    fn matches(self, record: &[u8]) -> bool {
        match self {
            Self::All => true,
            Self::Csw24 => record[246] == 0,
            Self::Nwl23 => record[246] == 1,
        }
    }
}

impl std::str::FromStr for LexiconFilter {
    type Err = LexiconFilterError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "all" => Ok(Self::All),
            "csw24" => Ok(Self::Csw24),
            "nwl23" => Ok(Self::Nwl23),
            _ => Err(LexiconFilterError(value.to_string())),
        }
    }
}

#[derive(Debug)]
struct LexiconFilterError(String);

impl std::fmt::Display for LexiconFilterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "unsupported lexicon filter {:?}", self.0)
    }
}

impl Error for LexiconFilterError {}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
enum InputMode {
    Full,
    MaskBoardLetters,
    MaskFutureLetters,
    MaskWordInputs,
}

impl InputMode {
    fn masks_board_letters(self) -> bool {
        matches!(self, Self::MaskBoardLetters | Self::MaskWordInputs)
    }

    fn masks_future_letters(self) -> bool {
        matches!(self, Self::MaskFutureLetters | Self::MaskWordInputs)
    }
}

impl std::str::FromStr for InputMode {
    type Err = InputModeError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "full" => Ok(Self::Full),
            "mask-board-letters" => Ok(Self::MaskBoardLetters),
            "mask-future-letters" => Ok(Self::MaskFutureLetters),
            "mask-word-inputs" => Ok(Self::MaskWordInputs),
            _ => Err(InputModeError(value.to_string())),
        }
    }
}

#[derive(Debug)]
struct InputModeError(String);

impl std::fmt::Display for InputModeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "unsupported input mode {:?}", self.0)
    }
}

impl Error for InputModeError {}

struct Dataset {
    mmap: Mmap,
    records: usize,
}

impl Dataset {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < HEADER_BYTES || &mmap[..8] != MAGIC {
            return Err(format!("{} is not a Kvadrat fragment dataset", path.display()).into());
        }
        let version = u32::from_le_bytes(mmap[8..12].try_into()?);
        let record_bytes = u32::from_le_bytes(mmap[12..16].try_into()?) as usize;
        let records = u64::from_le_bytes(mmap[16..24].try_into()?) as usize;
        if version != 1 || record_bytes != RECORD_BYTES {
            return Err(format!(
                "unsupported dataset version {version} or record size {record_bytes}"
            )
            .into());
        }
        if mmap.len() != HEADER_BYTES + records * RECORD_BYTES {
            return Err(format!("{} length does not match its header", path.display()).into());
        }
        Ok(Self { mmap, records })
    }

    fn record(&self, index: usize) -> &[u8] {
        let start = HEADER_BYTES + index * RECORD_BYTES;
        &self.mmap[start..start + RECORD_BYTES]
    }

    fn indices(&self, filter: LexiconFilter, limit: usize) -> Vec<u32> {
        (0..self.records)
            .filter(|&index| filter.matches(self.record(index)))
            .take(limit)
            .map(|index| index as u32)
            .collect()
    }
}

struct FragmentModel {
    letter_embedding: Embedding,
    boundary_embedding: Embedding,
    row_1: Linear,
    row_2: Linear,
    head_1: Linear,
    head_2: Linear,
}

impl FragmentModel {
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
        let row_count = batch.batch_size * ROWS;
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
            .relu()?;
        let rows = rows.reshape((batch.batch_size, ROWS, ROW_OUTPUT))?;
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
    targets: Vec<f32>,
    raw_targets: Vec<[f32; OUTPUTS]>,
    batch_size: usize,
}

struct DeviceBatch {
    letters: Tensor,
    boundaries: Tensor,
    row_numeric: Tensor,
    global: Tensor,
    targets: Tensor,
    batch_size: usize,
}

impl HostBatch {
    fn from_indices(dataset: &Dataset, indices: &[u32], input_mode: InputMode) -> Result<Self> {
        let batch_size = indices.len();
        let mut batch = Self {
            letters: Vec::with_capacity(batch_size * ROWS * COLS),
            boundaries: Vec::with_capacity(batch_size * ROWS * (COLS - 1)),
            row_numeric: Vec::with_capacity(batch_size * ROWS * ROW_NUMERIC),
            global: Vec::with_capacity(batch_size * GLOBAL_INPUT),
            targets: Vec::with_capacity(batch_size * OUTPUTS),
            raw_targets: Vec::with_capacity(batch_size),
            batch_size,
        };
        for &index in indices {
            batch.push_record(dataset.record(index as usize), input_mode)?;
        }
        Ok(batch)
    }

    fn push_record(&mut self, record: &[u8], input_mode: InputMode) -> Result<()> {
        let board = &record[..220];
        let mut queue_counts = [0u8; 26];
        for piece in 0..5 {
            let start = 220 + piece * 5;
            for &letter in &record[start + 1..start + 5] {
                if !(1..=26).contains(&letter) {
                    return Err("visible piece contains an invalid letter".into());
                }
                if !input_mode.masks_future_letters() {
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
            let row_cells = &board[row * COLS..(row + 1) * COLS];
            self.letters.extend(row_cells.iter().map(|cell| {
                if input_mode.masks_board_letters() && *cell != 0 {
                    1
                } else {
                    u32::from(cell & 0x1f)
                }
            }));
            self.boundaries.extend(row_cells.windows(2).map(|pair| {
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
                    .push(f32::from(row_cells[col] == 0 && row < first_filled[col]));
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
        let score_8 = read_u32(record, 256) as f32;
        let lines_8 = f32::from(read_u16(record, 266));
        let raw = [
            read_u32(record, 252) as f32,
            score_8,
            read_u32(record, 260) as f32,
            f32::from(read_u16(record, 264)),
            f32::from(read_u16(record, 268)),
            lines_8,
            if lines_8 > 0.0 {
                score_8 / lines_8
            } else {
                0.0
            },
        ];
        self.raw_targets.push(raw);
        self.targets.extend([
            (raw[SCORE_4] / 100.0).ln_1p(),
            (raw[SCORE_8] / 100.0).ln_1p(),
            (raw[SCORE_16] / 100.0).ln_1p(),
            raw[WORDS_8].ln_1p(),
            raw[WORD_LENGTH_8].ln_1p(),
            raw[LINES_8].ln_1p(),
            (raw[SCORE_PER_LINE_8] / 100.0).ln_1p(),
        ]);
        Ok(())
    }

    fn to_device(&self, device: &Device) -> CandleResult<DeviceBatch> {
        let rows = self.batch_size * ROWS;
        Ok(DeviceBatch {
            letters: Tensor::from_vec(self.letters.clone(), (rows, COLS), device)?,
            boundaries: Tensor::from_vec(self.boundaries.clone(), (rows, COLS - 1), device)?,
            row_numeric: Tensor::from_vec(self.row_numeric.clone(), (rows, ROW_NUMERIC), device)?,
            global: Tensor::from_vec(self.global.clone(), (self.batch_size, GLOBAL_INPUT), device)?,
            targets: Tensor::from_vec(self.targets.clone(), (self.batch_size, OUTPUTS), device)?,
            batch_size: self.batch_size,
        })
    }
}

fn read_u16(record: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(record[offset..offset + 2].try_into().expect("u16 field"))
}

fn read_u32(record: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(record[offset..offset + 4].try_into().expect("u32 field"))
}

#[derive(Default, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RegressionMetrics {
    count: u64,
    mean_target: f64,
    mean_prediction: f64,
    mean_absolute_error: f64,
    root_mean_square_error: f64,
    correlation: f64,
}

#[derive(Default)]
struct MetricAccumulator {
    count: u64,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
    absolute_error: f64,
    square_error: f64,
}

impl MetricAccumulator {
    fn add(&mut self, target: f32, prediction: f32) {
        let x = f64::from(target);
        let y = f64::from(prediction);
        self.count += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
        self.absolute_error += (x - y).abs();
        self.square_error += (x - y).powi(2);
    }

    fn finish(self) -> RegressionMetrics {
        let count = self.count.max(1) as f64;
        let covariance = self.count as f64 * self.sum_xy - self.sum_x * self.sum_y;
        let variance = ((self.count as f64 * self.sum_xx - self.sum_x.powi(2))
            * (self.count as f64 * self.sum_yy - self.sum_y.powi(2)))
        .max(0.0)
        .sqrt();
        RegressionMetrics {
            count: self.count,
            mean_target: self.sum_x / count,
            mean_prediction: self.sum_y / count,
            mean_absolute_error: self.absolute_error / count,
            root_mean_square_error: (self.square_error / count).sqrt(),
            correlation: if variance > 0.0 {
                covariance / variance
            } else {
                0.0
            },
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EvaluationMetrics {
    transformed_loss: f64,
    score_4: RegressionMetrics,
    score_8: RegressionMetrics,
    score_16: RegressionMetrics,
    words_8: RegressionMetrics,
    word_length_8: RegressionMetrics,
    lines_8: RegressionMetrics,
    score_per_line_8: RegressionMetrics,
    elapsed_seconds: f64,
}

fn inverse_target(value: f32, output: usize) -> f32 {
    if matches!(output, SCORE_4 | SCORE_8 | SCORE_16 | SCORE_PER_LINE_8) {
        value.clamp(0.0, 8.0).exp_m1() * 100.0
    } else {
        value.clamp(0.0, 8.0).exp_m1()
    }
}

fn evaluate(
    model: &FragmentModel,
    dataset: &Dataset,
    indices: &[u32],
    batch_size: usize,
    device: &Device,
    input_mode: InputMode,
) -> Result<EvaluationMetrics> {
    let began = Instant::now();
    let mut metrics =
        std::array::from_fn::<MetricAccumulator, OUTPUTS, _>(|_| MetricAccumulator::default());
    let mut transformed_square_error = 0f64;
    let mut transformed_values = 0u64;
    for batch_indices in indices.chunks(batch_size) {
        let host = HostBatch::from_indices(dataset, batch_indices, input_mode)?;
        let device_batch = host.to_device(device)?;
        let predictions = model.forward(&device_batch)?.to_vec2::<f32>()?;
        for (row, raw) in predictions.iter().zip(host.raw_targets) {
            for output in 0..OUTPUTS {
                let prediction = inverse_target(row[output], output);
                metrics[output].add(raw[output], prediction);
                let transformed_target =
                    if matches!(output, SCORE_4 | SCORE_8 | SCORE_16 | SCORE_PER_LINE_8) {
                        (raw[output] / 100.0).ln_1p()
                    } else {
                        raw[output].ln_1p()
                    };
                transformed_square_error += f64::from((row[output] - transformed_target).powi(2));
                transformed_values += 1;
            }
        }
    }
    let [
        score_4,
        score_8,
        score_16,
        words_8,
        word_length_8,
        lines_8,
        score_per_line_8,
    ] = metrics.map(MetricAccumulator::finish);
    Ok(EvaluationMetrics {
        transformed_loss: transformed_square_error / transformed_values.max(1) as f64,
        score_4,
        score_8,
        score_16,
        words_8,
        word_length_8,
        lines_8,
        score_per_line_8,
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
    let mut random = XorShift64::new(seed ^ 0x494e_4954_4941_4c01);
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
        let value = Tensor::from_vec(values, dimensions, device)?;
        varmap.set_one(name, value)?;
    }
    Ok(())
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
    let path = |name: &str| -> Result<PathBuf> {
        let value = values
            .get(name)
            .ok_or_else(|| format!("--{name} is required"))?;
        let path = PathBuf::from(value);
        Ok(if path.is_absolute() {
            path
        } else {
            env::current_dir()?.join(path)
        })
    };
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
    Ok(Options {
        data: path("data")?,
        output: path("output")?,
        epochs: option(&values, "epochs", 3)?,
        batch_size: option(&values, "batch-size", 1024)?,
        learning_rate: option(&values, "learning-rate", 0.001)?,
        max_train: option(&values, "max-train", usize::MAX)?,
        max_validation: option(&values, "max-validation", 200_000)?,
        max_test: option(&values, "max-test", usize::MAX)?,
        device: option(&values, "device", "cpu".to_string())?,
        input_mode: option(&values, "input-mode", InputMode::Full)?,
        lexicon: option(&values, "lexicon", LexiconFilter::All)?,
        initial_model: optional_path("initial-model")?,
        seed: option(&values, "seed", 0x4b56_4144_5241_5401)?,
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

fn run() -> Result<()> {
    let options = parse_options()?;
    if options.batch_size == 0 || options.epochs == 0 {
        return Err("epochs and batch size must be positive".into());
    }
    fs::create_dir_all(options.output.parent().ok_or("output has no parent")?)?;
    fs::create_dir(&options.output)?;
    let train = Dataset::open(&options.data.join("train.kvf"))?;
    let validation = Dataset::open(&options.data.join("validation.kvf"))?;
    let test = Dataset::open(&options.data.join("test.kvf"))?;
    let device = match options.device.as_str() {
        "metal" => Device::new_metal(0)?,
        "cpu" => Device::Cpu,
        other => return Err(format!("unsupported device {other:?}").into()),
    };
    // Candle's CPU backend does not expose a seedable RNG. The training sampler is
    // still deterministic; Metal initialization is seeded when that backend is used.
    if !matches!(device, Device::Cpu) {
        device.set_seed(options.seed)?;
    }
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = FragmentModel::new(vb)?;
    initialize_model_parameters(&mut varmap, &device, options.seed)?;
    if let Some(initial_model) = &options.initial_model {
        varmap.load(initial_model)?;
    }
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: options.learning_rate,
            weight_decay: 0.0001,
            ..Default::default()
        },
    )?;
    // Longer words and score per cleared line are deliberately strong auxiliary
    // objectives: they force the shared row encoder to model lexical quality
    // instead of spending all of its capacity on generic game progression.
    let loss_weights = Tensor::new(&[[0.4f32, 0.75, 0.2, 0.7, 0.8, 0.2, 1.0]], &device)?;
    let mut indices = train.indices(options.lexicon, options.max_train);
    let validation_indices = validation.indices(options.lexicon, options.max_validation);
    let test_indices = test.indices(options.lexicon, options.max_test);
    let train_count = indices.len();
    let mut random = XorShift64::new(options.seed);
    let best_path = options.output.join("model.safetensors");
    let began = Instant::now();
    let mut best_loss = f64::INFINITY;
    let mut epoch_reports = Vec::new();

    for epoch in 1..=options.epochs {
        random.shuffle(&mut indices);
        let epoch_began = Instant::now();
        let mut loss_sum = 0f64;
        let mut batches = 0u64;
        for batch_indices in indices.chunks(options.batch_size) {
            let host = HostBatch::from_indices(&train, batch_indices, options.input_mode)?;
            let batch = host.to_device(&device)?;
            let predictions = model.forward(&batch)?;
            let loss = predictions
                .sub(&batch.targets)?
                .sqr()?
                .broadcast_mul(&loss_weights)?
                .mean_all()?;
            optimizer.backward_step(&loss)?;
            loss_sum += f64::from(loss.to_vec0::<f32>()?);
            batches += 1;
            if batches.is_multiple_of(500) {
                println!(
                    "{}",
                    json!({
                        "status": "training",
                        "epoch": epoch,
                        "batches": batches,
                        "records": (batches as usize * options.batch_size).min(train_count),
                        "loss": loss_sum / batches as f64,
                        "elapsedSeconds": began.elapsed().as_secs_f64(),
                    })
                );
            }
        }
        let validation_metrics = evaluate(
            &model,
            &validation,
            &validation_indices,
            options.batch_size,
            &device,
            options.input_mode,
        )?;
        if validation_metrics.transformed_loss < best_loss {
            best_loss = validation_metrics.transformed_loss;
            varmap.save(&best_path)?;
        }
        let report = json!({
            "epoch": epoch,
            "trainingLoss": loss_sum / batches.max(1) as f64,
            "validation": validation_metrics,
            "elapsedSeconds": epoch_began.elapsed().as_secs_f64(),
        });
        println!("{report}");
        epoch_reports.push(report);
    }

    varmap.load(&best_path)?;
    let test_metrics = evaluate(
        &model,
        &test,
        &test_indices,
        options.batch_size,
        &device,
        options.input_mode,
    )?;
    let word_input_ablations = if matches!(options.input_mode, InputMode::Full) {
        Some(json!({
            "maskBoardLetters": evaluate(
                &model,
                &test,
                &test_indices,
                options.batch_size,
                &device,
                InputMode::MaskBoardLetters,
            )?,
            "maskFutureLetters": evaluate(
                &model,
                &test,
                &test_indices,
                options.batch_size,
                &device,
                InputMode::MaskFutureLetters,
            )?,
            "maskWordInputs": evaluate(
                &model,
                &test,
                &test_indices,
                options.batch_size,
                &device,
                InputMode::MaskWordInputs,
            )?,
        }))
    } else {
        None
    };
    let report = json!({
        "schemaVersion": 1,
        "architecture": {
            "letterEmbedding": LETTER_EMBED,
            "boundaryEmbedding": BOUNDARY_EMBED,
            "rowInput": ROW_INPUT,
            "rowHidden": ROW_HIDDEN,
            "rowOutput": ROW_OUTPUT,
            "headInput": HEAD_INPUT,
            "headHidden": HEAD_HIDDEN,
            "outputs": [
                "log1p(score4/100)",
                "log1p(score8/100)",
                "log1p(score16/100)",
                "log1p(words8)",
                "log1p(wordLength8)",
                "log1p(lines8)",
                "log1p(scorePerLine8/100)"
            ],
        },
        "options": {
            "data": options.data,
            "epochs": options.epochs,
            "batchSize": options.batch_size,
            "learningRate": options.learning_rate,
            "trainRecords": train_count,
            "validationRecords": validation_indices.len(),
            "testRecords": test_indices.len(),
            "device": options.device,
            "inputMode": options.input_mode,
            "lexicon": options.lexicon,
            "initialModel": options.initial_model,
            "seed": options.seed,
        },
        "epochs": epoch_reports,
        "bestValidationLoss": best_loss,
        "test": test_metrics,
        "wordInputAblations": word_input_ablations,
        "elapsedSeconds": began.elapsed().as_secs_f64(),
    });
    let mut file = File::create(options.output.join("REPORT.json"))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    println!(
        "{}",
        json!({
            "status": "complete",
            "bestValidationLoss": best_loss,
            "testScore8Correlation": test_metrics.score_8.correlation,
            "testScore8Mae": test_metrics.score_8.mean_absolute_error,
            "elapsedSeconds": began.elapsed().as_secs_f64(),
            "output": options.output,
        })
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("kvadrat-fragment-train: {error}");
        std::process::exit(1);
    }
}
