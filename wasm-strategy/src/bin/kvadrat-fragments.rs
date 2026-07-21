use flate2::read::MultiGzDecoder;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

const RECORD_BYTES: usize = 272;
const HEADER_BYTES: usize = 64;
const MAGIC: &[u8; 8] = b"KVFRAG1\0";
const PIECES: &str = "IJLOSTZ";

type AnyError = Box<dyn Error + Send + Sync>;
type Result<T> = std::result::Result<T, AnyError>;

#[derive(Clone)]
struct PrepareOptions {
    input: PathBuf,
    output: PathBuf,
    threads: usize,
    max_records: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CorpusManifest {
    status: String,
    aggregate: CorpusAggregate,
    shards: Vec<CorpusShard>,
}

#[derive(Deserialize)]
struct CorpusAggregate {
    positions: u64,
}

#[derive(Clone, Deserialize)]
struct CorpusShard {
    file: String,
    records: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceRecord {
    lexicon: String,
    seed: u32,
    policy: SourcePolicy,
    position: SourcePosition,
    target: SourceTarget,
}

#[derive(Deserialize)]
struct SourcePolicy {
    depth: u8,
}

#[derive(Deserialize)]
struct SourcePosition {
    board: SourceBoard,
    active: SourcePiece,
    next: Vec<SourcePiece>,
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

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceTarget {
    horizons: SourceHorizons,
}

#[derive(Deserialize)]
struct SourceHorizons {
    #[serde(rename = "4")]
    four: SourceDelta,
    #[serde(rename = "8")]
    eight: SourceDelta,
    #[serde(rename = "16")]
    sixteen: SourceDelta,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceDelta {
    score: u32,
    lines: u16,
    words: u16,
    word_length: u16,
}

#[derive(Default, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct SplitStats {
    records: u64,
    csw24_records: u64,
    nwl23_records: u64,
    scoring_records: u64,
    score_4: u64,
    score_8: u64,
    score_16: u64,
    words_8: u64,
    lines_8: u64,
}

impl SplitStats {
    fn add_record(&mut self, record: &SourceRecord) {
        self.records += 1;
        if record.lexicon == "CSW24" {
            self.csw24_records += 1;
        } else if record.lexicon == "NWL23" {
            self.nwl23_records += 1;
        }
        self.scoring_records += u64::from(record.target.horizons.eight.score > 0);
        self.score_4 += record.target.horizons.four.score as u64;
        self.score_8 += record.target.horizons.eight.score as u64;
        self.score_16 += record.target.horizons.sixteen.score as u64;
        self.words_8 += record.target.horizons.eight.words as u64;
        self.lines_8 += record.target.horizons.eight.lines as u64;
    }

    fn merge(&mut self, other: Self) {
        self.records += other.records;
        self.csw24_records += other.csw24_records;
        self.nwl23_records += other.nwl23_records;
        self.scoring_records += other.scoring_records;
        self.score_4 += other.score_4;
        self.score_8 += other.score_8;
        self.score_16 += other.score_16;
        self.words_8 += other.words_8;
        self.lines_8 += other.lines_8;
    }
}

struct ShardOutput {
    train: Vec<u8>,
    validation: Vec<u8>,
    test: Vec<u8>,
    stats: [SplitStats; 3],
}

struct SplitWriter {
    file: BufWriter<File>,
    records: u64,
}

impl SplitWriter {
    fn create(path: &Path) -> Result<Self> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&[0; HEADER_BYTES])?;
        Ok(Self { file, records: 0 })
    }

    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        if !bytes.len().is_multiple_of(RECORD_BYTES) {
            return Err("compact shard is not record-aligned".into());
        }
        self.file.write_all(bytes)?;
        self.records += (bytes.len() / RECORD_BYTES) as u64;
        Ok(())
    }

    fn finish(mut self) -> Result<u64> {
        self.file.flush()?;
        let mut file = self.file.into_inner()?;
        file.seek(SeekFrom::Start(0))?;
        let mut header = [0u8; HEADER_BYTES];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&1u32.to_le_bytes());
        header[12..16].copy_from_slice(&(RECORD_BYTES as u32).to_le_bytes());
        header[16..24].copy_from_slice(&self.records.to_le_bytes());
        file.write_all(&header)?;
        file.sync_all()?;
        Ok(self.records)
    }
}

fn parse_pairs(arguments: &[String]) -> Result<HashMap<String, String>> {
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
    Ok(values)
}

fn prepare_options(arguments: &[String]) -> Result<PrepareOptions> {
    let values = parse_pairs(arguments)?;
    let input = values.get("input").ok_or("prepare requires --input")?;
    let output = values.get("output").ok_or("prepare requires --output")?;
    let hardware_threads = thread::available_parallelism().map_or(1, usize::from);
    Ok(PrepareOptions {
        input: fs::canonicalize(input)?,
        output: absolute_path(output)?,
        threads: values
            .get("threads")
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(hardware_threads.saturating_sub(1).max(1)),
        max_records: values
            .get("max-records")
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(u64::MAX),
    })
}

fn absolute_path(value: &str) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(env::current_dir()?.join(path))
    }
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

fn encode_piece(piece: &SourcePiece, output: &mut [u8]) -> Result<()> {
    if output.len() != 5 || piece.letters.len() != 4 {
        return Err("piece encoding requires one shape and four letters".into());
    }
    output[0] = piece_id(&piece.piece)?;
    for (destination, &letter) in output[1..].iter_mut().zip(piece.letters.as_bytes()) {
        *destination = letter_id(letter)?;
    }
    Ok(())
}

fn encode_record(record: &SourceRecord) -> Result<[u8; RECORD_BYTES]> {
    if record.position.board.letters.len() != 22
        || record.position.board.pieces.len() != 22
        || record.position.next.len() != 4
    {
        return Err("invalid board or visible-piece dimensions".into());
    }
    let mut output = [0u8; RECORD_BYTES];
    for row in 0..22 {
        let letters = record.position.board.letters[row].as_bytes();
        let pieces = record.position.board.pieces[row].as_bytes();
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
                letter_id(letters[col])?
                    | ((piece_id(&String::from_utf8_lossy(&pieces[col..col + 1]))? + 1) << 5)
            };
        }
    }
    encode_piece(&record.position.active, &mut output[220..225])?;
    for (index, piece) in record.position.next.iter().enumerate() {
        let start = 225 + index * 5;
        encode_piece(piece, &mut output[start..start + 5])?;
    }
    output[245] = record.position.current.lines;
    output[246] = match record.lexicon.as_str() {
        "CSW24" => 0,
        "NWL23" => 1,
        other => return Err(format!("unknown lexicon {other}").into()),
    };
    output[247] = record.policy.depth;
    output[248..252].copy_from_slice(&record.seed.to_le_bytes());
    output[252..256].copy_from_slice(&record.target.horizons.four.score.to_le_bytes());
    output[256..260].copy_from_slice(&record.target.horizons.eight.score.to_le_bytes());
    output[260..264].copy_from_slice(&record.target.horizons.sixteen.score.to_le_bytes());
    output[264..266].copy_from_slice(&record.target.horizons.eight.words.to_le_bytes());
    output[266..268].copy_from_slice(&record.target.horizons.eight.lines.to_le_bytes());
    output[268..270].copy_from_slice(&record.target.horizons.eight.word_length.to_le_bytes());
    Ok(output)
}

fn split_index(seed: u32) -> usize {
    // The game seed advances by an odd constant while lexica alternate by game
    // index. Using seed % 10 therefore leaks lexicon parity into the split.
    // Avalanche the episode seed first so every lexicon reaches every split.
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

fn process_shard(root: &Path, shard: &CorpusShard, remaining: &AtomicUsize) -> Result<ShardOutput> {
    let file = File::open(root.join(&shard.file))?;
    let decoder = MultiGzDecoder::new(file);
    let reader = BufReader::with_capacity(1024 * 1024, decoder);
    let mut buffers = [Vec::new(), Vec::new(), Vec::new()];
    let mut stats = [SplitStats::default(); 3];
    let mut decoded = 0usize;
    for line in reader.lines() {
        let reservation = remaining.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_sub(1)
        });
        if reservation.is_err() {
            break;
        }
        let record: SourceRecord = serde_json::from_str(&line?)?;
        let split = split_index(record.seed);
        buffers[split].extend_from_slice(&encode_record(&record)?);
        stats[split].add_record(&record);
        decoded += 1;
    }
    if remaining.load(Ordering::Relaxed) > 0 && decoded != shard.records {
        return Err(format!(
            "{}: expected {} records, decoded {decoded}",
            shard.file, shard.records
        )
        .into());
    }
    let [train, validation, test] = buffers;
    Ok(ShardOutput {
        train,
        validation,
        test,
        stats,
    })
}

fn sha256(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn prepare(options: PrepareOptions) -> Result<()> {
    if options.threads == 0 {
        return Err("--threads must be at least one".into());
    }
    let manifest_path = options.input.join("manifest.json");
    let manifest: CorpusManifest = serde_json::from_reader(File::open(&manifest_path)?)?;
    if manifest.status != "complete" {
        return Err(format!("source corpus status is {}", manifest.status).into());
    }
    fs::create_dir_all(options.output.parent().ok_or("output has no parent")?)?;
    fs::create_dir(&options.output)?;
    let began = Instant::now();
    let remaining = Arc::new(AtomicUsize::new(
        usize::try_from(options.max_records).unwrap_or(usize::MAX),
    ));
    let shards = Arc::new(manifest.shards.clone());
    let next_shard = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::sync_channel::<Result<ShardOutput>>(options.threads * 2);
    let mut handles = Vec::new();
    for _ in 0..options.threads {
        let root = options.input.clone();
        let remaining = Arc::clone(&remaining);
        let shards = Arc::clone(&shards);
        let next_shard = Arc::clone(&next_shard);
        let sender = sender.clone();
        handles.push(thread::spawn(move || {
            loop {
                if remaining.load(Ordering::Relaxed) == 0 {
                    break;
                }
                let index = next_shard.fetch_add(1, Ordering::Relaxed);
                let Some(shard) = shards.get(index) else {
                    break;
                };
                if sender
                    .send(process_shard(&root, shard, &remaining))
                    .is_err()
                {
                    break;
                }
            }
        }));
    }
    drop(sender);

    let mut writers = [
        SplitWriter::create(&options.output.join("train.kvf"))?,
        SplitWriter::create(&options.output.join("validation.kvf"))?,
        SplitWriter::create(&options.output.join("test.kvf"))?,
    ];
    let mut totals = [SplitStats::default(); 3];
    let mut completed_shards = 0usize;
    for result in receiver {
        let output = result?;
        writers[0].append(&output.train)?;
        writers[1].append(&output.validation)?;
        writers[2].append(&output.test)?;
        for (total, stats) in totals.iter_mut().zip(output.stats) {
            total.merge(stats);
        }
        completed_shards += 1;
        if completed_shards.is_multiple_of(100) {
            println!(
                "{}",
                json!({
                    "status": "preparing",
                    "shards": completed_shards,
                    "records": totals.iter().map(|stats| stats.records).sum::<u64>(),
                    "elapsedSeconds": began.elapsed().as_secs(),
                })
            );
        }
    }
    for handle in handles {
        if handle.join().is_err() {
            return Err("fragment preparation worker panicked".into());
        }
    }
    let [train_writer, validation_writer, test_writer] = writers;
    let file_counts = [
        train_writer.finish()?,
        validation_writer.finish()?,
        test_writer.finish()?,
    ];
    for (count, stats) in file_counts.into_iter().zip(totals.iter()) {
        if count != stats.records {
            return Err("compact split count does not match its statistics".into());
        }
    }
    let prepared = totals.iter().map(|stats| stats.records).sum::<u64>();
    let metadata = json!({
        "schemaVersion": 1,
        "recordBytes": RECORD_BYTES,
        "source": options.input,
        "sourceManifestSha256": sha256(&manifest_path)?,
        "sourcePositions": manifest.aggregate.positions,
        "preparedPositions": prepared,
        "split": "hashed episode seed modulo 10: 0-7 train, 8 validation, 9 test",
        "excludedInputs": [
            "position.features",
            "action.evaluation",
            "policy.beamWidth",
            "policy.depth (stored only for stratified audits)"
        ],
        "inputs": {
            "board": "220 packed letter/color cells",
            "visiblePieces": "active plus four previews; one shape byte and four letter bytes each",
            "context": "current line count and lexicon; policy depth is stored only for stratified audits",
        },
        "targets": ["score4", "score8", "score16", "words8", "lines8", "wordLength8"],
        "splits": {
            "train": totals[0],
            "validation": totals[1],
            "test": totals[2],
        },
        "elapsedSeconds": began.elapsed().as_secs_f64(),
        "threads": options.threads,
    });
    let mut metadata_file = BufWriter::new(File::create(options.output.join("MANIFEST.json"))?);
    serde_json::to_writer_pretty(&mut metadata_file, &metadata)?;
    metadata_file.write_all(b"\n")?;
    println!(
        "{}",
        json!({
            "status": "complete",
            "records": prepared,
            "train": totals[0].records,
            "validation": totals[1].records,
            "test": totals[2].records,
            "elapsedSeconds": began.elapsed().as_secs_f64(),
            "output": options.output,
        })
    );
    Ok(())
}

fn run() -> Result<()> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let (command, rest) = arguments
        .split_first()
        .ok_or("usage: kvadrat-fragments prepare --input CORPUS --output DATASET [--threads N]")?;
    match command.as_str() {
        "prepare" => prepare(prepare_options(rest)?),
        other => Err(format!("unknown fragment command {other:?}").into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("kvadrat-fragments: {error}");
        std::process::exit(1);
    }
}
