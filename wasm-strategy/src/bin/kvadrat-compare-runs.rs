use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

type AnyError = Box<dyn Error + Send + Sync>;
type Result<T> = std::result::Result<T, AnyError>;

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Episode {
    lexicon: String,
    seed: u32,
    depth: usize,
    score: i32,
    words: u32,
    average_word_length: f64,
    phase: String,
}

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Comparison {
    games: usize,
    baseline_mean_score: f64,
    candidate_mean_score: f64,
    mean_score_delta: f64,
    relative_score_delta: f64,
    score_delta_standard_error: f64,
    score_delta_ci95: [f64; 2],
    median_score_delta: f64,
    wins: usize,
    ties: usize,
    losses: usize,
    baseline_completed: usize,
    candidate_completed: usize,
    mean_words_delta: f64,
    mean_average_word_length_delta: f64,
}

fn absolute(value: &str) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    Ok(if path.is_absolute() {
        path
    } else {
        env::current_dir()?.join(path)
    })
}

fn parse_options() -> Result<(PathBuf, PathBuf)> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.len() != 4 {
        return Err("usage: --baseline RUN --candidate RUN".into());
    }
    let mut baseline = None;
    let mut candidate = None;
    for pair in arguments.chunks_exact(2) {
        match pair[0].as_str() {
            "--baseline" => baseline = Some(absolute(&pair[1])?),
            "--candidate" => candidate = Some(absolute(&pair[1])?),
            other => return Err(format!("unsupported option {other:?}").into()),
        }
    }
    Ok((
        baseline.ok_or("--baseline is required")?,
        candidate.ok_or("--candidate is required")?,
    ))
}

fn load(path: &Path) -> Result<HashMap<u32, Episode>> {
    let file = File::open(path.join("episodes.jsonl"))?;
    let mut episodes = HashMap::new();
    for line in BufReader::new(file).lines() {
        let episode: Episode = serde_json::from_str(&line?)?;
        let seed = episode.seed;
        if episodes.insert(seed, episode).is_some() {
            return Err(format!("{} repeats seed {seed}", path.display()).into());
        }
    }
    Ok(episodes)
}

fn compare(pairs: &[(&Episode, &Episode)]) -> Comparison {
    if pairs.is_empty() {
        return Comparison::default();
    }
    let count = pairs.len() as f64;
    let baseline_score = pairs
        .iter()
        .map(|(baseline, _)| f64::from(baseline.score))
        .sum::<f64>()
        / count;
    let candidate_score = pairs
        .iter()
        .map(|(_, candidate)| f64::from(candidate.score))
        .sum::<f64>()
        / count;
    let mut deltas: Vec<f64> = pairs
        .iter()
        .map(|(baseline, candidate)| f64::from(candidate.score - baseline.score))
        .collect();
    deltas.sort_by(f64::total_cmp);
    let mean_delta = deltas.iter().sum::<f64>() / count;
    let variance = if pairs.len() > 1 {
        deltas
            .iter()
            .map(|delta| (delta - mean_delta).powi(2))
            .sum::<f64>()
            / (count - 1.0)
    } else {
        0.0
    };
    let standard_error = (variance / count).sqrt();
    let median = if deltas.len().is_multiple_of(2) {
        (deltas[deltas.len() / 2 - 1] + deltas[deltas.len() / 2]) / 2.0
    } else {
        deltas[deltas.len() / 2]
    };
    Comparison {
        games: pairs.len(),
        baseline_mean_score: baseline_score,
        candidate_mean_score: candidate_score,
        mean_score_delta: mean_delta,
        relative_score_delta: mean_delta / baseline_score,
        score_delta_standard_error: standard_error,
        score_delta_ci95: [
            mean_delta - 1.96 * standard_error,
            mean_delta + 1.96 * standard_error,
        ],
        median_score_delta: median,
        wins: pairs
            .iter()
            .filter(|(baseline, candidate)| candidate.score > baseline.score)
            .count(),
        ties: pairs
            .iter()
            .filter(|(baseline, candidate)| candidate.score == baseline.score)
            .count(),
        losses: pairs
            .iter()
            .filter(|(baseline, candidate)| candidate.score < baseline.score)
            .count(),
        baseline_completed: pairs
            .iter()
            .filter(|(baseline, _)| baseline.phase == "complete")
            .count(),
        candidate_completed: pairs
            .iter()
            .filter(|(_, candidate)| candidate.phase == "complete")
            .count(),
        mean_words_delta: pairs
            .iter()
            .map(|(baseline, candidate)| f64::from(candidate.words) - f64::from(baseline.words))
            .sum::<f64>()
            / count,
        mean_average_word_length_delta: pairs
            .iter()
            .map(|(baseline, candidate)| {
                candidate.average_word_length - baseline.average_word_length
            })
            .sum::<f64>()
            / count,
    }
}

fn run() -> Result<()> {
    let (baseline_path, candidate_path) = parse_options()?;
    let baseline = load(&baseline_path)?;
    let candidate = load(&candidate_path)?;
    if baseline.len() != candidate.len() {
        return Err("runs contain different game counts".into());
    }
    let mut groups: BTreeMap<String, Vec<(&Episode, &Episode)>> = BTreeMap::new();
    for (seed, baseline_episode) in &baseline {
        let candidate_episode = candidate
            .get(seed)
            .ok_or_else(|| format!("candidate lacks seed {seed}"))?;
        if baseline_episode.depth != candidate_episode.depth
            || baseline_episode.lexicon != candidate_episode.lexicon
        {
            return Err(format!("seed {seed} has mismatched policy metadata").into());
        }
        for key in [
            "all".to_string(),
            format!("depth{}", baseline_episode.depth),
            baseline_episode.lexicon.clone(),
            format!(
                "depth{}-{}",
                baseline_episode.depth, baseline_episode.lexicon
            ),
        ] {
            groups
                .entry(key)
                .or_default()
                .push((baseline_episode, candidate_episode));
        }
    }
    let comparisons: BTreeMap<_, _> = groups
        .into_iter()
        .map(|(name, pairs)| (name, compare(&pairs)))
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "baseline": baseline_path,
            "candidate": candidate_path,
            "groups": comparisons,
        }))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("kvadrat-compare-runs: {error}");
        std::process::exit(1);
    }
}
