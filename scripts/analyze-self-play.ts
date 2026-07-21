import { readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { gunzipSync } from "node:zlib";

type NumericSummary = {
  count: number;
  mean: number;
  standardDeviation: number;
  minimum: number;
  p10: number;
  median: number;
  p90: number;
  maximum: number;
  positiveRate: number;
};

type Accumulator = {
  count: number;
  sum: number;
  sumSquares: number;
  minimum: number;
  maximum: number;
  positive: number;
  sample: number[];
};

type GroupStats = {
  episodes: Set<string>;
  positions: number;
  score: Accumulator;
  lines: Accumulator;
  words: Accumulator;
};

type Correlation = {
  count: number;
  sumX: number;
  sumY: number;
  sumXX: number;
  sumYY: number;
  sumXY: number;
};

type EpisodeState = {
  lastStep: number;
  positions: number;
  seed: number;
  lexicon: string;
  depth: number;
  terminalScore: number;
  terminalLines: number;
  completed: boolean;
};

const SAMPLE_LIMIT = 100_000;

function parseArguments() {
  const values = new Map<string, string>();
  for (let index = 2; index < process.argv.length; index += 2) {
    const key = process.argv[index];
    const value = process.argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error("Usage: npm run analyze-self-play -- --input PATH [--allow-running true]");
    }
    values.set(key.slice(2), value);
  }
  const input = values.get("input");
  if (!input) throw new Error("--input is required.");
  return {
    input: resolve(input),
    allowRunning: values.get("allow-running") === "true",
    write: values.get("write") !== "false",
  };
}

function accumulator(): Accumulator {
  return {
    count: 0,
    sum: 0,
    sumSquares: 0,
    minimum: Number.POSITIVE_INFINITY,
    maximum: Number.NEGATIVE_INFINITY,
    positive: 0,
    sample: [],
  };
}

function mix32(value: number): number {
  let mixed = value | 0;
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x45d9f3b);
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x45d9f3b);
  return (mixed ^ (mixed >>> 16)) >>> 0;
}

function addValue(target: Accumulator, value: number): void {
  if (!Number.isFinite(value)) return;
  target.count += 1;
  target.sum += value;
  target.sumSquares += value * value;
  target.minimum = Math.min(target.minimum, value);
  target.maximum = Math.max(target.maximum, value);
  target.positive += Number(value > 0);
  if (target.sample.length < SAMPLE_LIMIT) {
    target.sample.push(value);
  } else {
    const replacement = mix32(target.count) % target.count;
    if (replacement < SAMPLE_LIMIT) target.sample[replacement] = value;
  }
}

function quantile(sorted: number[], fraction: number): number {
  if (sorted.length === 0) return 0;
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function summarize(target: Accumulator): NumericSummary {
  const sorted = [...target.sample].sort((left, right) => left - right);
  const mean = target.count ? target.sum / target.count : 0;
  const variance = target.count ? Math.max(0, target.sumSquares / target.count - mean * mean) : 0;
  return {
    count: target.count,
    mean,
    standardDeviation: Math.sqrt(variance),
    minimum: target.count ? target.minimum : 0,
    p10: quantile(sorted, 0.1),
    median: quantile(sorted, 0.5),
    p90: quantile(sorted, 0.9),
    maximum: target.count ? target.maximum : 0,
    positiveRate: target.count ? target.positive / target.count : 0,
  };
}

function groupStats(): GroupStats {
  return {
    episodes: new Set(),
    positions: 0,
    score: accumulator(),
    lines: accumulator(),
    words: accumulator(),
  };
}

function addGroup(target: GroupStats, record: Record<string, unknown>): void {
  const episodeId = String(record.episodeId);
  const labels = record.target as Record<string, number>;
  const isNewEpisode = !target.episodes.has(episodeId);
  target.episodes.add(episodeId);
  target.positions += 1;
  if (isNewEpisode) {
    addValue(target.score, labels.terminalScore);
    addValue(target.lines, labels.terminalLines);
    addValue(target.words, labels.wordsToGo + ((record.position as Record<string, Record<string, number>>).current.words));
  }
}

function serializeGroup(target: GroupStats) {
  return {
    episodes: target.episodes.size,
    positions: target.positions,
    terminalScore: summarize(target.score),
    terminalLines: summarize(target.lines),
    terminalWords: summarize(target.words),
  };
}

function correlation(): Correlation {
  return { count: 0, sumX: 0, sumY: 0, sumXX: 0, sumYY: 0, sumXY: 0 };
}

function addCorrelation(target: Correlation, x: number, y: number): void {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  target.count += 1;
  target.sumX += x;
  target.sumY += y;
  target.sumXX += x * x;
  target.sumYY += y * y;
  target.sumXY += x * y;
}

function correlationValue(target: Correlation): number {
  const numerator = target.count * target.sumXY - target.sumX * target.sumY;
  const denominator = Math.sqrt(
    (target.count * target.sumXX - target.sumX ** 2) *
    (target.count * target.sumYY - target.sumY ** 2),
  );
  return denominator ? numerator / denominator : 0;
}

function positionHash(record: Record<string, unknown>): bigint {
  const position = record.position as {
    board: { letters: string[]; pieces: string[] };
    active: { piece: string; letters: string };
    next: Array<{ piece: string; letters: string }>;
  };
  const text = [
    record.lexicon,
    ...position.board.letters,
    ...position.board.pieces,
    position.active.piece,
    position.active.letters,
    ...position.next.flatMap((piece) => [piece.piece, piece.letters]),
  ].join("|");
  let left = 0x811c9dc5;
  let right = 0x9e3779b9;
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    left = Math.imul(left ^ code, 0x01000193);
    right = Math.imul(right ^ (code + index), 0x85ebca6b);
  }
  return (BigInt(left >>> 0) << 32n) | BigInt(right >>> 0);
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function rounded(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}

const options = parseArguments();
const manifest = JSON.parse(await readFile(resolve(options.input, "manifest.json"), "utf8"));
if (manifest.status !== "complete" && !options.allowRunning) {
  throw new Error(`Manifest is ${manifest.status}; pass --allow-running true for a finalized-shards-only audit.`);
}

const shardEntries = manifest.status === "complete"
  ? manifest.shards
  : manifest.shards;
const expectedPositions = shardEntries.reduce((sum: number, shard: { records: number }) => sum + shard.records, 0);
const errors: string[] = [];
let invalidRecords = 0;
let decodedPositions = 0;
let duplicatePositions = 0;
let compressedBytes = 0;
let uncompressedBytes = 0;
const seenPositions = new Set<bigint>();
const episodes = new Map<string, EpisodeState>();
const byDepth = new Map<string, GroupStats>();
const byLexicon = new Map<string, GroupStats>();
const splitEpisodes = { train: new Set<string>(), validation: new Set<string>(), test: new Set<string>() };
const splitPositions = { train: 0, validation: 0, test: 0 };
const metrics = {
  scoreToGo: accumulator(),
  scorePerLineToGo: accumulator(),
  score4: accumulator(),
  score8: accumulator(),
  score16: accumulator(),
  words4: accumulator(),
  words8: accumulator(),
  words16: accumulator(),
  wordPotential: accumulator(),
  heuristicValue: accumulator(),
  maximumHeight: accumulator(),
  holes: accumulator(),
};
const correlations = {
  heuristicToScore16: correlation(),
  wordPotentialToScore16: correlation(),
  searchEvaluationToScore16: correlation(),
};

function recordError(message: string): void {
  invalidRecords += 1;
  if (errors.length < 100) errors.push(message);
}

for (const shard of shardEntries as Array<{ file: string; records: number; bytes: number }>) {
  const compressed = await readFile(resolve(options.input, shard.file));
  compressedBytes += compressed.byteLength;
  let raw: string;
  try {
    raw = gunzipSync(compressed).toString();
  } catch (error) {
    errors.push(`${shard.file}: gzip decode failed: ${error instanceof Error ? error.message : String(error)}`);
    continue;
  }
  uncompressedBytes += Buffer.byteLength(raw);
  const lines = raw.trim().split("\n").filter(Boolean);
  if (lines.length !== shard.records) {
    errors.push(`${shard.file}: manifest says ${shard.records} records, decoded ${lines.length}`);
  }

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    let record: Record<string, unknown>;
    try {
      record = JSON.parse(lines[lineIndex]);
    } catch (error) {
      recordError(`${shard.file}:${lineIndex + 1}: invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
      continue;
    }
    decodedPositions += 1;
    const episodeId = String(record.episodeId);
    const step = Number(record.step);
    const seed = Number(record.seed);
    const policy = record.policy as { depth: number; beamWidth: number };
    const position = record.position as {
      board: { letters: string[]; pieces: string[] };
      active: { piece: string; letters: string };
      next: unknown[];
      current: { score: number; lines: number; words: number; totalWordLength: number };
      features: {
        heuristicValue: number;
        wordPotential: number;
        maximumHeight: number;
        holes: number;
      };
    };
    const action = record.action as { depth: number; evaluation: number };
    const target = record.target as {
      completed: boolean;
      toppedOut: boolean;
      terminalScore: number;
      terminalLines: number;
      scoreToGo: number;
      linesToGo: number;
      wordsToGo: number;
      scorePerLineToGo: number;
      horizons: Record<"4" | "8" | "16", { score: number; lines: number; words: number; wordLength: number }>;
    };

    const boardValid = position?.board?.letters?.length === 22 &&
      position.board.pieces?.length === 22 &&
      position.board.letters.every((row) => /^[A-Z.]{10}$/.test(row)) &&
      position.board.pieces.every((row) => /^[IJLOSTZ.]{10}$/.test(row));
    const queueValid = position?.active?.letters?.length === 4 && position?.next?.length === 4;
    const targetValid = target && target.scoreToGo === target.terminalScore - position.current.score &&
      target.linesToGo === target.terminalLines - position.current.lines &&
      target.scoreToGo >= 0 && target.linesToGo >= 0 && target.wordsToGo >= 0 &&
      target.horizons["4"].score <= target.horizons["8"].score &&
      target.horizons["8"].score <= target.horizons["16"].score &&
      target.horizons["4"].words <= target.horizons["8"].words &&
      target.horizons["8"].words <= target.horizons["16"].words;
    const searchDepthValid = action.depth >= 1 && action.depth <= policy.depth;
    if (record.schemaVersion !== 1 || !boardValid || !queueValid || !targetValid || !searchDepthValid) {
      recordError(`${shard.file}:${lineIndex + 1}: schema or label invariant failed`);
    }

    const previous = episodes.get(episodeId);
    if (previous && step !== previous.lastStep + 1) {
      recordError(`${episodeId}: expected step ${previous.lastStep + 1}, found ${step}`);
    }
    if (previous && (previous.terminalScore !== target.terminalScore ||
      previous.terminalLines !== target.terminalLines || previous.seed !== seed)) {
      recordError(`${episodeId}: terminal labels changed within the episode`);
    }
    episodes.set(episodeId, {
      lastStep: step,
      positions: (previous?.positions ?? 0) + 1,
      seed,
      lexicon: String(record.lexicon),
      depth: policy.depth,
      terminalScore: target.terminalScore,
      terminalLines: target.terminalLines,
      completed: target.completed,
    });

    const hash = positionHash(record);
    duplicatePositions += Number(seenPositions.has(hash));
    seenPositions.add(hash);

    const depthKey = String(policy.depth);
    const lexiconKey = String(record.lexicon);
    if (!byDepth.has(depthKey)) byDepth.set(depthKey, groupStats());
    if (!byLexicon.has(lexiconKey)) byLexicon.set(lexiconKey, groupStats());
    addGroup(byDepth.get(depthKey)!, record);
    addGroup(byLexicon.get(lexiconKey)!, record);

    const split = seed % 10 === 9 ? "test" : seed % 10 === 8 ? "validation" : "train";
    splitEpisodes[split].add(episodeId);
    splitPositions[split] += 1;

    addValue(metrics.scoreToGo, target.scoreToGo);
    addValue(metrics.scorePerLineToGo, target.scorePerLineToGo);
    addValue(metrics.score4, target.horizons["4"].score);
    addValue(metrics.score8, target.horizons["8"].score);
    addValue(metrics.score16, target.horizons["16"].score);
    addValue(metrics.words4, target.horizons["4"].words);
    addValue(metrics.words8, target.horizons["8"].words);
    addValue(metrics.words16, target.horizons["16"].words);
    addValue(metrics.wordPotential, position.features.wordPotential);
    addValue(metrics.heuristicValue, position.features.heuristicValue);
    addValue(metrics.maximumHeight, position.features.maximumHeight);
    addValue(metrics.holes, position.features.holes);
    addCorrelation(correlations.heuristicToScore16, position.features.heuristicValue, target.horizons["16"].score);
    addCorrelation(correlations.wordPotentialToScore16, position.features.wordPotential, target.horizons["16"].score);
    addCorrelation(correlations.searchEvaluationToScore16, action.evaluation, target.horizons["16"].score);
  }
}

if (decodedPositions !== expectedPositions) {
  errors.push(`Expected ${expectedPositions} finalized positions, decoded ${decodedPositions}`);
}

const episodeLines = (await readFile(resolve(options.input, "episodes.jsonl"), "utf8"))
  .trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
const finalizedEpisodeIds = new Set(episodes.keys());
const summaries = episodeLines.filter((summary) => finalizedEpisodeIds.has(summary.episodeId));
if (summaries.length !== episodes.size) {
  errors.push(`Found ${summaries.length} matching episode summaries for ${episodes.size} decoded episodes`);
}
for (const summary of summaries) {
  const episode = episodes.get(summary.episodeId)!;
  if (summary.positions !== episode.positions || summary.score !== episode.terminalScore ||
    summary.lines !== episode.terminalLines || summary.depth !== episode.depth ||
    summary.lexicon !== episode.lexicon) {
    errors.push(`${summary.episodeId}: episode summary does not match position labels`);
  }
}

const results = {
  generatedAt: new Date().toISOString(),
  corpus: basename(options.input),
  manifestStatus: manifest.status,
  startedAt: manifest.startedAt,
  deadline: manifest.deadline,
  completedAt: manifest.completedAt,
  volume: {
    episodes: episodes.size,
    positions: decodedPositions,
    shards: shardEntries.length,
    compressedBytes,
    uncompressedBytes,
    compressionRatio: uncompressedBytes ? compressedBytes / uncompressedBytes : 0,
    uniquePositionHashes: seenPositions.size,
    duplicatePositions,
    duplicateRate: decodedPositions ? duplicatePositions / decodedPositions : 0,
  },
  quality: {
    valid: errors.length === 0 && invalidRecords === 0,
    invalidRecords,
    errors,
    decodedMatchesManifest: decodedPositions === expectedPositions,
    episodeSummariesMatch: summaries.length === episodes.size,
  },
  outcomes: {
    completedEpisodes: [...episodes.values()].filter((episode) => episode.completed).length,
    topOutEpisodes: [...episodes.values()].filter((episode) => !episode.completed).length,
  },
  byDepth: Object.fromEntries([...byDepth.entries()].map(([key, value]) => [key, serializeGroup(value)])),
  byLexicon: Object.fromEntries([...byLexicon.entries()].map(([key, value]) => [key, serializeGroup(value)])),
  splits: {
    method: "episode seed modulo 10: 0-7 train, 8 validation, 9 test",
    train: { episodes: splitEpisodes.train.size, positions: splitPositions.train },
    validation: { episodes: splitEpisodes.validation.size, positions: splitPositions.validation },
    test: { episodes: splitEpisodes.test.size, positions: splitPositions.test },
  },
  labels: Object.fromEntries(Object.entries(metrics).map(([key, value]) => [key, summarize(value)])),
  correlations: {
    heuristicValueToScoreNext16: correlationValue(correlations.heuristicToScore16),
    wordPotentialToScoreNext16: correlationValue(correlations.wordPotentialToScore16),
    searchEvaluationToScoreNext16: correlationValue(correlations.searchEvaluationToScore16),
  },
};

const markdown = `# Kvadrat self-play corpus quality report

Generated: ${results.generatedAt}

## Verdict

${results.quality.valid ? "**PASS** — every finalized shard, record, episode sequence, and target relationship passed validation." : `**FAIL** — ${errors.length} corpus errors and ${invalidRecords} invalid records were found.`}

## Volume

| Metric | Value |
| --- | ---: |
| Status | ${results.manifestStatus} |
| Episodes | ${rounded(results.volume.episodes)} |
| Positions | ${rounded(results.volume.positions)} |
| Shards | ${rounded(results.volume.shards)} |
| Compressed size | ${(results.volume.compressedBytes / 1048576).toFixed(2)} MiB |
| Compression ratio | ${percentage(results.volume.compressionRatio)} |
| Duplicate-position rate | ${percentage(results.volume.duplicateRate)} |
| Completed episodes | ${rounded(results.outcomes.completedEpisodes)} |
| Top-outs | ${rounded(results.outcomes.topOutEpisodes)} |

## Policy and lexicon coverage

| Group | Episodes | Positions | Mean terminal score |
| --- | ---: | ---: | ---: |
${Object.entries(results.byDepth).map(([key, value]) => `| Depth ${key} | ${rounded(value.episodes)} | ${rounded(value.positions)} | ${rounded(value.terminalScore.mean)} |`).join("\n")}
${Object.entries(results.byLexicon).map(([key, value]) => `| ${key} | ${rounded(value.episodes)} | ${rounded(value.positions)} | ${rounded(value.terminalScore.mean)} |`).join("\n")}

## Label health

| Label | Mean | Median | P90 | Nonzero |
| --- | ---: | ---: | ---: | ---: |
| Score next 4 | ${rounded(results.labels.score4.mean)} | ${rounded(results.labels.score4.median)} | ${rounded(results.labels.score4.p90)} | ${percentage(results.labels.score4.positiveRate)} |
| Score next 8 | ${rounded(results.labels.score8.mean)} | ${rounded(results.labels.score8.median)} | ${rounded(results.labels.score8.p90)} | ${percentage(results.labels.score8.positiveRate)} |
| Score next 16 | ${rounded(results.labels.score16.mean)} | ${rounded(results.labels.score16.median)} | ${rounded(results.labels.score16.p90)} | ${percentage(results.labels.score16.positiveRate)} |
| Terminal score-to-go | ${rounded(results.labels.scoreToGo.mean)} | ${rounded(results.labels.scoreToGo.median)} | ${rounded(results.labels.scoreToGo.p90)} | ${percentage(results.labels.scoreToGo.positiveRate)} |

## Leakage-safe split

Split by episode seed, never by individual position.

| Split | Episodes | Positions |
| --- | ---: | ---: |
| Train | ${rounded(results.splits.train.episodes)} | ${rounded(results.splits.train.positions)} |
| Validation | ${rounded(results.splits.validation.episodes)} | ${rounded(results.splits.validation.positions)} |
| Test | ${rounded(results.splits.test.episodes)} | ${rounded(results.splits.test.positions)} |

## Recommended first model

1. Encode the 22×10 letter plane and tetromino-color plane separately; condition on active/next pieces, lexicon, cleared lines, and search depth.
2. Use a multi-task loss for 4/8/16-placement score and word returns plus terminal score-per-line. The shorter horizons provide denser supervision than the terminal label alone.
3. Weight examples inversely by search depth and lexicon frequency if the final counts differ materially.
4. Keep the episode-seed split above to prevent adjacent states from the same game leaking across train and evaluation.
5. Treat this as on-policy bootstrapping data. After the first model is integrated, generate a second corpus with top-k exploration and compare candidate rankings on held-out episodes.

## Baseline correlations

| Existing signal | Correlation with score in next 16 placements |
| --- | ---: |
| Handwritten leaf heuristic | ${results.correlations.heuristicValueToScoreNext16.toFixed(4)} |
| Handwritten word potential | ${results.correlations.wordPotentialToScoreNext16.toFixed(4)} |
| Beam-search evaluation | ${results.correlations.searchEvaluationToScoreNext16.toFixed(4)} |
`;

if (options.write) {
  await writeFile(resolve(options.input, "analysis.json"), `${JSON.stringify(results, null, 2)}\n`);
  await writeFile(resolve(options.input, "QUALITY_REPORT.md"), markdown);
}

console.log(JSON.stringify({
  valid: results.quality.valid,
  status: results.manifestStatus,
  episodes: results.volume.episodes,
  positions: results.volume.positions,
  shards: results.volume.shards,
  errors: results.quality.errors.length,
  invalidRecords: results.quality.invalidRecords,
  report: options.write ? resolve(options.input, "QUALITY_REPORT.md") : null,
}));

if (!results.quality.valid) process.exitCode = 1;
