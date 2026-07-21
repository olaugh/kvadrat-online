import { appendFile, mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { gzipSync } from "node:zlib";
import { KvadratGame } from "../app/game-engine.ts";
import type {
  BotPlan,
  GameAssets,
  GameSnapshot,
  LexiconId,
  TrainingPosition,
} from "../app/game-engine.ts";

type Options = {
  hours: number;
  maxGames: number;
  output: string;
  seed: number;
  shardRecords: number;
  depths: number[];
};

type CurrentMetrics = TrainingPosition["current"];

type UnlabelledRecord = {
  schemaVersion: 1;
  episodeId: string;
  step: number;
  lexicon: LexiconId;
  seed: number;
  policy: { depth: number; beamWidth: number };
  position: TrainingPosition;
  action: BotPlan;
};

type TrainingRecord = UnlabelledRecord & {
  target: {
    completed: boolean;
    toppedOut: boolean;
    terminalScore: number;
    terminalLines: number;
    scoreToGo: number;
    linesToGo: number;
    wordsToGo: number;
    wordLengthToGo: number;
    scorePerLineToGo: number;
    horizons: Record<"4" | "8" | "16", {
      score: number;
      lines: number;
      words: number;
      wordLength: number;
    }>;
  };
};

type EpisodeSummary = {
  episodeId: string;
  lexicon: LexiconId;
  seed: number;
  depth: number;
  beamWidth: number;
  positions: number;
  score: number;
  lines: number;
  words: number;
  averageWordLength: number;
  phase: GameSnapshot["phase"];
  searchNodes: number;
  elapsedMs: number;
};

type Aggregate = {
  episodes: number;
  positions: number;
  completed: number;
  toppedOut: number;
  score: number;
  lines: number;
  words: number;
  searchNodes: number;
};

function parseOptions(): Options {
  const values = new Map<string, string>();
  for (let index = 2; index < process.argv.length; index += 2) {
    const key = process.argv[index];
    const value = process.argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Expected --name value arguments; received ${key ?? "nothing"}.`);
    }
    values.set(key.slice(2), value);
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const depths = (values.get("depths") ?? "2,3,4")
    .split(",")
    .map(Number)
    .filter((depth) => Number.isInteger(depth) && depth >= 1 && depth <= 5);
  if (depths.length === 0) throw new Error("At least one search depth is required.");

  return {
    hours: Number(values.get("hours") ?? "8"),
    maxGames: Number(values.get("games") ?? String(Number.MAX_SAFE_INTEGER)),
    output: resolve(values.get("output") ?? `training-data/selfplay-${stamp}`),
    seed: Number(values.get("seed") ?? "12628309") >>> 0,
    shardRecords: Number(values.get("shard-records") ?? "2000"),
    depths,
  };
}

function seededRandom(seed: number): () => number {
  let state = seed || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

function gameSeed(baseSeed: number, gameIndex: number): number {
  return (baseSeed + Math.imul(gameIndex + 1, 0x9e3779b1)) >>> 0;
}

function beamWidth(depth: number): number {
  if (depth <= 2) return 48;
  if (depth === 3) return 64;
  return 72;
}

async function loadAssets(lexicon: LexiconId): Promise<GameAssets> {
  const bagName = lexicon.toLowerCase();
  const [kwgBuffer, bagsText] = await Promise.all([
    readFile(new URL(`../public/data/${lexicon}.kwg`, import.meta.url)),
    readFile(new URL(`../public/data/${bagName}-bags.txt`, import.meta.url), "utf8"),
  ]);
  const view = new DataView(kwgBuffer.buffer, kwgBuffer.byteOffset, kwgBuffer.byteLength);
  const kwg = new Uint32Array(kwgBuffer.byteLength / 4);
  for (let index = 0; index < kwg.length; index += 1) {
    kwg[index] = view.getUint32(index * 4, true);
  }
  const wordBags = bagsText
    .split(/\r?\n/)
    .map((line) => line.trim().split(/\s+/).filter(Boolean))
    .filter((bag) => bag.length >= 28)
    .map((bag) => bag.slice(0, 28));
  return { kwg, wordBags };
}

function delta(from: CurrentMetrics, to: CurrentMetrics) {
  return {
    score: to.score - from.score,
    lines: to.lines - from.lines,
    words: to.words - from.words,
    wordLength: to.totalWordLength - from.totalWordLength,
  };
}

function labelTrajectory(
  records: UnlabelledRecord[],
  finalSnapshot: GameSnapshot,
): TrainingRecord[] {
  const finalMetrics: CurrentMetrics = {
    score: finalSnapshot.score,
    lines: finalSnapshot.lines,
    pieces: finalSnapshot.pieces,
    words: finalSnapshot.words,
    totalWordLength: Math.round(finalSnapshot.averageWordLength * finalSnapshot.words),
  };
  const metricAfter = (index: number, moves: number): CurrentMetrics =>
    records[index + moves]?.position.current ?? finalMetrics;

  return records.map((record, index) => {
    const current = record.position.current;
    const outcome = delta(current, finalMetrics);
    return {
      ...record,
      target: {
        completed: finalSnapshot.phase === "complete",
        toppedOut: finalSnapshot.phase === "over",
        terminalScore: finalSnapshot.score,
        terminalLines: finalSnapshot.lines,
        scoreToGo: outcome.score,
        linesToGo: outcome.lines,
        wordsToGo: outcome.words,
        wordLengthToGo: outcome.wordLength,
        scorePerLineToGo: outcome.lines > 0 ? outcome.score / outcome.lines : 0,
        horizons: {
          "4": delta(current, metricAfter(index, 4)),
          "8": delta(current, metricAfter(index, 8)),
          "16": delta(current, metricAfter(index, 16)),
        },
      },
    };
  });
}

function blankAggregate(): Aggregate {
  return {
    episodes: 0,
    positions: 0,
    completed: 0,
    toppedOut: 0,
    score: 0,
    lines: 0,
    words: 0,
    searchNodes: 0,
  };
}

function addSummary(aggregate: Aggregate, summary: EpisodeSummary): void {
  aggregate.episodes += 1;
  aggregate.positions += summary.positions;
  aggregate.completed += Number(summary.phase === "complete");
  aggregate.toppedOut += Number(summary.phase === "over");
  aggregate.score += summary.score;
  aggregate.lines += summary.lines;
  aggregate.words += summary.words;
  aggregate.searchNodes += summary.searchNodes;
}

const options = parseOptions();
if (!Number.isFinite(options.hours) || options.hours <= 0) throw new Error("--hours must be positive.");
if (!Number.isFinite(options.shardRecords) || options.shardRecords < 100) {
  throw new Error("--shard-records must be at least 100.");
}

await mkdir(dirname(options.output), { recursive: true });
await mkdir(options.output, { recursive: false });
const startedAt = new Date();
const deadline = startedAt.getTime() + options.hours * 60 * 60 * 1000;
const assets = new Map<LexiconId, GameAssets>(await Promise.all(
  (["CSW24", "NWL23"] as LexiconId[]).map(async (lexicon) => [lexicon, await loadAssets(lexicon)] as const),
));
const aggregate = blankAggregate();
const byDepth = Object.fromEntries(options.depths.map((depth) => [depth, blankAggregate()])) as Record<string, Aggregate>;
const byLexicon = {
  CSW24: blankAggregate(),
  NWL23: blankAggregate(),
};
const shards: Array<{ file: string; records: number; bytes: number }> = [];
let shardIndex = 0;
let currentShardRecords = 0;
let stopRequested = false;
let lastStatusAt = Date.now();

process.on("SIGINT", () => { stopRequested = true; });
process.on("SIGTERM", () => { stopRequested = true; });

const manifest = (status: "running" | "complete") => ({
  schemaVersion: 1,
  status,
  pid: process.pid,
  startedAt: startedAt.toISOString(),
  deadline: new Date(deadline).toISOString(),
  updatedAt: new Date().toISOString(),
  completedAt: status === "complete" ? new Date().toISOString() : null,
  options,
  aggregate,
  byDepth,
  byLexicon,
  shards,
  currentShard: currentShardRecords > 0 ? {
    file: `positions-${String(shardIndex).padStart(5, "0")}.jsonl.gz`,
    records: currentShardRecords,
  } : null,
});

async function writeManifest(status: "running" | "complete"): Promise<void> {
  const temporary = resolve(options.output, "manifest.next.json");
  const destination = resolve(options.output, "manifest.json");
  await writeFile(temporary, `${JSON.stringify(manifest(status), null, 2)}\n`);
  await rename(temporary, destination);
}

async function appendEpisode(records: TrainingRecord[]): Promise<void> {
  if (records.length === 0) return;
  const file = `positions-${String(shardIndex).padStart(5, "0")}.jsonl.gz`;
  const contents = `${records.map((record) => JSON.stringify(record)).join("\n")}\n`;
  const compressed = gzipSync(Buffer.from(contents), { level: 6 });
  await appendFile(resolve(options.output, file), compressed);
  currentShardRecords += records.length;
}

async function finalizeShard(): Promise<void> {
  if (currentShardRecords === 0) return;
  const file = `positions-${String(shardIndex).padStart(5, "0")}.jsonl.gz`;
  const fileStats = await stat(resolve(options.output, file));
  shards.push({ file, records: currentShardRecords, bytes: fileStats.size });
  currentShardRecords = 0;
  shardIndex += 1;
}

await writeFile(resolve(options.output, "schema.json"), `${JSON.stringify({
  schemaVersion: 1,
  format: "gzip-compressed JSON Lines",
  boardEncoding: {
    letters: "22 strings of 10 characters; . denotes empty and A-Z denotes a tile",
    pieces: "22 strings of 10 characters; . denotes empty and I/J/L/O/S/T/Z denotes color group",
  },
  labels: {
    horizons: "Observed score, line, word, and word-length gain after 4, 8, and 16 placements",
    scoreToGo: "Observed undiscounted terminal score minus score at this position",
    scorePerLineToGo: "Observed score-to-go divided by observed remaining cleared lines",
  },
  caveat: "On-policy Monte Carlo labels generated by the recorded beam-search depth and width; shard files contain concatenated gzip members",
}, null, 2)}\n`);
await writeManifest("running");

for (let gameIndex = 0; gameIndex < options.maxGames; gameIndex += 1) {
  if (Date.now() >= deadline || stopRequested) break;
  const depth = options.depths[gameIndex % options.depths.length];
  const lexicon: LexiconId = gameIndex % 2 === 0 ? "CSW24" : "NWL23";
  const seed = gameSeed(options.seed, gameIndex);
  const width = beamWidth(depth);
  const game = new KvadratGame(assets.get(lexicon)!, seededRandom(seed));
  const episodeId = `${startedAt.toISOString()}-${String(gameIndex).padStart(7, "0")}`;
  const records: UnlabelledRecord[] = [];
  let searchNodes = 0;
  const episodeStartedAt = performance.now();

  while (records.length < 400) {
    const snapshot = game.getSnapshot();
    if (snapshot.phase === "clearing") {
      for (let tick = 0; tick < 10; tick += 1) game.tick(50, false);
      continue;
    }
    if (snapshot.phase !== "playing") break;
    const position = game.getTrainingPosition();
    const plan = game.findBestMove(depth, width);
    if (!position || !plan) break;
    records.push({
      schemaVersion: 1,
      episodeId,
      step: records.length,
      lexicon,
      seed,
      policy: { depth, beamWidth: width },
      position,
      action: plan,
    });
    searchNodes += plan.nodes;
    if (!game.executeBotPlan(plan)) break;
  }

  let finalSnapshot = game.getSnapshot();
  while (finalSnapshot.phase === "clearing") {
    for (let tick = 0; tick < 10; tick += 1) game.tick(50, false);
    finalSnapshot = game.getSnapshot();
  }
  const labelled = labelTrajectory(records, finalSnapshot);
  await appendEpisode(labelled);
  const summary: EpisodeSummary = {
    episodeId,
    lexicon,
    seed,
    depth,
    beamWidth: width,
    positions: labelled.length,
    score: finalSnapshot.score,
    lines: finalSnapshot.lines,
    words: finalSnapshot.words,
    averageWordLength: finalSnapshot.averageWordLength,
    phase: finalSnapshot.phase,
    searchNodes,
    elapsedMs: Math.round(performance.now() - episodeStartedAt),
  };
  await appendFile(resolve(options.output, "episodes.jsonl"), `${JSON.stringify(summary)}\n`);
  addSummary(aggregate, summary);
  addSummary(byDepth[String(depth)], summary);
  addSummary(byLexicon[lexicon], summary);

  if (currentShardRecords >= options.shardRecords) await finalizeShard();
  await writeManifest("running");

  if (aggregate.episodes % 10 === 0 || Date.now() - lastStatusAt >= 60_000) {
    const elapsedMinutes = (Date.now() - startedAt.getTime()) / 60_000;
    const meanScore = aggregate.score / aggregate.episodes;
    const rate = aggregate.positions / Math.max(elapsedMinutes, 1 / 60);
    console.log(JSON.stringify({
      status: "running",
      elapsedMinutes: Number(elapsedMinutes.toFixed(2)),
      episodes: aggregate.episodes,
      positions: aggregate.positions,
      shards: shards.length,
      meanScore: Math.round(meanScore),
      positionsPerMinute: Math.round(rate),
      output: options.output,
    }));
    lastStatusAt = Date.now();
  }
}

await finalizeShard();
await writeManifest("complete");
console.log(JSON.stringify({
  status: "complete",
  episodes: aggregate.episodes,
  positions: aggregate.positions,
  shards: shards.length,
  elapsedMinutes: Number(((Date.now() - startedAt.getTime()) / 60_000).toFixed(2)),
  output: options.output,
}));
