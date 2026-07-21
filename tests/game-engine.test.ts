import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { KvadratGame } from "../app/game-engine.ts";
import type { LexiconId } from "../app/game-engine.ts";

async function loadAssets(lexicon: LexiconId = "CSW24") {
  const bagName = lexicon.toLowerCase();
  const [kwgBuffer, bagsText] = await Promise.all([
    readFile(new URL(`../public/data/${lexicon}.kwg`, import.meta.url)),
    readFile(new URL(`../public/data/${bagName}-bags.txt`, import.meta.url), "utf8"),
  ]);
  const view = new DataView(kwgBuffer.buffer, kwgBuffer.byteOffset, kwgBuffer.byteLength);
  const kwg = new Uint32Array(kwgBuffer.byteLength / 4);
  for (let index = 0; index < kwg.length; index += 1) kwg[index] = view.getUint32(index * 4, true);
  const wordBags = bagsText.split(/\r?\n/).map((line) => line.trim().split(/\s+/).filter(Boolean)).filter((bag) => bag.length >= 28);
  return { kwg, wordBags };
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    return state / 4_294_967_296;
  };
}

test("creates and advances a playable 40-line game", async () => {
  const game = new KvadratGame(await loadAssets());
  const initial = game.getSnapshot();
  assert.equal(initial.board.length, 20);
  assert.ok(initial.board.every((row) => row.length === 10));
  assert.equal(initial.next.length, 4);
  assert.equal(initial.lines, 0);
  assert.equal(initial.phase, "playing");

  assert.equal(game.hardDrop(), true);
  const afterDrop = game.getSnapshot();
  assert.ok(afterDrop.pieces >= 2 || afterDrop.phase === "clearing");

  for (let index = 0; index < 30; index += 1) {
    if (!game.hardDrop()) break;
    game.tick(500, false);
  }
  const advanced = game.getSnapshot();
  assert.ok(["playing", "clearing", "over", "complete"].includes(advanced.phase));
  assert.ok(advanced.pieces > 1);
});

test("supports the flagship CSW24 and NWL23 English lexica", async () => {
  const [cswGame, nwlGame] = await Promise.all([
    loadAssets("CSW24").then((assets) => new KvadratGame(assets)),
    loadAssets("NWL23").then((assets) => new KvadratGame(assets)),
  ]);
  assert.equal(cswGame.isValidWord("UWU"), true);
  assert.equal(nwlGame.isValidWord("UWU"), false);
  assert.equal(cswGame.isValidWord("FAV"), true);
  assert.equal(nwlGame.isValidWord("FAV"), true);
  assert.equal(cswGame.isValidWord("NERF"), true);
  assert.equal(nwlGame.isValidWord("NERF"), false);
  assert.equal(cswGame.isValidWord("ALF"), false);
  assert.equal(nwlGame.isValidWord("ALF"), false);
});

test("searches future pieces and executes a legal scoring plan", async () => {
  const game = new KvadratGame(await loadAssets(), seededRandom(2));
  const sourceLetters = game.getTrainingPosition()!.active.letters;
  const plan = game.findBestMove(2, 24);
  assert.ok(plan);
  assert.equal(plan.depth, 2);
  assert.ok(plan.nodes > 30);
  assert.ok(plan.col >= -3 && plan.col < 10);
  assert.ok(plan.reason.length > 20);
  assert.equal(plan.sourceLetters.join(""), sourceLetters);
  assert.ok(plan.letterShift > 0, "the bot should prefer a non-default cyclic letter order here");
  assert.equal(plan.letters.join(""), sourceLetters.slice(plan.letterShift) + sourceLetters.slice(0, plan.letterShift));
  assert.equal(game.executeBotPlan(plan), true);

  const afterMove = game.getSnapshot();
  assert.ok(afterMove.pieces >= 2 || afterMove.phase === "clearing");
});

test("cycles active letters in both directions with four-step wraparound", async () => {
  const game = new KvadratGame(await loadAssets(), () => 0.25);
  const original = game.getTrainingPosition()!.active.letters;

  assert.equal(game.cycleLetters(-1), true);
  assert.equal(game.getTrainingPosition()!.active.letters, original.slice(1) + original[0]);
  assert.equal(game.cycleLetters(1), true);
  assert.equal(game.getTrainingPosition()!.active.letters, original);

  for (let index = 0; index < 4; index += 1) assert.equal(game.cycleLetters(-1), true);
  assert.equal(game.getTrainingPosition()!.active.letters, original);

  assert.equal(game.togglePause(), true);
  assert.equal(game.cycleLetters(-1), false);
  assert.equal(game.getTrainingPosition()!.active.letters, original);
});

test("exports reproducible model-training position features", async () => {
  const game = new KvadratGame(await loadAssets(), () => 0.25);
  const position = game.getTrainingPosition();
  assert.ok(position);
  assert.equal(position.board.letters.length, 22);
  assert.ok(position.board.letters.every((row) => row.length === 10));
  assert.ok(position.board.pieces.every((row) => row.length === 10));
  assert.equal(position.active.letters.length, 4);
  assert.equal(position.next.length, 4);
  assert.deepEqual(position.features.heights, Array(10).fill(0));
  assert.equal(position.features.wordPotential, 0);
});
