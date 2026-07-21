import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { KvadratGame } from "../app/game-engine.ts";

async function loadAssets() {
  const [kwgBuffer, bagsText] = await Promise.all([
    readFile(new URL("../public/data/CSW24.kwg", import.meta.url)),
    readFile(new URL("../public/data/csw24-bags.txt", import.meta.url), "utf8"),
  ]);
  const view = new DataView(kwgBuffer.buffer, kwgBuffer.byteOffset, kwgBuffer.byteLength);
  const kwg = new Uint32Array(kwgBuffer.byteLength / 4);
  for (let index = 0; index < kwg.length; index += 1) kwg[index] = view.getUint32(index * 4, true);
  const wordBags = bagsText.split(/\r?\n/).map((line) => line.trim().split(/\s+/).filter(Boolean)).filter((bag) => bag.length >= 28);
  return { kwg, wordBags };
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

test("validates CSW24-specific additions", async () => {
  const game = new KvadratGame(await loadAssets());
  assert.equal(game.isValidWord("UWU"), true);
  assert.equal(game.isValidWord("OWO"), true);
  assert.equal(game.isValidWord("FAV"), true);
  assert.equal(game.isValidWord("NERF"), true);
  assert.equal(game.isValidWord("RESPAWN"), true);
  assert.equal(game.isValidWord("ALF"), false);
});
