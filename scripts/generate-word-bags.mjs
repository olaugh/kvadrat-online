import { readFile, writeFile } from "node:fs/promises";

const letterCounts = {
  A: 11, B: 2, C: 3, D: 5, E: 14, F: 2, G: 3, H: 2, I: 10,
  J: 1, K: 1, L: 4, M: 2, N: 7, O: 9, P: 2, Q: 1, R: 7,
  S: 6, T: 7, U: 4, V: 2, W: 2, X: 1, Y: 2, Z: 2,
};
const vowels = new Set(["A", "E", "I", "O", "U"]);
const priorities = "QXZJKVBPYGFWMUCLDRHSNIOATE";

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

function shuffle(items, random) {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [items[index], items[swap]] = [items[swap], items[index]];
  }
}

function makeWordRecord(word) {
  const required = new Map();
  for (const letter of word) required.set(letter, (required.get(letter) ?? 0) + 1);
  return { word, required: [...required] };
}

function fits(record, remaining) {
  return record.required.every(([letter, count]) => remaining.get(letter) >= count);
}

function partitionQuadrant(letters, recordsByLetter, random) {
  const original = new Map();
  for (const letter of letters) original.set(letter, (original.get(letter) ?? 0) + 1);

  for (let attempt = 0; attempt < 120; attempt += 1) {
    const remaining = new Map(original);
    const chosen = [];
    while (chosen.length < 7) {
      const priority = [...priorities].find((letter) => remaining.get(letter) > 0);
      if (!priority) break;
      const candidates = recordsByLetter.get(priority).filter((record) => fits(record, remaining));
      if (!candidates.length) break;
      const selection = candidates[Math.floor(random() * candidates.length)];
      chosen.push(selection.word);
      for (const [letter, count] of selection.required) {
        remaining.set(letter, remaining.get(letter) - count);
      }
    }
    if (chosen.length === 7) return chosen;
  }
  return null;
}

function makeBag(recordsByLetter, random) {
  const vowelTiles = [];
  const consonantTiles = [];
  for (const [letter, count] of Object.entries(letterCounts)) {
    const destination = vowels.has(letter) ? vowelTiles : consonantTiles;
    for (let index = 0; index < count; index += 1) destination.push(letter);
  }
  shuffle(vowelTiles, random);
  shuffle(consonantTiles, random);

  const quadrants = Array.from({ length: 4 }, () => []);
  vowelTiles.forEach((letter, index) => quadrants[index % 4].push(letter));
  consonantTiles.forEach((letter, index) => quadrants[index % 4].push(letter));

  const words = [];
  for (const quadrant of quadrants) {
    const partition = partitionQuadrant(quadrant, recordsByLetter, random);
    if (!partition) return null;
    words.push(...partition);
  }
  return words;
}

const [inputPath, outputPath, requestedCount = "10000", seedText = "12628309"] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error("Usage: node scripts/generate-word-bags.mjs WORDS.txt OUTPUT.txt [COUNT] [SEED]");
}

const rawWords = await readFile(inputPath, "utf8");
const records = rawWords
  .split(/\r?\n/)
  .map((word) => word.trim().toUpperCase())
  .filter((word) => /^[A-Z]{4}$/.test(word))
  .map(makeWordRecord);
const recordsByLetter = new Map([...priorities].map((letter) => [letter, []]));
for (const record of records) {
  for (const [letter] of record.required) recordsByLetter.get(letter).push(record);
}

const random = seededRandom(Number(seedText));
const lines = [];
const count = Number(requestedCount);
let attempts = 0;
while (lines.length < count) {
  const bag = makeBag(recordsByLetter, random);
  attempts += 1;
  if (bag) lines.push(`${bag.join(" ")} `);
  if (attempts > count * 20) throw new Error(`Only generated ${lines.length} bags after ${attempts} attempts`);
}

await writeFile(outputPath, `${lines.join("\n")}\n`);
console.log(`Generated ${lines.length} bags from ${records.length} four-letter words in ${attempts} attempts.`);
