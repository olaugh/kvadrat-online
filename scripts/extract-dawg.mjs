import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const ARC_MASK = 0x3fffff;
const IS_END = 0x400000;
const ACCEPTS = 0x800000;

function usage() {
  throw new Error("Usage: node scripts/extract-dawg.mjs input.kwg output.kwg");
}

const inputPath = process.argv[2] ? resolve(process.argv[2]) : usage();
const outputPath = process.argv[3] ? resolve(process.argv[3]) : usage();
const input = await readFile(inputPath);
if (input.byteLength % 4 !== 0 || input.byteLength < 8) {
  throw new Error(`${inputPath} is not a valid 32-bit KWG.`);
}

const view = new DataView(input.buffer, input.byteOffset, input.byteLength);
const nodes = new Uint32Array(input.byteLength / 4);
for (let index = 0; index < nodes.length; index += 1) {
  nodes[index] = view.getUint32(index * 4, true);
}

const root = nodes[0] & ARC_MASK;
if (root <= 1 || root >= nodes.length) throw new Error("KWG slot 0 has no valid DAWG root.");

const reachable = new Set();
const visitedStarts = new Set();
const pending = [root];
while (pending.length > 0) {
  const start = pending.pop();
  if (!start || visitedStarts.has(start)) continue;
  visitedStarts.add(start);
  for (let index = start; index < nodes.length; index += 1) {
    reachable.add(index);
    const child = nodes[index] & ARC_MASK;
    if (child && !visitedStarts.has(child)) pending.push(child);
    if (nodes[index] & IS_END) break;
  }
}

const ordered = [...reachable].sort((left, right) => left - right);
const remap = new Map([[0, 0], [1, 1]]);
for (let index = 0; index < ordered.length; index += 1) remap.set(ordered[index], index + 2);

const compact = new Uint32Array(ordered.length + 2);
compact[0] = IS_END | remap.get(root);
compact[1] = IS_END;
for (let index = 0; index < ordered.length; index += 1) {
  const node = nodes[ordered[index]];
  const child = node & ARC_MASK;
  const compactChild = child === 0 ? 0 : remap.get(child);
  if (compactChild === undefined) throw new Error(`Reachable node points outside the DAWG: ${child}.`);
  compact[index + 2] = (node & ~ARC_MASK) | compactChild;
}

function accepts(graph, word) {
  let nodeIndex = graph[0] & ARC_MASK;
  let accepted = false;
  for (const tile of word) {
    let found = false;
    for (let index = nodeIndex; index < graph.length; index += 1) {
      const node = graph[index];
      if ((node >>> 24) === tile) {
        found = true;
        accepted = Boolean(node & ACCEPTS);
        nodeIndex = node & ARC_MASK;
        break;
      }
      if (node & IS_END) break;
    }
    if (!found) return false;
  }
  return accepted;
}

function fingerprint(graph) {
  const hash = createHash("sha256");
  const word = [];
  let count = 0;
  let maximumLength = 0;

  function visit(start) {
    for (let index = start; index < graph.length; index += 1) {
      const node = graph[index];
      word.push(node >>> 24);
      if (node & ACCEPTS) {
        if (!accepts(graph, word)) throw new Error("DAWG traversal produced an unaccepted word.");
        hash.update(Uint8Array.from(word));
        hash.update(Uint8Array.of(0));
        count += 1;
        maximumLength = Math.max(maximumLength, word.length);
      }
      const child = node & ARC_MASK;
      if (child) visit(child);
      word.pop();
      if (node & IS_END) break;
    }
  }

  visit(graph[0] & ARC_MASK);
  return { count, maximumLength, sha256: hash.digest("hex") };
}

const originalFingerprint = fingerprint(nodes);
const compactFingerprint = fingerprint(compact);
if (JSON.stringify(originalFingerprint) !== JSON.stringify(compactFingerprint)) {
  throw new Error(`DAWG mismatch: ${JSON.stringify({ originalFingerprint, compactFingerprint })}`);
}

const output = Buffer.allocUnsafe(compact.byteLength);
const outputView = new DataView(output.buffer, output.byteOffset, output.byteLength);
for (let index = 0; index < compact.length; index += 1) {
  outputView.setUint32(index * 4, compact[index], true);
}
await writeFile(outputPath, output);

console.log(JSON.stringify({
  input: inputPath,
  output: outputPath,
  inputNodes: nodes.length,
  outputNodes: compact.length,
  inputBytes: input.byteLength,
  outputBytes: output.byteLength,
  reductionPercent: Number((100 * (1 - output.byteLength / input.byteLength)).toFixed(2)),
  words: compactFingerprint.count,
  maximumWordLength: compactFingerprint.maximumLength,
  wordSetSha256: compactFingerprint.sha256,
}));
