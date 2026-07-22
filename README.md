# Kvadrat Online

Kvadrat is a falling-block word game. Every tetromino carries four letters;
build valid words across pieces, complete lines, and maximize your score over a
40-line run.

This repository is a fresh web port of the original Raylib prototype. The first
playable includes:

- the original 10 × 22 field with 20 visible rows
- seven-bag tetrominos and lexicon-specific four-letter bags
- flagship CSW24 and NWL23 English word validation
- a ghost piece, lock delay, wall kicks, and a 40-line finish
- keyboard and standard Gamepad API controls
- four-ply, lexicon-aware Rust/WASM beam-search hints and autoplay
- responsive, high-resolution, procedurally rendered UI
- device-local high scores

The strategy engine is a dependency-free Rust module compiled to WebAssembly.
One call simulates every cyclic letter order, direct-drop rotation, and column,
then retains the strongest continuations across the three visible next pieces.
DAWG traversal, row scoring, placement generation, board evaluation, and beam
expansion all stay inside WASM. It scores banked words exactly and balances live
word material against holes, height, surface roughness, wells, and the limited
40-line budget. Use **Hint** to inspect its recommendation or **Watch bot** to
let it play.

The crate also contains an experimental learned leaf evaluator. Its native
Rust tensorizer, trainer, evaluator, compact model exporter, dependency-free
inference implementation, and WASM ABI are complete. The first model is not
enabled in autoplay: a seed-disjoint 400-game paired run did not beat the
heuristic, so the production bot deliberately retains the baseline policy.

## Controls

| Action | Keyboard | Controller |
| --- | --- | --- |
| Move | Left / Right | D-pad or left stick |
| Soft drop | Down | D-pad or left stick down |
| Hard drop | Space | D-pad up |
| Rotate clockwise | X or Up | A / Cross |
| Rotate counterclockwise | Z | B / Circle |
| Cycle letters left | C | X / Square |
| Cycle letters right | V | Y / Triangle |
| Pause | P or Escape | Start |
| Restart | R | Select / Back |

## Development

Requires Node.js 22.13 or newer. Normal web builds use the checked-in WASM
artifact and do not require Rust.

```bash
npm install
npm run dev
```

Run `npm test` for a production build, server-render test, and engine smoke
test. The test suite also compares Rust/WASM plans and scoring against the
TypeScript reference across both English lexica and mixed search depths.

GitHub Actions runs the full web and Rust presubmit on every pull request and
push to `main`. Coverage is enforced at 80% for lines, functions, and branches
across the TypeScript game core (`game-engine.ts` and `strategy-wasm.ts`), and
at 80% for lines and functions across the shipping Rust strategy library.
Offline corpus/training binaries still run under tests and Clippy, but are not
included in the product-engine coverage denominator.

Run the web coverage gate locally with:

```bash
npm run test:coverage
```

For native coverage, install the pinned tool and Rust instrumentation component
once, then run the complete presubmit:

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --version 0.8.6 --locked
npm run presubmit
```

To rebuild `public/wasm/kvadrat-strategy.wasm`, install stable Rust with the
`wasm32-unknown-unknown` target and run:

```bash
npm run build:wasm
```

## Self-play data

Generate checkpointed, gzip-compressed JSONL trajectories with the native Rust
self-play runner. Stable Rust is required for this command:

```bash
npm run self-play -- --hours 8 --depths 2,3,3,3
```

The generator runs one fully native game per worker and defaults to all but one
available CPU thread. Override that with `--threads N`; repeat a depth in
`--depths` to weight the policy mix. It alternates CSW24 and NWL23, uses
reproducible per-game seeds, preserves game-index output order across workers,
and records raw boards, visible pieces, structural heuristics, chosen moves,
4/8/16-placement returns, and terminal score-to-go labels. Generated corpora
are written under `training-data/` and intentionally excluded from Git.

After a run completes, validate every compressed shard and create a JSON plus
Markdown quality report with:

```bash
npm run analyze-self-play -- --input training-data/your-run
```

## Fragment model experiments

Prepare leakage-safe, memory-mapped train/validation/test tensors from a
completed corpus with native Rust:

```bash
npm run prepare-fragments -- prepare \
  --input training-data/your-run \
  --output training-data/fragments
```

Episode seeds are avalanche-hashed before the 80/10/10 split so CSW24 and
NWL23 are represented in every partition. Search depth is retained only for
audits; it is excluded from deployed model inputs. Train the full lexical model
and its masked-letter context control on CPU:

```bash
npm run train-fragments -- \
  --data training-data/fragments \
  --output training-data/fragment-full \
  --device cpu --input-mode full --epochs 3

npm run train-fragments -- \
  --data training-data/fragments \
  --output training-data/fragment-context \
  --device cpu --input-mode mask-word-inputs --epochs 3
```

The shared row encoder sees letters, gaps, column geometry, the next five
pieces, line budget, lexicon, and three boundary states: empty edge, adjacent
cells of the same tetromino color, or different colors. It predicts score and
word-quality returns without enumerating possible completions. Export either
Candle safetensors checkpoint to the fixed-size dependency-free inference
format with:

```bash
npm run export-fragment -- \
  --input training-data/fragment-full/model.safetensors \
  --output public/data/models/fragment-full.kfm
```

For a paired policy test, run baseline and candidate games with identical
seeds, then compare them:

```bash
npm run self-play -- --games 400 --hours 1 --depths 3 --seed 3512640997 \
  --output training-data/eval-baseline

npm run self-play -- --games 400 --hours 1 --depths 3 --seed 3512640997 \
  --fragment-full public/data/models/fragment-full-v4.kfm \
  --fragment-context public/data/models/fragment-context-v4.kfm \
  --fragment-weight 0.25 --fragment-candidates 6 \
  --output training-data/eval-candidate

npm run compare-self-play -- \
  --baseline training-data/eval-baseline \
  --candidate training-data/eval-candidate
```

Model hashes, held-out prediction metrics, rerank settings, and the rejected
paired result are recorded in `public/data/models/MANIFEST.json`. Training data
and checkpoints remain intentionally excluded from Git.

## Counterfactual root-action experiments

The second training path compares alternative root placements from the same
position instead of fitting only the move chosen by self-play. It takes unique
root actions from a depth-3, beam-160 frontier, preserves the original
depth-3/beam-64 action as candidate zero, and rolls each action forward for
eight placements with the baseline policy and identical future pieces. This
produces listwise labels for immediate score plus rollout score, with a top-out
penalty:

```bash
npm run counterfactuals -- \
  --input training-data/your-run \
  --output training-data/counterfactuals \
  --positions 10000 --candidates 12 \
  --search-depth 3 --candidate-beam-width 160 \
  --rollout-depth 3 --rollout-beam-width 64 --horizon 8

npm run train-counterfactuals -- \
  --data training-data/counterfactuals/counterfactuals.kvcf \
  --output training-data/root-ranker --device cpu --epochs 10

npm run export-ranker -- \
  --input training-data/root-ranker/model.safetensors \
  --output training-data/root-ranker.kvr
```

The 10,000-position pilot found substantial candidate-set headroom: an oracle
over the top 12 improved the eight-move objective by 179.6 points on average.
The first listwise ranker recovered only 7.2% of that headroom on held-out
positions. In full games, widening root search without the model improved score
by 425.6 points (+4.07%, 95% CI +324.8 to +526.4) over 400 paired games. The
ranker at its tuned 0.5 weight added 16.8 points in a separate 800-game run, but
the 95% CI was -61.8 to +95.3 and it caused three additional top-outs. It is
therefore retained as an experiment, not enabled in the web bot. The verified
next offline policy baseline keeps the depth-3/beam-64 action, adds up to 12
unique roots from a beam-160 frontier, and chooses among their best leaves with
the existing heuristic:

```bash
npm run self-play -- --depths 3 \
  --root-candidate-beam 160 --root-candidates 12
```

## Game data

The web build ships CSW24 for World English and NWL23 for North American
English. Each ruleset has its own compact DAWG-only KWG, reproducible
four-letter bags, bot search, and device-local high score. The GADDAG portion
is omitted because Kvadrat only performs forward membership queries. The
checked assets contain the same 299,162 CSW24 and 212,868 NWL23 words as their
source KWGs; hashes and counts live in `public/data/DAWG_MANIFEST.json`.

Extract and exhaustively verify a DAWG-only file from a combined Wolges KWG
with:

```bash
node scripts/extract-dawg.mjs /path/to/combined.kwg /path/to/compact.kwg
```

The bags use the original prototype's tile-distribution algorithm. Regenerate
either set with:

```bash
node scripts/generate-word-bags.mjs /path/to/CSW24.txt public/data/csw24-bags.txt
node scripts/generate-word-bags.mjs /path/to/NWL23.txt public/data/nwl23-bags.txt
```
