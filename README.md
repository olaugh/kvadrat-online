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
- four-piece, lexicon-aware beam-search hints and autoplay
- responsive, high-resolution, procedurally rendered UI
- device-local high scores

The strategy engine simulates every direct-drop rotation and column for the
current piece, then retains the strongest continuations across the three visible
next pieces. It scores banked words exactly and balances live word material
against holes, height, surface roughness, wells, and the limited 40-line budget.
Use **Hint** to inspect its recommendation or **Watch bot** to let it play.

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

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Run `npm test` for a production build, server-render test, and engine smoke
test.

## Self-play data

Generate checkpointed, gzip-compressed JSONL trajectories for training a leaf
position evaluator with:

```bash
npm run self-play -- --hours 8 --depths 2,3,4
```

The generator alternates CSW24 and NWL23, uses reproducible per-game seeds,
and records raw boards, visible pieces, structural heuristics, chosen moves,
4/8/16-placement returns, and terminal score-to-go labels. Generated corpora
are written under `training-data/` and intentionally excluded from Git.

After a run completes, validate every compressed shard and create a JSON plus
Markdown quality report with:

```bash
npm run analyze-self-play -- --input training-data/your-run
```

## Game data

The web build ships CSW24 for World English and NWL23 for North American
English. Each ruleset has its own KWG, reproducible four-letter bags, bot
search, and device-local high score. The bags use the original prototype's
tile-distribution algorithm. Regenerate either set with:

```bash
node scripts/generate-word-bags.mjs /path/to/CSW24.txt public/data/csw24-bags.txt
node scripts/generate-word-bags.mjs /path/to/NWL23.txt public/data/nwl23-bags.txt
```
