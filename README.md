# Kvadrat Online

Kvadrat is a falling-block word game. Every tetromino carries four letters;
build valid words across pieces, complete lines, and maximize your score over a
40-line run.

This repository is a fresh web port of the original Raylib prototype. The first
playable includes:

- the original 10 × 22 field with 20 visible rows
- seven-bag tetrominos and four-letter CSW24 bags
- CSW24 word validation and original word scoring
- a ghost piece, lock delay, wall kicks, and a 40-line finish
- keyboard and standard Gamepad API controls
- four-piece, CSW24-aware beam-search hints and autoplay
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

## Game data

The web build uses a CSW24 KWG and reproducible four-letter bags generated from
the complete CSW24 list with the original prototype's tile-distribution
algorithm. Regenerate the bags with:

```bash
node scripts/generate-word-bags.mjs /path/to/CSW24.txt public/data/csw24-bags.txt
```

Future versions will make dictionaries and language packs modular ruleset
assets.
