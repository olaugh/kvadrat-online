# Kvadrat Online

Kvadrat is a falling-block word game. Every tetromino carries four letters;
build valid words across pieces, complete lines, and maximize your score over a
40-line run.

This repository is a fresh web port of the original Raylib prototype. The first
playable includes:

- the original 10 × 22 field with 20 visible rows
- seven-bag tetrominos and four-letter CSW21 bags
- CSW21 word validation and original word scoring
- a ghost piece, lock delay, wall kicks, and a 40-line finish
- keyboard and standard Gamepad API controls
- responsive, high-resolution, procedurally rendered UI
- device-local high scores

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

The initial web build carries forward the original prototype's CSW21 KWG and
four-letter bag data. Future versions will make dictionaries and language packs
modular ruleset assets.
