# Kvadrat strategy kernel

This Rust crate owns the performance-sensitive strategy path:

- DAWG membership checks
- row segmentation and scoring
- cyclic-letter placement generation
- line clearing and board collapse
- structural and word-potential evaluation
- bounded beam expansion through five plies

The browser keeps authoritative game state, input, rendering, and bot-plan
presentation in TypeScript. It sends one compact board plus the known piece
sequence to WASM per requested move. The result contains only the selected root
placement and projected metrics.

For offline work, `kvadrat-self-play` links the same core as a native Rust
library. It runs independent deterministic games across CPU workers without a
Node or WASM search layer and writes the existing gzip JSONL training schema.
Native-only serialization and compression dependencies are excluded from the
WASM target.

## ABI

`app/strategy-wasm.ts` is the ABI owner. Version 1 uses packed cells: bits 0–4
store A–Z as 1–26 and bits 5–7 store the tetromino/color group as 1–7.

The module exports:

- `kv_alloc` / `kv_dealloc`
- `kv_is_word`
- `kv_analyze_row`
- `kv_find_best_move`

Lexicon bytes are copied into WASM once. Scratch input and output buffers are
allocated once per strategy instance and reused. Search does not call back into
JavaScript.

## Build

Install stable Rust with `wasm32-unknown-unknown`, then run:

```bash
npm run build:wasm
```

The build copies the optimized module to `public/wasm/kvadrat-strategy.wasm`
and writes its reproducibility manifest. Normal application builds consume the
checked artifact and do not invoke Cargo.

Run native multithreaded self-play through the repository command:

```bash
npm run self-play -- --hours 8 --depths 2,3,3,3 --threads 8
```

On the 18-game mixed depth-2/3 scheduling check used for this port, one worker
took 25.2 seconds and 17 workers took 3.85 seconds. Both runs emitted the same
1,906 records in the same game-index order after timestamp normalization.

## Reference and parity

The original TypeScript implementation remains as a readable fallback and
parity oracle. Tests compare complete plans and resulting boards across CSW24,
NWL23, and 2/3/4-ply searches.

On the matched CSW24 depth-4 benchmark used during the port, both engines chose
the same 108 moves, searched 3,082,808 nodes, and completed with 13,212 points.
Total search time was 22.5 seconds in TypeScript and 3.6 seconds in WASM, a
6.26× speedup.
