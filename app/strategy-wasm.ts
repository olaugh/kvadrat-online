export type StrategyWord = {
  start: number;
  end: number;
  text: string;
  score: number;
};

export type StrategyPiece = {
  piece: number;
  letters: string;
};

export type StrategySearchResult = {
  letterShift: number;
  rotation: number;
  row: number;
  col: number;
  immediateScore: number;
  immediateLines: number;
  immediateWords: string[];
  projectedScore: number;
  projectedLines: number;
  setupWords: string[];
  depth: number;
  nodes: number;
  evaluation: number;
};

type StrategyExports = {
  memory: WebAssembly.Memory;
  kv_alloc(size: number): number;
  kv_dealloc(pointer: number, size: number): void;
  kv_is_word(lexiconPointer: number, lexiconNodes: number, wordPointer: number, wordLength: number): number;
  kv_analyze_row(
    lexiconPointer: number,
    lexiconNodes: number,
    rowPointer: number,
    outputPointer: number,
    outputCapacity: number,
  ): number;
  kv_find_best_move(
    lexiconPointer: number,
    lexiconNodes: number,
    inputPointer: number,
    inputLength: number,
    outputPointer: number,
    outputCapacity: number,
  ): number;
};

const BOARD_CELLS = 22 * 10;
const WORD_CAPACITY = 32;
const INPUT_CAPACITY = 256;
const OUTPUT_CAPACITY = 1024;
let browserModule: Promise<WebAssembly.Module> | null = null;

function letterTiles(text: string): Uint8Array | null {
  const upper = text.trim().toUpperCase();
  const tiles = new Uint8Array(upper.length);
  for (let index = 0; index < upper.length; index += 1) {
    const tile = upper.charCodeAt(index) - 64;
    if (tile < 1 || tile > 26) return null;
    tiles[index] = tile;
  }
  return tiles;
}

function tileText(bytes: Uint8Array): string {
  return String.fromCharCode(...bytes.map((tile) => tile + 64));
}

export class WasmStrategy {
  readonly kind = "wasm";
  private readonly exports: StrategyExports;
  private readonly lexiconPointer: number;
  private readonly lexiconNodes: number;
  private readonly wordPointer: number;
  private readonly rowPointer: number;
  private readonly inputPointer: number;
  private readonly outputPointer: number;
  private disposed = false;

  constructor(instance: WebAssembly.Instance, lexiconBytes: ArrayBuffer) {
    this.exports = instance.exports as unknown as StrategyExports;
    if (!(this.exports.memory instanceof WebAssembly.Memory)) {
      throw new Error("Kvadrat strategy WASM did not export linear memory.");
    }
    if (lexiconBytes.byteLength % 4 !== 0) throw new Error("DAWG byte length must be divisible by four.");
    this.lexiconNodes = lexiconBytes.byteLength / 4;
    this.lexiconPointer = this.allocate(lexiconBytes.byteLength);
    this.wordPointer = this.allocate(WORD_CAPACITY);
    this.rowPointer = this.allocate(10);
    this.inputPointer = this.allocate(INPUT_CAPACITY);
    this.outputPointer = this.allocate(OUTPUT_CAPACITY);
    this.write(this.lexiconPointer, new Uint8Array(lexiconBytes));
  }

  private allocate(size: number): number {
    const pointer = this.exports.kv_alloc(size) >>> 0;
    if (!pointer) throw new Error(`Kvadrat strategy WASM could not allocate ${size} bytes.`);
    return pointer;
  }

  private memory(pointer: number, length: number): Uint8Array {
    return new Uint8Array(this.exports.memory.buffer, pointer, length);
  }

  private write(pointer: number, bytes: Uint8Array): void {
    this.memory(pointer, bytes.length).set(bytes);
  }

  isWord(text: string): boolean {
    const tiles = letterTiles(text);
    if (!tiles || tiles.length < 2 || tiles.length > WORD_CAPACITY) return false;
    this.write(this.wordPointer, tiles);
    return Boolean(this.exports.kv_is_word(
      this.lexiconPointer,
      this.lexiconNodes,
      this.wordPointer,
      tiles.length,
    ));
  }

  analyzeRow(row: Uint8Array): StrategyWord[] {
    if (row.length !== 10) throw new Error("Strategy rows must contain exactly ten cells.");
    this.write(this.rowPointer, row);
    const length = this.exports.kv_analyze_row(
      this.lexiconPointer,
      this.lexiconNodes,
      this.rowPointer,
      this.outputPointer,
      OUTPUT_CAPACITY,
    );
    if (!length) return [];
    const bytes = this.memory(this.outputPointer, length);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const count = bytes[0];
    const words: StrategyWord[] = [];
    let position = 1;
    for (let index = 0; index < count; index += 1) {
      const start = bytes[position++];
      const end = bytes[position++];
      const score = view.getInt32(position, true);
      position += 4;
      const textLength = bytes[position++];
      const text = tileText(bytes.slice(position, position + textLength));
      position += textLength;
      words.push({ start, end, text, score });
    }
    return words;
  }

  findBestMove(
    board: Uint8Array,
    currentLines: number,
    sequence: StrategyPiece[],
    depth: number,
    beamWidth: number,
  ): StrategySearchResult | null {
    if (board.length !== BOARD_CELLS) throw new Error(`Strategy boards must contain ${BOARD_CELLS} cells.`);
    const searchDepth = Math.max(1, Math.min(5, Math.floor(depth)));
    const width = Math.max(12, Math.min(160, Math.floor(beamWidth)));
    if (sequence.length < searchDepth) throw new Error("Strategy search sequence is shorter than its depth.");

    const inputLength = 226 + searchDepth * 5;
    const input = new Uint8Array(inputLength);
    input[0] = 1;
    input[1] = searchDepth;
    input[2] = width;
    input[3] = Math.max(0, Math.min(255, Math.floor(currentLines)));
    input[4] = searchDepth;
    input.set(board, 6);
    for (let index = 0; index < searchDepth; index += 1) {
      const offset = 226 + index * 5;
      const piece = sequence[index];
      const letters = letterTiles(piece.letters);
      if (!letters || letters.length !== 4 || piece.piece < 0 || piece.piece > 6) {
        throw new Error("Strategy pieces require a valid tetromino index and four A–Z letters.");
      }
      input[offset] = piece.piece;
      input.set(letters, offset + 1);
    }
    this.write(this.inputPointer, input);

    const length = this.exports.kv_find_best_move(
      this.lexiconPointer,
      this.lexiconNodes,
      this.inputPointer,
      inputLength,
      this.outputPointer,
      OUTPUT_CAPACITY,
    );
    if (!length) return null;
    const bytes = this.memory(this.outputPointer, length);
    if (bytes[0] !== 1) throw new Error(`Unknown strategy result version ${bytes[0]}.`);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const immediateWordCount = bytes[7];
    const setupWordCount = bytes[8];
    let position = 28;
    const readTexts = (count: number) => Array.from({ length: count }, () => {
      const textLength = bytes[position++];
      const text = tileText(bytes.slice(position, position + textLength));
      position += textLength;
      return text;
    });
    const immediateWords = readTexts(immediateWordCount);
    const setupWords = readTexts(setupWordCount);
    return {
      letterShift: bytes[1],
      rotation: bytes[2],
      row: view.getInt8(3),
      col: view.getInt8(4),
      immediateScore: view.getInt32(10, true),
      immediateLines: bytes[5],
      immediateWords,
      projectedScore: view.getInt32(14, true),
      projectedLines: view.getUint16(18, true),
      setupWords,
      depth: bytes[6],
      nodes: view.getUint32(20, true),
      evaluation: view.getInt32(24, true),
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.exports.kv_dealloc(this.lexiconPointer, this.lexiconNodes * 4);
    this.exports.kv_dealloc(this.wordPointer, WORD_CAPACITY);
    this.exports.kv_dealloc(this.rowPointer, 10);
    this.exports.kv_dealloc(this.inputPointer, INPUT_CAPACITY);
    this.exports.kv_dealloc(this.outputPointer, OUTPUT_CAPACITY);
  }
}

export async function instantiateWasmStrategy(
  wasmBytes: BufferSource,
  lexiconBytes: ArrayBuffer,
): Promise<WasmStrategy> {
  const source = await WebAssembly.instantiate(wasmBytes, {});
  return new WasmStrategy(source.instance, lexiconBytes);
}

export async function loadWasmStrategy(lexiconBytes: ArrayBuffer): Promise<WasmStrategy> {
  browserModule ??= fetch("/wasm/kvadrat-strategy.wasm").then(async (response) => {
    if (!response.ok) throw new Error("Could not load the Kvadrat strategy engine.");
    return WebAssembly.compile(await response.arrayBuffer());
  });
  const instance = await WebAssembly.instantiate(await browserModule, {});
  return new WasmStrategy(instance, lexiconBytes);
}
