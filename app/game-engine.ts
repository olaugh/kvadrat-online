export const BOARD_WIDTH = 10;
export const BOARD_HEIGHT = 22;
export const VISIBLE_HEIGHT = 20;
export const MAX_LINES = 40;

export const LEXICA = {
  CSW24: { name: "CSW24", region: "World English" },
  NWL23: { name: "NWL23", region: "North American English" },
} as const;
export type LexiconId = keyof typeof LEXICA;

export const PIECE_NAMES = ["I", "J", "L", "O", "S", "T", "Z"] as const;
export type PieceName = (typeof PIECE_NAMES)[number];
export type GamePhase = "playing" | "clearing" | "paused" | "over" | "complete";

export const LETTER_VALUES: Record<string, number> = {
  A: 1, B: 3, C: 3, D: 2, E: 1, F: 4, G: 2, H: 4, I: 1,
  J: 8, K: 5, L: 1, M: 3, N: 1, O: 1, P: 3, Q: 10, R: 1,
  S: 1, T: 1, U: 1, V: 4, W: 4, X: 8, Y: 4, Z: 10,
};

type StoredCell = { letter: string; piece: PieceName };
type Board = Array<Array<StoredCell | null>>;
type Block = { x: number; y: number; letterIndex: number };

type ActivePiece = {
  piece: PieceName;
  letters: string[];
  rotation: number;
  row: number;
  col: number;
};

type WordSegment = {
  start: number;
  end: number;
  text: string;
  score: number;
};

export type RecentWord = {
  id: number;
  text: string;
  score: number;
};

export type RenderCell = {
  letter: string;
  value: number;
  piece: PieceName;
  active?: boolean;
  ghost?: boolean;
  clearing?: boolean;
  wordScore?: number;
};

export type PreviewPiece = {
  piece: PieceName;
  cells: Array<{ x: number; y: number; letter: string }>;
};

export type GameSnapshot = {
  board: Array<Array<RenderCell | null>>;
  next: PreviewPiece[];
  score: number;
  lines: number;
  pieces: number;
  words: number;
  averageWordLength: number;
  elapsedMs: number;
  phase: GamePhase;
  recentWords: RecentWord[];
};

export type BotPlan = {
  piece: PieceName;
  letters: string[];
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
  reason: string;
};

export type GameAssets = {
  kwg: Uint32Array;
  wordBags: string[][];
};

const BASE_BLOCKS: Record<PieceName, { size: number; blocks: Block[] }> = {
  I: { size: 4, blocks: [
    { x: 0, y: 1, letterIndex: 0 }, { x: 1, y: 1, letterIndex: 1 },
    { x: 2, y: 1, letterIndex: 2 }, { x: 3, y: 1, letterIndex: 3 },
  ] },
  J: { size: 3, blocks: [
    { x: 0, y: 0, letterIndex: 0 }, { x: 0, y: 1, letterIndex: 1 },
    { x: 1, y: 1, letterIndex: 2 }, { x: 2, y: 1, letterIndex: 3 },
  ] },
  L: { size: 3, blocks: [
    { x: 2, y: 0, letterIndex: 0 }, { x: 0, y: 1, letterIndex: 1 },
    { x: 1, y: 1, letterIndex: 2 }, { x: 2, y: 1, letterIndex: 3 },
  ] },
  O: { size: 4, blocks: [
    { x: 1, y: 1, letterIndex: 0 }, { x: 2, y: 1, letterIndex: 1 },
    { x: 1, y: 2, letterIndex: 2 }, { x: 2, y: 2, letterIndex: 3 },
  ] },
  S: { size: 3, blocks: [
    { x: 1, y: 0, letterIndex: 0 }, { x: 2, y: 0, letterIndex: 1 },
    { x: 0, y: 1, letterIndex: 2 }, { x: 1, y: 1, letterIndex: 3 },
  ] },
  T: { size: 3, blocks: [
    { x: 1, y: 0, letterIndex: 0 }, { x: 0, y: 1, letterIndex: 1 },
    { x: 1, y: 1, letterIndex: 2 }, { x: 2, y: 1, letterIndex: 3 },
  ] },
  Z: { size: 3, blocks: [
    { x: 0, y: 0, letterIndex: 0 }, { x: 1, y: 0, letterIndex: 1 },
    { x: 1, y: 1, letterIndex: 2 }, { x: 2, y: 1, letterIndex: 3 },
  ] },
};

function rotateBlocks(blocks: Block[], size: number): Block[] {
  return blocks.map(({ x, y, letterIndex }) => ({
    x: size - 1 - y,
    y: x,
    letterIndex,
  }));
}

const ROTATIONS = Object.fromEntries(
  PIECE_NAMES.map((piece) => {
    const { size, blocks } = BASE_BLOCKS[piece];
    const rotations: Block[][] = [blocks];
    for (let index = 1; index < 4; index += 1) {
      rotations.push(rotateBlocks(rotations[index - 1], size));
    }
    return [piece, rotations];
  }),
) as Record<PieceName, Block[][]>;

// The original uses SRS kick tables. This compact set keeps the same useful
// wall/floor behavior while remaining forgiving for a first browser pass.
const KICKS: Array<[number, number]> = [
  [0, 0], [-1, 0], [1, 0], [-2, 0], [2, 0], [0, -1],
  [-1, -1], [1, -1], [0, -2],
];

const KWG_IS_END = 0x400000;
const KWG_ACCEPTS = 0x800000;
const KWG_ARC_MASK = 0x3fffff;
const MINIMUM_WORD_SCORE = 40;
const DEFAULT_GRAVITY_MS = 360;

type SearchPiece = {
  piece: PieceName;
  letters: string[];
};

type SimulatedPlacement = {
  board: Board;
  rotation: number;
  row: number;
  col: number;
  score: number;
  lines: number;
  words: WordSegment[];
};

type BoardEvaluation = {
  value: number;
  setupWords: string[];
};

type SearchState = {
  board: Board;
  score: number;
  lines: number;
  root: SimulatedPlacement;
  value: number;
  setupWords: string[];
};

function emptyBoard(): Board {
  return Array.from({ length: BOARD_HEIGHT }, () =>
    Array<StoredCell | null>(BOARD_WIDTH).fill(null),
  );
}

function shuffledPieces(): PieceName[] {
  const bag = [...PIECE_NAMES];
  for (let index = bag.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [bag[index], bag[swapIndex]] = [bag[swapIndex], bag[index]];
  }
  return bag;
}

function scoreWord(text: string): number {
  const rawScore = [...text].reduce((sum, letter) => sum + LETTER_VALUES[letter], 0);
  return rawScore * text.length * text.length;
}

export async function loadGameAssets(lexicon: LexiconId = "CSW24"): Promise<GameAssets> {
  const bagName = lexicon.toLowerCase();
  const [kwgResponse, bagsResponse] = await Promise.all([
    fetch(`/data/${lexicon}.kwg`),
    fetch(`/data/${bagName}-bags.txt`),
  ]);

  if (!kwgResponse.ok || !bagsResponse.ok) {
    throw new Error(`Could not load the ${lexicon} game data.`);
  }

  const [kwgBuffer, bagsText] = await Promise.all([
    kwgResponse.arrayBuffer(),
    bagsResponse.text(),
  ]);

  const view = new DataView(kwgBuffer);
  const kwg = new Uint32Array(kwgBuffer.byteLength / 4);
  for (let index = 0; index < kwg.length; index += 1) {
    kwg[index] = view.getUint32(index * 4, true);
  }

  const wordBags = bagsText
    .split(/\r?\n/)
    .map((line) => line.trim().split(/\s+/).filter(Boolean))
    .filter((bag) => bag.length >= 28)
    .map((bag) => bag.slice(0, 28));

  return { kwg, wordBags };
}

export class KvadratGame {
  private readonly kwg: Uint32Array;
  private readonly wordBags: string[][];
  private board: Board = emptyBoard();
  private pieceQueue: PieceName[] = [];
  private letterQueue: string[] = [];
  private active: ActivePiece | null = null;
  private marks: number[][] = [];
  private clearingRows: number[] = [];
  private clearTimer = 0;
  private pendingScore = 0;
  private pendingWords: WordSegment[] = [];
  private gravityTimer = 0;
  private lockTimer = 0;
  private phase: Exclude<GamePhase, "paused"> = "playing";
  private paused = false;
  private score = 0;
  private lines = 0;
  private pieces = 0;
  private words = 0;
  private totalWordLength = 0;
  private elapsedMs = 0;
  private recentWords: RecentWord[] = [];
  private wordId = 0;

  constructor(assets: GameAssets) {
    this.kwg = assets.kwg;
    this.wordBags = assets.wordBags;
    this.reset();
  }

  reset(): void {
    this.board = emptyBoard();
    this.pieceQueue = [];
    this.letterQueue = [];
    this.active = null;
    this.marks = Array.from({ length: BOARD_HEIGHT }, () => Array(BOARD_WIDTH).fill(0));
    this.clearingRows = [];
    this.clearTimer = 0;
    this.pendingScore = 0;
    this.pendingWords = [];
    this.gravityTimer = 0;
    this.lockTimer = 0;
    this.phase = "playing";
    this.paused = false;
    this.score = 0;
    this.lines = 0;
    this.pieces = 0;
    this.words = 0;
    this.totalWordLength = 0;
    this.elapsedMs = 0;
    this.recentWords = [];
    this.wordId = 0;
    this.ensureQueues();
    this.spawnPiece();
  }

  private ensureQueues(): void {
    while (this.pieceQueue.length < 14) this.pieceQueue.push(...shuffledPieces());
    while (this.letterQueue.length < 56) {
      const bag = this.wordBags[Math.floor(Math.random() * this.wordBags.length)];
      this.letterQueue.push(...bag);
    }
  }

  private spawnPiece(): void {
    this.ensureQueues();
    const piece = this.pieceQueue.shift()!;
    const letters = this.letterQueue.shift()!.split("");
    this.active = { piece, letters, rotation: 0, row: 1, col: 3 };
    if (this.collides(this.active, 0, 0, 0)) {
      this.active.row = 0;
      if (this.collides(this.active, 0, 0, 0)) {
        this.active = null;
        this.phase = "over";
        return;
      }
    }
    this.pieces += 1;
    this.gravityTimer = 0;
    this.lockTimer = 0;
  }

  private blocksFor(piece: ActivePiece, rotation = piece.rotation): Block[] {
    return ROTATIONS[piece.piece][rotation];
  }

  private collides(piece: ActivePiece, deltaRow: number, deltaCol: number, rotation: number): boolean {
    return this.blocksFor(piece, rotation).some((block) => {
      const row = piece.row + deltaRow + block.y;
      const col = piece.col + deltaCol + block.x;
      return row < 0 || row >= BOARD_HEIGHT || col < 0 || col >= BOARD_WIDTH || this.board[row][col] !== null;
    });
  }

  move(direction: -1 | 1): boolean {
    if (!this.canControl() || !this.active) return false;
    if (this.collides(this.active, 0, direction, this.active.rotation)) return false;
    this.active.col += direction;
    this.lockTimer = 0;
    return true;
  }

  rotate(direction: -1 | 1): boolean {
    if (!this.canControl() || !this.active) return false;
    const rotation = (this.active.rotation + direction + 4) % 4;
    for (const [deltaCol, deltaRow] of KICKS) {
      if (!this.collides(this.active, deltaRow, deltaCol, rotation)) {
        this.active.col += deltaCol;
        this.active.row += deltaRow;
        this.active.rotation = rotation;
        this.lockTimer = 0;
        return true;
      }
    }
    return false;
  }

  hardDrop(): boolean {
    if (!this.canControl() || !this.active) return false;
    while (!this.collides(this.active, 1, 0, this.active.rotation)) this.active.row += 1;
    this.lockPiece();
    return true;
  }

  togglePause(): boolean {
    if (this.phase === "over" || this.phase === "complete") return false;
    this.paused = !this.paused;
    return true;
  }

  isValidWord(text: string): boolean {
    return this.isWord(text.trim().toUpperCase());
  }

  private canControl(): boolean {
    return !this.paused && this.phase === "playing";
  }

  tick(deltaMs: number, softDrop: boolean): boolean {
    if (this.paused || this.phase === "over" || this.phase === "complete") return false;
    const delta = Math.min(deltaMs, 50);
    this.elapsedMs += delta;

    if (this.phase === "clearing") {
      this.clearTimer -= delta;
      if (this.clearTimer <= 0) {
        this.finishLineClear();
        return true;
      }
      return false;
    }

    if (!this.active) return false;
    let changed = false;
    const interval = softDrop ? 34 : DEFAULT_GRAVITY_MS;
    this.gravityTimer += delta;
    while (this.gravityTimer >= interval) {
      this.gravityTimer -= interval;
      if (!this.collides(this.active, 1, 0, this.active.rotation)) {
        this.active.row += 1;
        changed = true;
      } else {
        break;
      }
    }

    if (this.collides(this.active, 1, 0, this.active.rotation)) {
      this.lockTimer += delta;
      if (this.lockTimer >= 340) {
        this.lockPiece();
        changed = true;
      }
    } else {
      this.lockTimer = 0;
    }
    return changed;
  }

  private lockPiece(): void {
    if (!this.active) return;
    for (const block of this.blocksFor(this.active)) {
      const row = this.active.row + block.y;
      const col = this.active.col + block.x;
      this.board[row][col] = {
        letter: this.active.letters[block.letterIndex],
        piece: this.active.piece,
      };
    }
    this.active = null;
    this.recalculateWords();
    this.clearingRows = this.board
      .map((row, index) => (row.every(Boolean) ? index : -1))
      .filter((index) => index >= 0);

    if (this.clearingRows.length > 0) {
      this.pendingWords = this.clearingRows.flatMap((row) => this.analyzeRow(this.board[row]));
      this.pendingScore = this.pendingWords.reduce((sum, word) => sum + word.score, 0);
      this.recentWords = [
        ...this.pendingWords.map((word) => ({ id: ++this.wordId, text: word.text, score: word.score })),
        ...this.recentWords,
      ].slice(0, 6);
      this.phase = "clearing";
      this.clearTimer = 420;
    } else {
      this.spawnPiece();
    }
  }

  private finishLineClear(): void {
    const rows = new Set(this.clearingRows);
    this.board = this.board.filter((_, index) => !rows.has(index));
    while (this.board.length < BOARD_HEIGHT) {
      this.board.unshift(Array<StoredCell | null>(BOARD_WIDTH).fill(null));
    }
    this.lines += this.clearingRows.length;
    this.score += this.pendingScore;
    this.words += this.pendingWords.length;
    this.totalWordLength += this.pendingWords.reduce((sum, word) => sum + word.text.length, 0);
    this.clearingRows = [];
    this.pendingScore = 0;
    this.pendingWords = [];
    this.recalculateWords();

    if (this.lines >= MAX_LINES) {
      this.phase = "complete";
      return;
    }
    this.phase = "playing";
    this.spawnPiece();
  }

  private isWord(text: string): boolean {
    if (text.length < 2) return false;
    let nodeIndex = this.kwg[0] & KWG_ARC_MASK;
    let accepts = false;

    for (let letterIndex = 0; letterIndex < text.length; letterIndex += 1) {
      const tile = text.charCodeAt(letterIndex) - 64;
      let found = false;
      for (let index = nodeIndex; index < this.kwg.length; index += 1) {
        const node = this.kwg[index];
        if ((node >>> 24) === tile) {
          found = true;
          accepts = (node & KWG_ACCEPTS) !== 0;
          nodeIndex = node & KWG_ARC_MASK;
          break;
        }
        if ((node & KWG_IS_END) !== 0) break;
      }
      if (!found) return false;
      if (letterIndex < text.length - 1 && nodeIndex === 0) return false;
    }
    return accepts;
  }

  private analyzeRow(row: Array<StoredCell | null>): WordSegment[] {
    const candidates = new Map<number, WordSegment[]>();
    for (let start = 0; start < BOARD_WIDTH; start += 1) {
      if (!row[start]) continue;
      let changedPiece = false;
      for (let end = start; end < BOARD_WIDTH && row[end]; end += 1) {
        if (end > start && row[end]!.piece !== row[end - 1]!.piece) changedPiece = true;
        const length = end - start + 1;
        if (length < 2 || !changedPiece) continue;
        const text = row.slice(start, end + 1).map((cell) => cell!.letter).join("");
        if (!this.isWord(text)) continue;
        const score = scoreWord(text);
        if (score < MINIMUM_WORD_SCORE) continue;
        const segment = { start, end, text, score };
        candidates.set(start, [...(candidates.get(start) ?? []), segment]);
      }
    }

    const bestScore = Array<number>(BOARD_WIDTH + 1).fill(0);
    const choice = Array<WordSegment | null>(BOARD_WIDTH).fill(null);
    for (let position = BOARD_WIDTH - 1; position >= 0; position -= 1) {
      bestScore[position] = bestScore[position + 1];
      for (const candidate of candidates.get(position) ?? []) {
        const total = candidate.score + bestScore[candidate.end + 1];
        if (total > bestScore[position]) {
          bestScore[position] = total;
          choice[position] = candidate;
        }
      }
    }

    const words: WordSegment[] = [];
    for (let position = 0; position < BOARD_WIDTH;) {
      const candidate = choice[position];
      if (candidate && candidate.score + bestScore[candidate.end + 1] === bestScore[position]) {
        words.push(candidate);
        position = candidate.end + 1;
      } else {
        position += 1;
      }
    }
    return words;
  }

  private recalculateWords(): void {
    this.marks = Array.from({ length: BOARD_HEIGHT }, () => Array(BOARD_WIDTH).fill(0));
    for (let row = 0; row < BOARD_HEIGHT; row += 1) {
      for (const word of this.analyzeRow(this.board[row])) {
        for (let col = word.start; col <= word.end; col += 1) this.marks[row][col] = word.score;
      }
    }
  }

  private collidesOnBoard(
    board: Board,
    piece: SearchPiece,
    rotation: number,
    row: number,
    col: number,
  ): boolean {
    return ROTATIONS[piece.piece][rotation].some((block) => {
      const boardRow = row + block.y;
      const boardCol = col + block.x;
      return boardRow < 0 || boardRow >= BOARD_HEIGHT || boardCol < 0 ||
        boardCol >= BOARD_WIDTH || board[boardRow][boardCol] !== null;
    });
  }

  private simulatePlacements(board: Board, piece: SearchPiece): SimulatedPlacement[] {
    const placements: SimulatedPlacement[] = [];

    for (let rotation = 0; rotation < 4; rotation += 1) {
      const blocks = ROTATIONS[piece.piece][rotation];
      const minX = Math.min(...blocks.map((block) => block.x));
      const maxX = Math.max(...blocks.map((block) => block.x));
      const minY = Math.min(...blocks.map((block) => block.y));

      for (let col = -minX; col < BOARD_WIDTH - maxX; col += 1) {
        let row = -minY;
        if (this.collidesOnBoard(board, piece, rotation, row, col)) continue;
        while (!this.collidesOnBoard(board, piece, rotation, row + 1, col)) row += 1;

        const placedBoard = board.map((boardRow) => boardRow.slice());
        for (const block of blocks) {
          placedBoard[row + block.y][col + block.x] = {
            letter: piece.letters[block.letterIndex],
            piece: piece.piece,
          };
        }

        const clearingRows = placedBoard
          .map((boardRow, index) => (boardRow.every(Boolean) ? index : -1))
          .filter((index) => index >= 0);
        const words = clearingRows.flatMap((rowIndex) => this.analyzeRow(placedBoard[rowIndex]));
        const score = words.reduce((sum, word) => sum + word.score, 0);
        const cleared = new Set(clearingRows);
        const collapsedBoard = placedBoard.filter((_, index) => !cleared.has(index));
        while (collapsedBoard.length < BOARD_HEIGHT) {
          collapsedBoard.unshift(Array<StoredCell | null>(BOARD_WIDTH).fill(null));
        }

        placements.push({
          board: collapsedBoard,
          rotation,
          row,
          col,
          score,
          lines: clearingRows.length,
          words,
        });
      }
    }

    return placements;
  }

  private evaluateBoard(board: Board): BoardEvaluation {
    const heights: number[] = [];
    let holes = 0;
    let buriedDepth = 0;

    for (let col = 0; col < BOARD_WIDTH; col += 1) {
      let firstFilled = BOARD_HEIGHT;
      let blocksAbove = 0;
      for (let row = 0; row < BOARD_HEIGHT; row += 1) {
        if (board[row][col]) {
          if (firstFilled === BOARD_HEIGHT) firstFilled = row;
          blocksAbove += 1;
        } else if (firstFilled !== BOARD_HEIGHT) {
          holes += 1;
          buriedDepth += blocksAbove;
        }
      }
      heights.push(BOARD_HEIGHT - firstFilled);
    }

    const aggregateHeight = heights.reduce((sum, height) => sum + height, 0);
    const maximumHeight = Math.max(...heights);
    const bumpiness = heights.slice(1).reduce(
      (sum, height, index) => sum + Math.abs(height - heights[index]),
      0,
    );
    let wells = 0;
    for (let col = 0; col < BOARD_WIDTH; col += 1) {
      const left = col === 0 ? BOARD_HEIGHT : heights[col - 1];
      const right = col === BOARD_WIDTH - 1 ? BOARD_HEIGHT : heights[col + 1];
      wells += Math.max(0, Math.min(left, right) - heights[col]);
    }

    let wordPotential = 0;
    const wordCandidates: Array<{ text: string; value: number }> = [];
    for (const row of board) {
      const fullness = row.filter(Boolean).length / BOARD_WIDTH;
      if (fullness < 0.2 || fullness === 1) continue;
      for (const word of this.analyzeRow(row)) {
        const value = word.score * (0.18 + 1.12 * fullness ** 4);
        wordPotential += value;
        wordCandidates.push({ text: word.text, value });
      }
    }

    const setupWords = [...new Map(
      wordCandidates
        .sort((left, right) => right.value - left.value)
        .map((candidate) => [candidate.text, candidate]),
    ).values()].slice(0, 4).map((candidate) => candidate.text);

    return {
      value: wordPotential - holes * 285 - buriedDepth * 34 - aggregateHeight * 4.5 -
        maximumHeight * maximumHeight * 1.7 - bumpiness * 8 - wells * 3,
      setupWords,
    };
  }

  findBestMove(depth = 3, beamWidth = 64): BotPlan | null {
    if (!this.canControl() || !this.active) return null;
    this.ensureQueues();

    const searchDepth = Math.max(1, Math.min(5, Math.floor(depth)));
    const width = Math.max(12, Math.min(160, Math.floor(beamWidth)));
    const sequence: SearchPiece[] = [{
      piece: this.active.piece,
      letters: [...this.active.letters],
    }];
    for (let index = 0; index < searchDepth - 1; index += 1) {
      sequence.push({
        piece: this.pieceQueue[index],
        letters: this.letterQueue[index].split(""),
      });
    }

    let nodes = 0;
    let reachedDepth = 1;
    let frontier: SearchState[] = this.simulatePlacements(this.board, sequence[0]).map((placement) => {
      nodes += 1;
      const boardEvaluation = this.evaluateBoard(placement.board);
      return {
        board: placement.board,
        score: placement.score,
        lines: placement.lines,
        root: placement,
        value: placement.score * 1.25 - placement.lines * 90 + boardEvaluation.value,
        setupWords: boardEvaluation.setupWords,
      };
    });
    if (frontier.length === 0) return null;
    frontier.sort((left, right) => right.value - left.value);
    frontier = frontier.slice(0, width);
    const finished: SearchState[] = [];

    for (let ply = 1; ply < sequence.length && frontier.length > 0; ply += 1) {
      const next: SearchState[] = [];
      for (const state of frontier) {
        if (this.lines + state.lines >= MAX_LINES) {
          finished.push(state);
          continue;
        }
        for (const placement of this.simulatePlacements(state.board, sequence[ply])) {
          nodes += 1;
          const score = state.score + placement.score;
          const lines = state.lines + placement.lines;
          const boardEvaluation = this.evaluateBoard(placement.board);
          next.push({
            board: placement.board,
            score,
            lines,
            root: state.root,
            value: score * 1.25 - lines * 90 + boardEvaluation.value,
            setupWords: boardEvaluation.setupWords,
          });
        }
      }
      if (next.length === 0) break;
      reachedDepth = ply + 1;
      next.sort((left, right) => right.value - left.value);
      frontier = next.slice(0, width);
    }

    const candidates = [...frontier, ...finished].sort((left, right) => right.value - left.value);
    const best = candidates[0];
    if (!best) return null;
    const rootEvaluation = this.evaluateBoard(best.root.board);
    const immediateWords = best.root.words.map((word) => word.text);
    const setupWords = rootEvaluation.setupWords.length > 0 ? rootEvaluation.setupWords : best.setupWords;
    const leftmostColumn = Math.min(...ROTATIONS[this.active.piece][best.root.rotation]
      .map((block) => best.root.col + block.x)) + 1;
    const placementText = `left edge column ${leftmostColumn}, ${best.root.rotation * 90}°`;
    let reason = `Build from ${placementText} while keeping the surface open for the next ${reachedDepth - 1} piece${reachedDepth === 2 ? "" : "s"}.`;
    if (immediateWords.length > 0) {
      reason = `Bank ${immediateWords.join(" + ")} for ${best.root.score.toLocaleString()} points at ${placementText}.`;
    } else if (setupWords.length > 0) {
      reason = `Preserve ${setupWords.slice(0, 2).join(" / ")} as live scoring material from ${placementText}.`;
    }

    return {
      piece: this.active.piece,
      letters: [...this.active.letters],
      rotation: best.root.rotation,
      row: best.root.row,
      col: best.root.col,
      immediateScore: best.root.score,
      immediateLines: best.root.lines,
      immediateWords,
      projectedScore: best.score,
      projectedLines: best.lines,
      setupWords,
      depth: reachedDepth,
      nodes,
      evaluation: Math.round(best.value),
      reason,
    };
  }

  executeBotPlan(plan: BotPlan): boolean {
    if (!this.canControl() || !this.active || this.active.piece !== plan.piece ||
      this.active.letters.join("") !== plan.letters.join("")) return false;
    const rotation = ((plan.rotation % 4) + 4) % 4;
    if (this.collidesOnBoard(this.board, this.active, rotation, plan.row, plan.col) ||
      !this.collidesOnBoard(this.board, this.active, rotation, plan.row + 1, plan.col)) return false;
    this.active.rotation = rotation;
    this.active.row = plan.row;
    this.active.col = plan.col;
    this.lockPiece();
    return true;
  }

  private ghostRow(): number {
    if (!this.active) return 0;
    let row = this.active.row;
    while (!this.collides({ ...this.active, row }, 1, 0, this.active.rotation)) row += 1;
    return row;
  }

  getSnapshot(): GameSnapshot {
    const renderBoard: Array<Array<RenderCell | null>> = this.board.map((row, rowIndex) =>
      row.map((cell, colIndex) => cell ? {
        letter: cell.letter,
        value: LETTER_VALUES[cell.letter],
        piece: cell.piece,
        wordScore: this.marks[rowIndex][colIndex] || undefined,
        clearing: this.clearingRows.includes(rowIndex) || undefined,
      } : null),
    );

    if (this.active && this.phase === "playing") {
      const ghostRow = this.ghostRow();
      for (const block of this.blocksFor(this.active)) {
        const ghostBoardRow = ghostRow + block.y;
        const col = this.active.col + block.x;
        if (!renderBoard[ghostBoardRow][col]) {
          const letter = this.active.letters[block.letterIndex];
          renderBoard[ghostBoardRow][col] = {
            letter,
            value: LETTER_VALUES[letter],
            piece: this.active.piece,
            ghost: true,
          };
        }
      }
      for (const block of this.blocksFor(this.active)) {
        const row = this.active.row + block.y;
        const col = this.active.col + block.x;
        const letter = this.active.letters[block.letterIndex];
        renderBoard[row][col] = {
          letter,
          value: LETTER_VALUES[letter],
          piece: this.active.piece,
          active: true,
        };
      }
    }

    this.ensureQueues();
    const next = this.pieceQueue.slice(0, 4).map((piece, index) => ({
      piece,
      cells: ROTATIONS[piece][0].map((block) => ({
        x: block.x,
        y: block.y,
        letter: this.letterQueue[index][block.letterIndex],
      })),
    }));

    return {
      board: renderBoard.slice(BOARD_HEIGHT - VISIBLE_HEIGHT),
      next,
      score: this.score,
      lines: this.lines,
      pieces: this.pieces,
      words: this.words,
      averageWordLength: this.words ? this.totalWordLength / this.words : 0,
      elapsedMs: this.elapsedMs,
      phase: this.paused ? "paused" : this.phase,
      recentWords: this.recentWords,
    };
  }
}
