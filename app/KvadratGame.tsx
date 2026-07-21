"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  BotPlan,
  GameSnapshot,
  KvadratGame,
  LEXICA,
  LexiconId,
  MAX_LINES,
  PreviewPiece,
  loadGameAssets,
} from "./game-engine";

const KEYBOARD_KEYS = new Set([
  "ArrowLeft", "ArrowRight", "ArrowDown", "ArrowUp", "Space", "KeyZ",
  "KeyX", "KeyP", "Escape", "KeyR",
]);

function formatTime(milliseconds: number): string {
  const totalSeconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const hundredths = Math.floor((milliseconds % 1000) / 10);
  return `${minutes}:${seconds.toString().padStart(2, "0")}.${hundredths.toString().padStart(2, "0")}`;
}

function MiniPiece({ preview, featured = false }: { preview: PreviewPiece; featured?: boolean }) {
  const cells = Array.from({ length: 16 }, () => null as { letter: string } | null);
  for (const cell of preview.cells) cells[cell.y * 4 + cell.x] = { letter: cell.letter };
  return (
    <div className={`mini-piece piece-${preview.piece.toLowerCase()} ${featured ? "featured" : ""}`}>
      {cells.map((cell, index) => (
        <div className={`mini-cell ${cell ? "filled" : ""}`} key={index}>
          {cell?.letter}
        </div>
      ))}
    </div>
  );
}

function Stat({ label, value, detail }: { label: string; value: string | number; detail?: string }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {detail && <span className="stat-detail">{detail}</span>}
    </div>
  );
}

function LexiconSelect({ value, compact = false, onChange }: {
  value: LexiconId;
  compact?: boolean;
  onChange: (lexicon: LexiconId) => void;
}) {
  return (
    <label className={compact ? "lexicon-select compact" : "lexicon-select"}>
      {!compact && <span>English ·</span>}
      <select
        aria-label="English lexicon"
        value={value}
        onChange={(event) => onChange(event.target.value as LexiconId)}
      >
        {Object.entries(LEXICA).map(([id, details]) => (
          <option value={id} key={id}>{details.name} · {details.region}</option>
        ))}
      </select>
    </label>
  );
}

export default function KvadratGameView() {
  const engineRef = useRef<KvadratGame | null>(null);
  const heldKeysRef = useRef(new Set<string>());
  const keyboardRepeatRef = useRef({ direction: 0, nextAt: 0 });
  const gamepadButtonsRef = useRef<boolean[]>([]);
  const gamepadRepeatRef = useRef({ direction: 0, nextAt: 0 });
  const gamepadSoftDropRef = useRef(false);
  const lastFrameRef = useRef(0);
  const lastRenderRef = useRef(0);
  const [snapshot, setSnapshot] = useState<GameSnapshot | null>(null);
  const [loadError, setLoadError] = useState("");
  const [controller, setController] = useState("");
  const controllerRef = useRef("");
  const [bestScore, setBestScore] = useState(0);
  const [botEnabled, setBotEnabled] = useState(false);
  const [botPlan, setBotPlan] = useState<BotPlan | null>(null);
  const [lexicon, setLexicon] = useState<LexiconId>("CSW24");

  const selectLexicon = useCallback((nextLexicon: LexiconId) => {
    if (nextLexicon === lexicon) return;
    engineRef.current = null;
    setSnapshot(null);
    setLoadError("");
    setBotEnabled(false);
    setBotPlan(null);
    setLexicon(nextLexicon);
  }, [lexicon]);

  const applySnapshot = useCallback((nextSnapshot: GameSnapshot) => {
    setSnapshot(nextSnapshot);
    if (nextSnapshot.phase === "complete") {
      setBestScore((currentBest) => {
        if (nextSnapshot.score <= currentBest) return currentBest;
        window.localStorage.setItem(`kvadrat-best-score-${lexicon}`, String(nextSnapshot.score));
        return nextSnapshot.score;
      });
    }
  }, [lexicon]);

  const refresh = useCallback(() => {
    if (engineRef.current) applySnapshot(engineRef.current.getSnapshot());
  }, [applySnapshot]);

  const restart = useCallback(() => {
    setBotEnabled(false);
    setBotPlan(null);
    engineRef.current?.reset();
    refresh();
  }, [refresh]);

  const requestHint = useCallback(() => {
    const engine = engineRef.current;
    if (!engine) return;
    setBotPlan(engine.findBestMove(4, 72));
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadGameAssets(lexicon)
      .then((assets) => {
        if (cancelled) return;
        const storedBest = window.localStorage.getItem(`kvadrat-best-score-${lexicon}`) ??
          (lexicon === "CSW24" ? window.localStorage.getItem("kvadrat-best-score") : null);
        setBestScore(Number(storedBest ?? 0));
        engineRef.current = new KvadratGame(assets);
        applySnapshot(engineRef.current.getSnapshot());
      })
      .catch((error: unknown) => {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : "Game data failed to load.");
      });
    return () => { cancelled = true; };
  }, [applySnapshot, lexicon]);

  useEffect(() => {
    const executeKey = (code: string) => {
      const engine = engineRef.current;
      if (!engine) return;
      let changed = false;
      if (code === "ArrowLeft") changed = engine.move(-1);
      if (code === "ArrowRight") changed = engine.move(1);
      if (code === "ArrowUp" || code === "KeyX") changed = engine.rotate(1);
      if (code === "KeyZ") changed = engine.rotate(-1);
      if (code === "Space") changed = engine.hardDrop();
      if (code === "KeyP" || code === "Escape") changed = engine.togglePause();
      if (code === "KeyR") { engine.reset(); changed = true; }
      if (changed) refresh();
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (!KEYBOARD_KEYS.has(event.code)) return;
      event.preventDefault();
      if (!event.repeat) {
        heldKeysRef.current.add(event.code);
        executeKey(event.code);
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      heldKeysRef.current.delete(event.code);
    };
    const onBlur = () => {
      heldKeysRef.current.clear();
      gamepadSoftDropRef.current = false;
    };

    window.addEventListener("keydown", onKeyDown, { passive: false });
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [refresh]);

  useEffect(() => {
    let animationFrame = 0;

    const frame = (now: number) => {
      const engine = engineRef.current;
      const delta = lastFrameRef.current ? now - lastFrameRef.current : 0;
      lastFrameRef.current = now;
      let changed = false;

      if (engine) {
        const held = heldKeysRef.current;
        const keyboardDirection = held.has("ArrowLeft") === held.has("ArrowRight")
          ? 0
          : held.has("ArrowLeft") ? -1 : 1;
        const keyboardRepeat = keyboardRepeatRef.current;
        if (keyboardDirection !== keyboardRepeat.direction) {
          keyboardRepeat.direction = keyboardDirection;
          keyboardRepeat.nextAt = now + 145;
        } else if (keyboardDirection && now >= keyboardRepeat.nextAt) {
          changed = engine.move(keyboardDirection as -1 | 1) || changed;
          keyboardRepeat.nextAt = now + 38;
        }

        const pads = navigator.getGamepads?.() ?? [];
        const pad = Array.from(pads).find(Boolean) ?? null;
        const nextController = pad?.id ?? "";
        if (nextController !== controllerRef.current) {
          controllerRef.current = nextController;
          setController(nextController);
        }

        if (pad) {
          const pressed = pad.buttons.map((button) => button.pressed);
          const wasPressed = gamepadButtonsRef.current;
          const edge = (index: number) => pressed[index] && !wasPressed[index];
          const left = pressed[14] || pad.axes[0] < -0.55;
          const right = pressed[15] || pad.axes[0] > 0.55;
          const direction = left === right ? 0 : left ? -1 : 1;
          const repeat = gamepadRepeatRef.current;
          if (direction !== repeat.direction) {
            repeat.direction = direction;
            repeat.nextAt = now + 145;
            if (direction) changed = engine.move(direction as -1 | 1) || changed;
          } else if (direction && now >= repeat.nextAt) {
            changed = engine.move(direction as -1 | 1) || changed;
            repeat.nextAt = now + 38;
          }
          gamepadSoftDropRef.current = Boolean(pressed[13] || pad.axes[1] > 0.55);
          if (edge(12)) changed = engine.hardDrop() || changed;
          if (edge(0)) changed = engine.rotate(1) || changed;
          if (edge(1)) changed = engine.rotate(-1) || changed;
          if (edge(9)) changed = engine.togglePause() || changed;
          if (edge(8)) { engine.reset(); changed = true; }
          gamepadButtonsRef.current = pressed;
        } else {
          gamepadSoftDropRef.current = false;
          gamepadButtonsRef.current = [];
        }

        changed = engine.tick(
          delta,
          heldKeysRef.current.has("ArrowDown") || gamepadSoftDropRef.current,
        ) || changed;

        if (changed || now - lastRenderRef.current > 50) {
          lastRenderRef.current = now;
          applySnapshot(engine.getSnapshot());
        }
      }
      animationFrame = window.requestAnimationFrame(frame);
    };
    animationFrame = window.requestAnimationFrame(frame);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [applySnapshot]);

  useEffect(() => {
    if (!botEnabled) return;
    let cancelled = false;
    let timer = 0;

    const playNextMove = () => {
      if (cancelled) return;
      const engine = engineRef.current;
      if (!engine) {
        timer = window.setTimeout(playNextMove, 160);
        return;
      }

      const current = engine.getSnapshot();
      if (current.phase === "over" || current.phase === "complete") {
        setBotEnabled(false);
        return;
      }
      if (current.phase === "playing") {
        const plan = engine.findBestMove(4, 72);
        setBotPlan(plan);
        if (plan && engine.executeBotPlan(plan)) applySnapshot(engine.getSnapshot());
      }
      timer = window.setTimeout(playNextMove, current.phase === "clearing" ? 140 : 680);
    };

    timer = window.setTimeout(playNextMove, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [applySnapshot, botEnabled]);

  if (loadError) {
    return (
      <main className="loading-screen">
        <div className="load-mark">K</div>
        <h1>Couldn’t load Kvadrat</h1>
        <p>{loadError}</p>
        <button onClick={() => window.location.reload()}>Try again</button>
      </main>
    );
  }

  if (!snapshot) {
    return (
      <main className="loading-screen">
        <div className="load-mark">K</div>
        <h1>Building the letter bag…</h1>
        <p>Loading {lexicon} and preparing your first seven pieces.</p>
      </main>
    );
  }

  const phaseLabel = snapshot.phase === "playing" || snapshot.phase === "clearing"
    ? "LIVE"
    : snapshot.phase.toUpperCase();

  return (
    <main className="game-shell">
      <header className="topbar">
        <div className="brand-group">
          <div className="brand-mark" aria-hidden="true">K</div>
          <div>
            <h1>KVADRAT</h1>
            <p>Words under pressure</p>
          </div>
        </div>
        <div className="mode-heading">
          <span className="eyebrow">SOLO</span>
          <strong>40 LINES</strong>
          <LexiconSelect value={lexicon} onChange={selectLexicon} />
        </div>
        <div className="session-actions">
          <LexiconSelect value={lexicon} compact onChange={selectLexicon} />
          <span className={`status-pill ${snapshot.phase}`}><i />{phaseLabel}</span>
          <button
            className={botEnabled ? "bot-toggle active" : "bot-toggle"}
            onClick={() => setBotEnabled((enabled) => !enabled)}
          >
            {botEnabled ? "Stop bot" : "Watch bot"}
          </button>
          <button onClick={() => { engineRef.current?.togglePause(); refresh(); }}>
            {snapshot.phase === "paused" ? "Resume" : "Pause"}
          </button>
          <button className="primary-action" onClick={restart}>Restart</button>
        </div>
      </header>

      <section className="game-layout" aria-label="Kvadrat single player game">
        <aside className="panel stats-panel">
          <div className="panel-heading">
            <span>Run</span><small>Local best {bestScore.toLocaleString()}</small>
          </div>
          <div className="score-block">
            <span>Score</span>
            <strong>{snapshot.score.toLocaleString()}</strong>
            <small>{snapshot.lines ? Math.round(snapshot.score / snapshot.lines).toLocaleString() : "—"} per line</small>
          </div>
          <div className="stats-grid">
            <Stat label="Lines" value={`${snapshot.lines}/${MAX_LINES}`} detail={`${Math.min(100, (snapshot.lines / MAX_LINES) * 100).toFixed(0)}%`} />
            <Stat label="Time" value={formatTime(snapshot.elapsedMs)} />
            <Stat label="Pieces" value={snapshot.pieces} detail={`${(snapshot.pieces / Math.max(1, snapshot.elapsedMs / 1000)).toFixed(2)}/s`} />
            <Stat label="Words" value={snapshot.words} detail={snapshot.words ? `${snapshot.averageWordLength.toFixed(2)} avg` : "— avg"} />
          </div>
          <div className="progress-track" aria-label={`${snapshot.lines} of ${MAX_LINES} lines`}>
            <div style={{ width: `${Math.min(100, (snapshot.lines / MAX_LINES) * 100)}%` }} />
          </div>
          <div className="rule-note">
            <span className="rule-icon">Aa</span>
            <p>Words score when they cross colors. Clear the line to bank them.</p>
          </div>
        </aside>

        <div className="board-wrap">
          <div className="board-frame">
            <div className="board" role="grid" aria-label="10 by 20 Kvadrat playfield">
              {snapshot.board.flatMap((row, rowIndex) => row.map((cell, colIndex) => (
                <div
                  className={[
                    "board-cell",
                    cell ? "occupied" : "",
                    cell ? `piece-${cell.piece.toLowerCase()}` : "",
                    cell?.active ? "active" : "",
                    cell?.ghost ? "ghost" : "",
                    cell?.clearing ? "clearing" : "",
                    cell?.wordScore ? "word-cell" : "",
                  ].filter(Boolean).join(" ")}
                  key={`${rowIndex}-${colIndex}`}
                  role="gridcell"
                  aria-label={cell ? `${cell.letter}, ${cell.piece} piece` : "empty"}
                >
                  {cell && <><span className="letter">{cell.letter}</span><span className="letter-value">{cell.value}</span></>}
                </div>
              )))}
            </div>
            <div className="board-footer"><span>DROP ZONE</span><span>10 × 20</span></div>
          </div>

          {snapshot.phase !== "playing" && snapshot.phase !== "clearing" && (
            <div className="game-overlay">
              <span className="eyebrow">{snapshot.phase === "complete" ? "RUN COMPLETE" : snapshot.phase === "over" ? "TOPPED OUT" : "GAME PAUSED"}</span>
              <h2>{snapshot.phase === "complete" ? snapshot.score.toLocaleString() : snapshot.phase === "over" ? "Good run." : "Take a breath."}</h2>
              <p>{snapshot.phase === "complete" ? `${snapshot.lines} lines in ${formatTime(snapshot.elapsedMs)}` : snapshot.phase === "over" ? `${snapshot.score.toLocaleString()} points · ${snapshot.lines} lines` : "Your timer is stopped."}</p>
              <button onClick={() => {
                if (snapshot.phase === "paused") engineRef.current?.togglePause();
                else engineRef.current?.reset();
                refresh();
              }}>{snapshot.phase === "paused" ? "Resume" : "Play again"}</button>
            </div>
          )}
        </div>

        <aside className="panel queue-panel">
          <div className="panel-heading"><span>Next</span><small>7-bag</small></div>
          <div className="next-stack">
            {snapshot.next.map((preview, index) => (
              <MiniPiece preview={preview} featured={index === 0} key={`${preview.piece}-${index}`} />
            ))}
          </div>
          <div className="bot-panel">
            <div className="panel-heading">
              <span>Strategy engine</span>
              <small>{botEnabled ? "Autoplay" : botPlan ? `Depth ${botPlan.depth}` : `${lexicon} beam`}</small>
            </div>
            <div className="bot-readout" aria-live="polite">
              {botPlan ? (
                <>
                  <div className="bot-metrics">
                    <span>{botPlan.nodes.toLocaleString()} nodes</span>
                    <span>{botPlan.projectedLines} line{botPlan.projectedLines === 1 ? "" : "s"} forecast</span>
                  </div>
                  <p>{botPlan.reason}</p>
                  {(botPlan.projectedScore > 0 || botPlan.setupWords.length > 0) && (
                    <div className="bot-chips">
                      {botPlan.projectedScore > 0 && <span>+{botPlan.projectedScore.toLocaleString()} projected</span>}
                      {botPlan.setupWords.slice(0, 2).map((word) => <span key={word}>{word}</span>)}
                    </div>
                  )}
                </>
              ) : (
                <p>Search four known pieces for score, word setups, and a stable board.</p>
              )}
            </div>
            <div className="bot-actions">
              <button onClick={requestHint} disabled={snapshot.phase !== "playing"}>Hint</button>
              <button
                className={botEnabled ? "active" : ""}
                onClick={() => setBotEnabled((enabled) => !enabled)}
              >
                {botEnabled ? "Stop" : "Watch bot"}
              </button>
            </div>
          </div>
          <div className="word-feed">
            <div className="panel-heading"><span>Word feed</span><small>Banked</small></div>
            {snapshot.recentWords.length ? snapshot.recentWords.map((word) => (
              <div className="word-row" key={word.id}>
                <span>{word.text}</span><strong>+{word.score}</strong>
              </div>
            )) : <p className="empty-feed">Completed words will appear here.</p>}
          </div>
        </aside>
      </section>

      <footer className="controls-bar">
        <div className="input-status">
          <i className={controller ? "connected" : ""} />
          <span>{controller ? "Controller connected" : "Keyboard ready"}</span>
        </div>
        <div className="controls-list" aria-label="Game controls">
          <span><kbd>←</kbd><kbd>→</kbd> Move</span>
          <span><kbd>↓</kbd> Soft drop</span>
          <span><kbd>Space</kbd> Hard drop</span>
          <span><kbd>Z</kbd><kbd>X</kbd> Rotate</span>
          <span><kbd>P</kbd> Pause</span>
          <span><kbd>R</kbd> Restart</span>
        </div>
        <span className="build-label">SCORING BOT · DEPTH 4</span>
      </footer>
    </main>
  );
}
