import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const manifest = resolve(root, "wasm-strategy/Cargo.toml");

function succeeds(command, args) {
  return spawnSync(command, args, { encoding: "utf8" }).status === 0;
}

const rustupCandidates = [
  process.env.RUSTUP,
  "rustup",
  "/opt/homebrew/opt/rustup/bin/rustup",
].filter(Boolean);
const rustup = rustupCandidates.find((candidate) =>
  existsSync(candidate) || succeeds(candidate, ["--version"]));

let command = "cargo";
let args = [
  "run",
  "--release",
  "--manifest-path",
  manifest,
  "--bin",
  "kvadrat-self-play",
  "--",
  ...process.argv.slice(2),
];
const env = { ...process.env };
if (rustup) {
  const located = spawnSync(rustup, ["which", "rustc", "--toolchain", "stable"], { encoding: "utf8" });
  if (located.status !== 0) throw new Error(located.stderr || "Could not locate stable rustc through rustup.");
  env.RUSTC = located.stdout.trim();
  command = rustup;
  args = ["run", "stable", "cargo", ...args];
}

const result = spawnSync(command, args, { cwd: root, env, stdio: "inherit" });
if (result.error) throw result.error;
process.exit(result.status ?? 1);
