import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const manifest = resolve(root, "wasm-strategy/Cargo.toml");
const artifact = resolve(root, "wasm-strategy/target/wasm32-unknown-unknown/release/kvadrat_strategy.wasm");
const destination = resolve(root, "public/wasm/kvadrat-strategy.wasm");
const manifestPath = resolve(root, "public/wasm/MANIFEST.json");

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
let args = ["build", "--manifest-path", manifest, "--lib", "--target", "wasm32-unknown-unknown", "--release"];
const env = { ...process.env };
if (rustup) {
  const located = spawnSync(rustup, ["which", "rustc", "--toolchain", "stable"], { encoding: "utf8" });
  if (located.status !== 0) throw new Error(located.stderr || "Could not locate stable rustc through rustup.");
  env.RUSTC = located.stdout.trim();
  command = rustup;
  args = ["run", "stable", "cargo", ...args];
}

const built = spawnSync(command, args, { cwd: root, env, encoding: "utf8", stdio: "inherit" });
if (built.status !== 0) process.exit(built.status ?? 1);

mkdirSync(resolve(root, "public/wasm"), { recursive: true });
copyFileSync(artifact, destination);
const bytes = readFileSync(destination);
const compiler = spawnSync(env.RUSTC ?? "rustc", ["--version"], { encoding: "utf8" });
const manifestData = {
  abiVersion: 1,
  target: "wasm32-unknown-unknown",
  rustc: compiler.status === 0 ? compiler.stdout.trim() : "unknown",
  bytes: bytes.byteLength,
  sha256: createHash("sha256").update(bytes).digest("hex"),
};
writeFileSync(manifestPath, `${JSON.stringify(manifestData, null, 2)}\n`);
console.log(`Wrote ${destination} (${statSync(destination).size.toLocaleString()} bytes)`);
console.log(`Wrote ${manifestPath}`);
