/**
 * Child-process helpers for the Kiwibro browser rig:
 *  - startApp(): real Flask app on a scratch port/data dir
 *  - startFakeLm(): deterministic fake LM Studio (python tests/e2e/fake_lm.py)
 *  - stopAll(): kill both on exit
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, "..", "..", "..");
const RUNTIME = path.join(REPO_ROOT, ".dev", "qa", "browser");

const children = new Set();
let cleaned = false;

export function pythonBin() {
  return process.env.PYTHON || "python3";
}

export function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

export async function waitForHealth(baseUrl, { timeoutMs = 25000, proc = null } = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastError = "";
  while (Date.now() < deadline) {
    if (proc && proc.exitCode !== null) throw new Error(`app exited during boot (code ${proc.exitCode})`);
    try {
      const response = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(1000) });
      if (response.ok) {
        const data = await response.json();
        if (data.ok) return data;
      }
    } catch (error) {
      lastError = String(error);
    }
    await sleep(80);
  }
  throw new Error(`app did not become healthy at ${baseUrl} (${lastError})`);
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function freshDir(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

/** Start `python3 app.py` with isolated data dir/workspace. */
export async function startApp({ label = "app", lmstudioUrl = null } = {}) {
  const port = await freePort();
  const runDir = freshDir(path.join(RUNTIME, label));
  const dataDir = path.join(runDir, "data");
  const workspace = path.join(runDir, "workspace");
  fs.mkdirSync(dataDir, { recursive: true });
  fs.mkdirSync(workspace, { recursive: true });
  const logPath = path.join(runDir, "app.log");
  const logStream = fs.createWriteStream(logPath);
  const env = {
    ...process.env,
    KIWIBRO_HOST: "127.0.0.1",
    KIWIBRO_PORT: String(port),
    KIWIBRO_DATA_DIR: dataDir,
    KIWIBRO_WORKSPACE: workspace,
    PYTHONUNBUFFERED: "1",
    PYTHONDONTWRITEBYTECODE: "1",
  };
  if (lmstudioUrl) env.KIWIBRO_LMSTUDIO_URL = lmstudioUrl;
  const proc = spawn(pythonBin(), ["app.py"], {
    cwd: REPO_ROOT,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  children.add(proc);
  proc.stdout.pipe(logStream);
  proc.stderr.pipe(logStream);
  proc.on("exit", () => children.delete(proc));
  const baseUrl = `http://127.0.0.1:${port}`;
  try {
    await waitForHealth(baseUrl, { proc });
  } catch (error) {
    proc.kill("SIGKILL");
    const log = fs.existsSync(logPath) ? fs.readFileSync(logPath, "utf8").slice(-4000) : "";
    throw new Error(`${error.message}\n--- app.log ---\n${log}`);
  }
  return {
    proc,
    port,
    baseUrl,
    dataDir,
    workspace,
    logPath,
    async stop() {
      await killChild(proc);
    },
  };
}

/** Start the deterministic fake LM Studio server; returns its URL. */
export async function startFakeLm({ models = "default" } = {}) {
  const proc = spawn(pythonBin(), ["tests/e2e/fake_lm.py", "--port", "0", "--models", models, "--print-url"], {
    cwd: REPO_ROOT,
    stdio: ["ignore", "pipe", "pipe"],
  });
  children.add(proc);
  let stderr = "";
  proc.stderr.on("data", (chunk) => (stderr += String(chunk)));
  try {
    const url = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`fake LM did not start: ${stderr}`)), 15000);
      let buffer = "";
      proc.stdout.on("data", (chunk) => {
        buffer += String(chunk);
        const line = buffer.split("\n").find((entry) => entry.trim().startsWith("{"));
        if (line) {
          clearTimeout(timer);
          const parsed = JSON.parse(line);
          resolve(parsed.url);
        }
      });
      proc.on("exit", (code) => {
        clearTimeout(timer);
        reject(new Error(`fake LM exited early (${code}): ${stderr}`));
      });
    });
    return { proc, url, async stop() { await killChild(proc); } };
  } catch (error) {
    proc.kill("SIGKILL");
    throw error;
  }
}

export async function killChild(proc) {
  if (!proc || proc.exitCode !== null) return;
  proc.kill("SIGTERM");
  const exited = await Promise.race([
    new Promise((resolve) => proc.once("exit", () => resolve(true))),
    sleep(4000).then(() => false),
  ]);
  if (!exited && proc.exitCode === null) proc.kill("SIGKILL");
}

export function killAll() {
  if (cleaned) return;
  cleaned = true;
  for (const proc of children) {
    try {
      proc.kill("SIGKILL");
    } catch {
      /* already gone */
    }
  }
  children.clear();
}
