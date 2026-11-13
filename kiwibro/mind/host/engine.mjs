// Node wrapper around the Kiwi Mind WebAssembly engine (kiwibro/mind/engine/kiwimind.c).
//
// Derived from the Flyingkiwi engine wrapper by the same author. The engine owns all model memory; this file
// copies verified bytes in (weights are read from disk straight into WebAssembly memory) and reads results out.
// Plain Node 20+, no dependencies.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';

const WASM_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'wasm');

// Smallest modules that validate only where the feature exists (SIMD128; relaxed SIMD's i7 dot product).
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 22, 1, 20, 0,
  253, 12, ...new Array(16).fill(0), 11]);
const RELAXED_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 61, 1, 59,
  0, 253, 12, ...new Array(16).fill(0), 253, 12, ...new Array(16).fill(0), 253, 12, ...new Array(16).fill(0), 253, 147, 2, 11]);

export function wasmFeatures() {
  const valid = bytes => { try { return WebAssembly.validate(bytes); } catch { return false; } };
  let threads = false;
  try {
    threads = typeof SharedArrayBuffer === 'function' && typeof Worker === 'function' &&
      new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true }).buffer instanceof SharedArrayBuffer;
  } catch { /* no shared memory */ }
  return { simd: valid(SIMD_PROBE), relaxed: valid(RELAXED_PROBE), threads };
}

export const VARIANTS = ['relaxed-mt', 'simd-mt', 'simd'];

export function pickVariant(features = wasmFeatures()) {
  if (!features.simd) throw new Error('this Node.js has no WebAssembly SIMD');
  if (features.threads && features.relaxed) return 'relaxed-mt';
  return features.threads ? 'simd-mt' : 'simd';
}

export function defaultThreads() { return Math.max(1, Math.min(6, os.availableParallelism() - 2)); }

const INIT_ERRORS = {
  '-1': 'not a trunk file', '-2': 'unsupported model shape (sliding window, flags or recall head)',
  '-3': 'tensor table does not match the header', '-4': 'a tensor lies outside the file', '-5': 'bad embedding tensor',
  '-6': 'bad norm vector', '-7': 'bad layer matrix', '-8': 'out of memory', '-9': 'bad final norm', '-10': 'already initialised',
};
const SKILL_ERRORS = {
  '-1': 'bad slot', '-2': 'not a skill file', '-3': 'skill file is truncated', '-4': 'bad entry in the skill file',
  '-5': 'skill does not fit this trunk', '-6': 'another loaded skill starts at a different layer',
  '-7': 'out of memory', '-8': 'the context is not empty and the split layer would move',
};
const HEAD_ERRORS = { '-1': 'not a head file', '-2': 'head file is truncated', '-3': 'head does not fit this trunk' };
const EVAL_ERRORS = { '-1': 'engine not ready or no current slot', '-2': 'token id out of range', '-3': 'context is full',
  '-4': 'out of memory' };

// The engine's numeric result code stays on the error as `code` so callers can react to one case (-8 for skills).
const fail = (table, rc, what) => { throw Object.assign(new Error(`${what}: ${table[rc] ?? `error ${rc}`}`), { code: rc }); };

export function parseSkillHeader(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 96 || view.getUint32(0, true) !== 0x31534d4b) throw new Error('not a skill file');
  const nameBytes = bytes.subarray(8, 72);
  const end = nameBytes.indexOf(0);
  return { name: new TextDecoder().decode(end < 0 ? nameBytes : nameBytes.subarray(0, end)), firstLayer: view.getUint32(72, true),
    entries: view.getUint32(76, true) };
}

export function parseHeadHeader(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 64 || view.getUint32(0, true) !== 0x31484d4b) throw new Error('not a head file');
  const at = view.getUint32(28, true), length = view.getUint32(32, true);
  if (at + length > bytes.byteLength) throw new Error('head file is truncated');
  const names = JSON.parse(new TextDecoder().decode(bytes.subarray(at, at + length))).names;
  if (!Array.isArray(names) || names.some(n => typeof n !== 'string')) throw new Error('head file has no class names');
  return { names, outputs: view.getUint32(20, true) };
}

export class MindEngine {
  // variant: one of VARIANTS (default: the best the runtime supports); threads: total threads including the caller.
  static async create({ variant = pickVariant(), threads = defaultThreads(), wasmDir = WASM_DIR } = {}) {
    if (!VARIANTS.includes(variant)) throw new Error(`unknown variant ${variant}`);
    const bytes = fs.readFileSync(path.join(wasmDir, `kiwimind-${variant}.wasm`));
    if (variant === 'simd') {
      const { instance } = await WebAssembly.instantiate(bytes, {});
      return new MindEngine(instance.exports, variant, 1, []);
    }
    const module = await WebAssembly.compile(bytes);
    const memory = new WebAssembly.Memory({ initial: 64, maximum: 65536, shared: true });
    const instance = await WebAssembly.instantiate(module, { env: { memory } });
    const engine = new MindEngine({ ...instance.exports, memory }, variant, threads, []);
    const stackBytes = 1 << 20;
    const threadUrl = new URL('./thread.mjs', import.meta.url);
    engine.helpers = await Promise.all(Array.from({ length: threads - 1 }, (_, i) => new Promise((resolve, reject) => {
      const stack = engine.alloc(stackBytes);
      const helper = new Worker(threadUrl, { workerData: { module, memory, index: i + 1, stackTop: stack + stackBytes } });
      helper.once('message', () => { helper.unref(); resolve(helper); });
      helper.once('error', reject);
    })));
    instance.exports.km_set_threads(threads);
    return engine;
  }

  constructor(exports, variant, threads, helpers) {
    this.e = exports;
    this.variant = variant;
    this.threads = threads;
    this.helpers = helpers;
    this.ready = false;
  }

  get memory() { return this.e.memory.buffer; }

  // WebAssembly returns i32; pointers above 2 GiB must be read as unsigned.
  alloc(bytes) {
    const p = this.e.km_alloc(bytes) >>> 0;
    if (!p) throw new Error('out of memory');
    return p;
  }

  // Reads a trunk file straight into engine memory and starts the engine. The bytes are repacked in place.
  loadTrunk(file, maxCtx = 4096) {
    const fd = fs.openSync(file, 'r');
    try {
      const size = fs.fstatSync(fd).size;
      if (size > 0xfffff000) throw new Error('trunk file is too large');
      const head = Buffer.alloc(200);
      if (fs.readSync(fd, head, 0, 200, 0) !== 200) throw new Error('trunk file is truncated');
      const base = this.alloc(size);
      for (let at = 0; at < size;) {
        const n = fs.readSync(fd, new Uint8Array(this.memory, base + at, Math.min(size - at, 1 << 26)), 0, Math.min(size - at, 1 << 26), at);
        if (!n) throw new Error('trunk file is truncated');
        at += n;
      }
      const rc = this.e.km_init(base, size, maxCtx);
      if (rc) fail(INIT_ERRORS, rc, 'trunk');
      this.#finishInit(new DataView(head.buffer, head.byteOffset, 200));
    } finally {
      fs.closeSync(fd);
    }
  }

  #finishInit(header) {
    const headDim = header.getUint32(36, true), theta = header.getFloat32(64, true);
    const ctx = this.e.km_max_ctx(), half = headDim / 2;
    const rope = new Float32Array(this.memory, this.e.km_rope_table() >>> 0, ctx * headDim);
    // float32 inverse frequencies and angles, as the PyTorch reference computes them
    for (let i = 0; i < half; i++) {
      const inv = Math.fround(1 / Math.pow(theta, (2 * i) / headDim));
      for (let p = 0; p < ctx; p++) {
        const angle = Math.fround(p * inv);
        rope[p * half + i] = Math.cos(angle);
        rope[ctx * half + p * half + i] = Math.sin(angle);
      }
    }
    this.vocab = this.e.km_vocab();
    this.maxCtx = ctx;
    this.dim = this.e.km_dim();
    this.nLayers = this.e.km_n_layers();
    this.skillSlots = this.e.km_skill_slots();
    this.maxSlots = this.e.km_slot_max();
    this.blockPositions = this.e.km_block_positions();
    this.scratch = this.alloc(4 * 4096);
    this.controlOut = this.alloc(4 * 256);
    this.ready = true;
  }

  #stage(bytes) {
    const p = this.e.km_staging(bytes.byteLength) >>> 0;
    if (!p) throw new Error('out of memory');
    new Uint8Array(this.memory, p, bytes.byteLength).set(bytes);
    return p;
  }

  loadSkill(slot, bytes) {
    const rc = this.e.km_skill_load(slot, this.#stage(bytes), bytes.byteLength);
    if (rc) fail(SKILL_ERRORS, rc, 'skill');
  }

  unloadSkill(slot) { if (this.e.km_skill_unload(slot)) throw new Error('skill slot is empty'); }

  // Select the skill for the upper layers of the current slot (-1 none) and rebuild its upper cache when it was
  // built under another skill. selectSkill only chooses; the next evalUpper does the work (in steps, if wanted).
  useSkill(skill) { if (this.e.km_skill_use(skill)) throw new Error('skill slot is empty'); }
  selectSkill(skill) { if (this.e.km_skill_select(skill)) throw new Error('skill slot is empty'); }

  get activeSkill() { return this.e.km_active_skill(); }

  // Conversations. A new slot is empty; the one made by loadTrunk is slot 0 and current. All evaluation, sampling,
  // logits and history calls work on the current slot.
  slotNew() { return this.e.km_slot_new(); }
  slotFree(id) { if (this.e.km_slot_free(id)) throw new Error(`slot ${id} is not in use`); }
  slotUse(id) { if (this.e.km_slot_use(id)) throw new Error(`slot ${id} is not in use`); }
  get slotCurrent() { return this.e.km_slot_current(); }
  get slotCount() { return this.e.km_slot_count(); }
  slotBytes(id) { return this.e.km_slot_bytes(id) >>> 0; }
  slotNPast(id) { return this.e.km_slot_n_past(id); }
  slotSkill(id) { return this.e.km_slot_skill(id); }

  loadHead(bytes) {
    const rc = this.e.km_head_load(this.#stage(bytes), bytes.byteLength);
    if (rc) fail(HEAD_ERRORS, rc, 'head');
  }

  get split() { return this.e.km_split(); }
  get nPast() { return this.e.km_n_past(); }
  get nUpper() { return this.e.km_n_upper(); }
  get logitsReady() { return this.e.km_logits_ready() === 1; }

  // Raw head outputs for the last evaluated position, or null when they are stale (truncated or nothing evaluated).
  control() {
    const n = this.e.km_control(this.controlOut);
    if (n === -3) return null;
    if (n < 0) throw new Error(n === -1 ? 'no control head loaded' : "the head's layer lies above the skills' first layer");
    return new Float32Array(this.memory, this.controlOut, n).slice();
  }

  #tokens(tokens, fn) {
    for (let i = 0; i < tokens.length; i += 4096) {
      const part = tokens.slice(i, i + 4096);
      new Int32Array(this.memory, this.scratch, part.length).set(part);
      const rc = fn(this.scratch, part.length);
      if (rc) fail(EVAL_ERRORS, rc, 'eval');
    }
  }

  evalLower(tokens) { this.#tokens(tokens, (p, n) => this.e.km_eval_lower(p, n)); }
  // Upper layers for at most `max` positions; returns how many positions still wait (0 = the logits are ready).
  evalUpper(max = 0x7fffffff) {
    const rc = this.e.km_eval_upper_n(max);
    if (rc < 0) fail(EVAL_ERRORS, rc, 'eval');
    return rc;
  }
  eval(tokens) { this.#tokens(tokens, (p, n) => this.e.km_eval(p, n)); }

  truncate(n) { return this.e.km_truncate(n); }
  reset() { this.e.km_reset(); }

  // A live view of the current slot's logits for its last position (valid until its next evaluation).
  get logits() { return new Float32Array(this.memory, this.e.km_logits() >>> 0, this.vocab); }

  // Added to the logits before sampling; belongs to the current slot and stays until changed (zeros at first).
  get bias() { return new Float32Array(this.memory, this.e.km_bias() >>> 0, this.vocab); }

  // The hidden state at the input of the split layer for every cached position of the current slot ([nPast * dim]).
  get states() {
    const out = new Float32Array(this.nPast * this.dim);
    for (let p = 0; p < this.nPast; p++) out.set(new Float32Array(this.memory, this.e.km_state_at(p) >>> 0, this.dim), p * this.dim);
    return out;
  }

  history(n = this.nPast) { return Array.from({ length: n }, (_, p) => this.e.km_token_at(p)); }

  sample({ temperature = 0, topK = 40, topP = 1, minP = 0, repeatPenalty = 1, window = 256 } = {}) {
    return this.e.km_sample(temperature, topK, topP, minP, repeatPenalty, window);
  }

  argmax() { return this.e.km_argmax(); }
  seed(value) { this.e.km_seed(value >>> 0); }

  // Test hook: out[t][r] = W[r] . x[t] through the production kernels. `blob` is a tensor as the file stores it.
  matmul(type, rows, cols, T, blob, x) {
    const w = this.alloc(blob.byteLength), xp = this.alloc(x.byteLength), out = this.alloc(rows * T * 4);
    new Uint8Array(this.memory, w, blob.byteLength).set(blob);
    new Float32Array(this.memory, xp, x.length).set(x);
    const rc = this.e.km_test_matmul(type, rows, cols, T, w, xp, out);
    if (rc) throw new Error(`matmul hook: error ${rc}`);
    return new Float32Array(this.memory, out, rows * T).slice();
  }

  close() { for (const helper of this.helpers) helper.terminate(); this.helpers = []; }
}
