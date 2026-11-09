/**
 * Kiwi Sense in the browser: the same WebAssembly core and data file the server reads
 * replies with (static/sense/sense.wasm + sense.bin). No DOM in here, so Node can run
 * it too (tests/sense_browser.mjs checks it against the Python reader).
 */

const MAX_CHARS = 4000;
const CJK_RANGES = [
  [0x4e00, 0x9fff], [0x3400, 0x4dbf], [0x20000, 0x2a6df], [0x2a700, 0x2b73f],
  [0x2b740, 0x2b81f], [0x2b820, 0x2ceaf], [0xf900, 0xfaff], [0x2f800, 0x2fa1f],
];
const MARK = /^\p{Mn}$/u;
const SPACE = /^\p{Zs}$/u;
const CONTROL = /^\p{C}$/u;
const PUNCT = /^\p{P}$/u;

function asciiPunct(code) {
  return (code >= 33 && code <= 47) || (code >= 58 && code <= 64) || (code >= 91 && code <= 96) || (code >= 123 && code <= 126);
}

/** Lower case, accents removed, every word and punctuation mark separated by one space. */
export function normalize(text) {
  const head = Array.from(String(text ?? "")).slice(0, MAX_CHARS).join("");
  let out = "";
  for (const ch of head.toLowerCase().normalize("NFD")) {
    if (MARK.test(ch)) continue;
    if (ch === "\t" || ch === "\n" || ch === "\r" || SPACE.test(ch)) {
      out += " ";
      continue;
    }
    if (CONTROL.test(ch)) continue;
    const code = ch.codePointAt(0);
    if (asciiPunct(code) || PUNCT.test(ch) || CJK_RANGES.some(([lo, hi]) => code >= lo && code <= hi)) out += ` ${ch} `;
    else out += ch;
  }
  return out.split(/\s+/).filter(Boolean).join(" ");
}

function softmax(scores) {
  const top = Math.max(...scores);
  const raised = scores.map((score) => Math.exp(score - top));
  const total = raised.reduce((sum, value) => sum + value, 0);
  return raised.map((value) => value / total);
}

// The data file's table of contents: "KSNS", version, count, then (tag, offset, length) rows.
function metaOf(blob) {
  const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const tag = (at) => String.fromCharCode(blob[at], blob[at + 1], blob[at + 2], blob[at + 3]);
  if (blob.length < 12 || tag(0) !== "KSNS") throw new Error("not a Kiwi Sense data file");
  const count = view.getUint32(8, true);
  for (let index = 0; index < count; index += 1) {
    const at = 12 + 12 * index;
    if (tag(at) !== "META") continue;
    const offset = view.getUint32(at + 4, true);
    const length = view.getUint32(at + 8, true);
    return JSON.parse(new TextDecoder().decode(blob.subarray(offset, offset + length)));
  }
  return {};
}

/** A reader over the compiled core. `wasm` and `blob` are the two files' bytes. */
export async function createSense(wasm, blob) {
  const data = blob instanceof Uint8Array ? blob : new Uint8Array(blob);
  const meta = metaOf(data);
  const { instance } = await WebAssembly.instantiate(wasm, {});
  const ks = instance.exports;
  const encoder = new TextEncoder();

  const put = (bytes) => {
    const at = ks.ks_alloc(Math.max(1, bytes.length));
    if (!at) throw new Error("Kiwi Sense ran out of memory");
    new Uint8Array(ks.memory.buffer, at, bytes.length).set(bytes);
    return at;
  };
  const doubles = (at, count) => Array.from(new Float64Array(ks.memory.buffer.slice(at, at + 8 * count)));

  const status = ks.ks_init(put(data), data.length);
  if (status !== 0) throw new Error(`Kiwi Sense could not read its data file (${status})`);
  const dims = ks.ks_dims();
  const classes = ks.ks_classes();
  const labels = Array.isArray(meta.classes) ? meta.classes : [];

  return {
    meta,
    labels,
    dims,
    threshold: Number(meta.probe?.threshold) || 0.8,

    /** { label: probability } for one sentence; {} when there is nothing to read. */
    probe(sentence) {
      const bytes = encoder.encode(normalize(sentence));
      if (!bytes.length || !labels.length) return {};
      const at = put(bytes);
      const result = ks.ks_alloc(8 * classes);
      ks.ks_probe(at, bytes.length, result);
      const scores = softmax(doubles(result, classes));
      ks.ks_reset();
      return Object.fromEntries(labels.map((label, index) => [label, scores[index]]));
    },

    /** { vector, coverage } (unit vector, share of words known) or null when no word is known. */
    embed(text) {
      const bytes = encoder.encode(normalize(text));
      const at = put(bytes);
      const result = ks.ks_alloc(8 * dims);
      const known = ks.ks_embed(at, bytes.length, result);
      const words = ks.ks_words();
      const vector = doubles(result, dims);
      ks.ks_reset();
      return known ? { vector, coverage: known / words } : null;
    },

    /** Cosine of two meanings (-1..1); null when one of them cannot be read. */
    similarity(a, b) {
      const first = this.embed(a);
      const second = this.embed(b);
      if (!first || !second || Math.min(first.coverage, second.coverage) < 0.5) return null;
      return first.vector.reduce((sum, value, index) => sum + value * second.vector[index], 0);
    },
  };
}

let shared = null;

/** The page's reader, fetched and compiled on first use. */
export function loadSense(base = "/static/sense") {
  if (!shared) {
    shared = Promise.all([
      fetch(`${base}/sense.wasm`).then((response) => response.arrayBuffer()),
      fetch(`${base}/sense.bin`).then((response) => response.arrayBuffer()),
    ])
      .then(([wasm, blob]) => createSense(wasm, blob))
      .catch((error) => {
        shared = null;
        throw error;
      });
  }
  return shared;
}
