// Runs the Kiwi Sense WebAssembly core over a list of (already normalised) texts and
// prints what it computed as JSON. tests/test_sense.py compares this, number for number,
// with the Python twin.
//
//   node tests/sense_wasm.mjs <sense.wasm> <sense.bin> <cases.json>
import { readFileSync } from "node:fs";

const [wasmPath, blobPath, casesPath] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const ks = instance.exports;
const encoder = new TextEncoder();

function put(bytes) {
  const at = ks.ks_alloc(Math.max(1, bytes.length));
  if (!at) throw new Error("out of memory");
  new Uint8Array(ks.memory.buffer, at, bytes.length).set(bytes);
  return at;
}

const blob = readFileSync(blobPath);
const status = ks.ks_init(put(blob), blob.length);
if (status !== 0) throw new Error(`ks_init failed: ${status}`);
const dims = ks.ks_dims();
const classes = ks.ks_classes();

function doubles(at, count) {
  return Array.from(new Float64Array(ks.memory.buffer.slice(at, at + 8 * count)));
}

const cases = JSON.parse(readFileSync(casesPath, "utf8"));
const out = { dims, classes, texts: [], pairs: [] };
for (const text of cases.texts) {
  const bytes = encoder.encode(text);
  let at = put(bytes);
  let result = ks.ks_alloc(8 * dims);
  const known = ks.ks_embed(at, bytes.length, result);
  const words = ks.ks_words();
  const vector = doubles(result, dims);
  ks.ks_reset();
  at = put(bytes);
  result = ks.ks_alloc(8 * classes);
  ks.ks_probe(at, bytes.length, result);
  const scores = doubles(result, classes);
  ks.ks_reset();
  at = put(bytes);
  const repeat = ks.ks_repeat(at, bytes.length);
  ks.ks_reset();
  out.texts.push({ known, words, vector, scores, repeat });
}
for (const [a, b] of cases.pairs) {
  const first = encoder.encode(a);
  const second = encoder.encode(b);
  let pa = put(first);
  let pb = put(second);
  const similarity = ks.ks_similarity(pa, first.length, pb, second.length);
  ks.ks_reset();
  pa = put(first);
  pb = put(second);
  const lexsim = ks.ks_lexsim(pa, first.length, pb, second.length);
  ks.ks_reset();
  out.pairs.push({ similarity, lexsim });
}
// the helpers: every head over every text (one-text and per-word heads) or pair
out.heads = [];
const tagName = (v) => String.fromCharCode(v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255);
for (let i = 0; i < ks.ks_heads(); i++) {
  const kind = ks.ks_head_kind(i);
  const width = ks.ks_head_classes(i);
  const results = [];
  if (kind === 2) {
    for (const [a, b] of cases.pairs) {
      const first = encoder.encode(a);
      const second = encoder.encode(b);
      const pa = put(first);
      const pb = put(second);
      const result = ks.ks_alloc(8 * width);
      ks.ks_head_pair(i, pa, first.length, pb, second.length, result);
      results.push(doubles(result, width));
      ks.ks_reset();
    }
  } else {
    for (const text of cases.texts.concat(cases.raw || [])) {
      const bytes = encoder.encode(text);
      const at = put(bytes);
      const result = ks.ks_alloc(8 * width * 128);
      if (kind === 1) {
        ks.ks_head_text(i, at, bytes.length, result);
        results.push(doubles(result, width));
      } else {
        const count = ks.ks_head_tokens(i, at, bytes.length, result);
        const flat = doubles(result, width * count);
        const rows = [];
        for (let k = 0; k < count; k++) rows.push(flat.slice(k * width, (k + 1) * width));
        results.push(rows);
      }
      ks.ks_reset();
    }
  }
  out.heads.push({ tag: tagName(ks.ks_head_tag(i)), kind, results });
}
process.stdout.write(JSON.stringify(out));
