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
process.stdout.write(JSON.stringify(out));
