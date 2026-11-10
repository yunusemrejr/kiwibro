// Runs the browser's Kiwi Sense reader (static/js/sense.js) over raw texts and prints what
// it computed as JSON. tests/test_sense.py compares this with the Python reader.
//
//   node tests/sense_browser.mjs <sense.js> <sense.wasm> <sense.bin> <texts.json>
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [modulePath, wasmPath, blobPath, textsPath] = process.argv.slice(2);
const { createSense, normalize } = await import(pathToFileURL(modulePath).href);
const reader = await createSense(readFileSync(wasmPath), new Uint8Array(readFileSync(blobPath)));
const texts = JSON.parse(readFileSync(textsPath, "utf8"));

const out = { labels: reader.labels, threshold: reader.threshold, texts: [] };
for (const text of texts) {
  const embedded = reader.embed(text);
  out.texts.push({
    normalized: normalize(text),
    probe: reader.probe(text),
    vector: embedded ? embedded.vector : null,
    similarity: reader.similarity(text, texts[0]),
  });
}
process.stdout.write(JSON.stringify(out));
