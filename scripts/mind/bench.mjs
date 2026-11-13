// Speed and memory of the Kiwi Mind engine on a real trunk file.
//
//   node bench.mjs --model X.fkm [--threads 6] [--prompt-tokens 512] [--decode 64] [--skill f.kms] [--runs 3]
//                  [--ctx 4096] [--variant relaxed-mt|simd-mt|simd]
//   node bench.mjs --model X.fkm --threads 6 --slots 3 --fill 2048 [--skill f.kms]     memory of several slots
//
// Prints prefill and decode speed, the time a skill switch takes for the prompt, and the peak resident memory of
// this process. The tokens are random ids: the weights of a test trunk mean nothing, but the cost of a token does
// not depend on what it is.
import { performance } from 'node:perf_hooks';
import fs from 'node:fs';
import { MindEngine, pickVariant } from '../../kiwibro/mind/host/engine.mjs';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  if (!process.argv[i].startsWith('--') || process.argv[i + 1] === undefined) {
    console.error('usage: node bench.mjs --model X.fkm [--threads 6] [--prompt-tokens 512] [--decode 64] [--skill f.kms] [--runs 3] [--ctx 4096] [--slots N --fill T]');
    process.exit(2);
  }
  args.set(process.argv[i].slice(2), process.argv[i + 1]);
}
const model = args.get('model');
if (!model) { console.error('--model is required'); process.exit(2); }
const threads = Number(args.get('threads') ?? 6);
const promptTokens = Number(args.get('prompt-tokens') ?? 512);
const decodeTokens = Number(args.get('decode') ?? 64);
const runs = Number(args.get('runs') ?? 3);
const ctx = Number(args.get('ctx') ?? 4096);
const slots = Number(args.get('slots') ?? 1);
const fill = args.has('fill') ? Number(args.get('fill')) : null;
const variant = args.get('variant') ?? pickVariant();

const mib = bytes => (bytes / 1048576).toFixed(0);
const rss = () => process.memoryUsage().rss;
const peakRss = () => process.resourceUsage().maxRSS * 1024; // kilobytes -> bytes
const ms = t => `${t.toFixed(0)} ms`;
const median = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];

function randomTokens(n, vocab, seed) {
  let x = seed >>> 0;
  return Array.from({ length: n }, () => { x = (Math.imul(x, 1664525) + 1013904223) >>> 0; return (x >>> 8) % vocab; });
}

const t0 = performance.now();
const engine = await MindEngine.create({ variant, threads });
engine.loadTrunk(model, ctx);
const loadMs = performance.now() - t0;
const skill = args.get('skill');
if (skill) engine.loadSkill(0, fs.readFileSync(skill));
console.log(`model ${model}  ${(fs.statSync(model).size / 1048576).toFixed(0)} MiB file, ${engine.nLayers} layers, split at layer ${engine.split}`);
console.log(`engine ${engine.variant}, ${threads} threads, context ${engine.maxCtx}; trunk loaded in ${ms(loadMs)}; resident memory ${mib(rss())} MiB`);

if (fill !== null) {
  // several conversations at once: N slots with T tokens each, then a round of decoding in every slot
  const ids = Array.from({ length: slots }, (_, i) => (i === 0 ? 0 : engine.slotNew()));
  const start = performance.now();
  for (const [i, id] of ids.entries()) {
    engine.slotUse(id);
    if (skill) engine.selectSkill(0);
    engine.eval(randomTokens(fill, engine.vocab, 100 + i));
  }
  const prefillMs = performance.now() - start;
  const d0 = performance.now();
  for (let step = 0; step < 16; step++) {
    for (const id of ids) {
      engine.slotUse(id);
      engine.eval([engine.sample({ temperature: 0 })]);
    }
  }
  const decodeMs = performance.now() - d0;
  console.log(`${slots} slot(s) of ${fill} tokens: filled in ${ms(prefillMs)}; 16 decode rounds over all slots ${ms(decodeMs)} (${(16 * slots / (decodeMs / 1000)).toFixed(1)} tokens/s in total)`);
  console.log(`slot memory: ${ids.map(id => `${mib(engine.slotBytes(id))} MiB`).join(', ')}`);
  console.log(`wasm memory ${mib(engine.memory.byteLength)} MiB; resident now ${mib(rss())} MiB; peak resident ${mib(peakRss())} MiB`);
  engine.close();
  process.exit(0);
}

const prompt = randomTokens(promptTokens, engine.vocab, 1);
engine.eval(prompt.slice(0, 64)); // wake the threads and touch the weights once
engine.reset();

const speeds = [];
let lowerMs = 0, upperMs = 0;
for (let run = 0; run < runs; run++) {
  engine.reset();
  const a = performance.now();
  engine.evalLower(prompt);
  const b = performance.now();
  engine.evalUpper();
  const c = performance.now();
  speeds.push(promptTokens / ((c - a) / 1000));
  lowerMs = b - a;
  upperMs = c - b;
}
console.log(`prefill  ${median(speeds).toFixed(1)} tokens/s for ${promptTokens} tokens (runs: ${speeds.map(s => s.toFixed(1)).join(', ')}); last run: layers below the split ${ms(lowerMs)}, above ${ms(upperMs)}`);

const d0 = performance.now();
for (let i = 0; i < decodeTokens; i++) engine.eval([engine.sample({ temperature: 0 })]);
const decodeMs = performance.now() - d0;
console.log(`decode   ${(decodeTokens / (decodeMs / 1000)).toFixed(1)} tokens/s (${decodeTokens} tokens after the ${promptTokens}-token prompt, ${(decodeMs / decodeTokens).toFixed(1)} ms each)`);

if (skill) {
  engine.reset();
  engine.eval(prompt);
  const s0 = performance.now();
  engine.useSkill(0);
  const toSkill = performance.now() - s0;
  const s1 = performance.now();
  engine.useSkill(-1);
  const toNone = performance.now() - s1;
  console.log(`skill switch for a ${promptTokens}-token context: none -> skill ${ms(toSkill)}, skill -> none ${ms(toNone)} (upper layers recomputed from the stored split states)`);
}
console.log(`peak resident memory ${mib(peakRss())} MiB; wasm memory ${mib(engine.memory.byteLength)} MiB`);
engine.close();
process.exit(0);
