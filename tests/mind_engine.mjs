// Kiwi Mind engine, tokenizer and host, tested on the tiny random model of scripts/mind/formats.py.
//
//   node tests/mind_engine.mjs          exit code 0 when every check passed, 1 otherwise; one line per check
//
// Needs: node, python3 with numpy (the reference forward pass and the file writers are Python). Nothing here
// imports the kiwibro package.
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { MindEngine, VARIANTS, wasmFeatures } from '../kiwibro/mind/host/engine.mjs';
import { BpeTokenizer } from '../kiwibro/mind/host/tokenizer.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const HOST = path.join(ROOT, 'kiwibro/mind/host/host.mjs');
const SCRIPTS = path.join(ROOT, 'scripts/mind');
const KEY = 'test-secret-key';
let failed = 0;

function check(name, ok, detail = '') {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? `  (${detail})` : ''}`);
  if (!ok) failed++;
}

async function group(name, fn) {
  try {
    await fn();
  } catch (e) {
    check(`${name}: raised an error`, false, e.stack ?? String(e));
  }
}

function python(args) {
  const r = spawnSync('python3', ['-I', ...args], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`python3 ${args.join(' ')}\n${r.stderr}`);
  return r.stdout;
}

const range = (from, to) => Array.from({ length: to - from + 1 }, (_, i) => from + i);
const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const peak = a => Math.max(...Array.from(a, Math.abs));
const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

// A fixed pseudo-random token sequence (ids below the three special tokens of the tiny vocabulary).
function tokensOf(n, seed) {
  let x = seed >>> 0;
  return Array.from({ length: n }, () => { x = (Math.imul(x, 1664525) + 1013904223) >>> 0; return (x >>> 8) % 290; });
}

// ------------------------------------------------------------------ files

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiwimind-test-'));
process.on('exit', () => fs.rmSync(dir, { recursive: true, force: true }));
python([path.join(SCRIPTS, 'formats.py'), 'tiny', dir]);
const file = name => path.join(dir, name);
const SPLIT = 2; // first layer of the tiny skills

async function makeEngine({ variant = 'simd', threads = 1, ctx = 512, head = 'head.kmh', skills = true } = {}) {
  const e = await MindEngine.create({ variant, threads });
  e.loadTrunk(file('tiny.fkm'), ctx);
  if (skills) {
    e.loadSkill(0, fs.readFileSync(file('alpha.kms')));
    e.loadSkill(1, fs.readFileSync(file('beta.kms')));
  }
  if (head) e.loadHead(fs.readFileSync(file(head)));
  return e;
}

const variants = VARIANTS.filter(v => {
  const f = wasmFeatures();
  return f.simd && (v === 'simd' || (f.threads && (v === 'simd-mt' || f.relaxed)));
});

// ------------------------------------------------------------------ 1. engine against the NumPy reference

await group('reference', async () => {
  // Tolerance. The reference repeats the engine's three roundings (int8 activations, int8 KV cache, int16 query), so
  // what is left is float32 summation order, the engine's polynomial exp against numpy's, and the rare activation
  // that lands next to a rounding boundary and rounds the other way. That moves a logit by a few thousandths of
  // the logit range; a wrong kernel, a wrong layer or a wrong cache position moves it by whole units.
  const TOL = 0.02;       // largest |engine - reference| over all logits, relative to the largest |reference logit|
  const TOL_STATE = 0.02; // the same for the split-layer hidden states
  const long = tokensOf(150, 7); // longer than one memory block (128 positions) of a slot
  const cases = [
    { name: 'no skill, 8 tokens', tokens: [297, 298, 5, 6, 7, 8, 9, 10], skill: null, head: 'head.kmh', split: SPLIT },
    { name: 'skill alpha, 8 tokens', tokens: [297, 298, 5, 6, 7, 8, 9, 10], skill: 'alpha', head: null },
    { name: 'skill beta, 8 tokens', tokens: [297, 298, 5, 6, 7, 8, 9, 10], skill: 'beta', head: 'head.kmh' },
    { name: 'skill alpha, 150 tokens (two memory blocks)', tokens: long, skill: 'alpha', head: 'head.kmh' },
    { name: 'no skill, 70 tokens, head on layer 1', tokens: tokensOf(70, 3), skill: null, head: 'head_l1.kmh', split: SPLIT },
  ];
  const spec = {
    model: 'tiny.fkm',
    cases: cases.map(c => ({
      tokens: c.tokens, skill: c.skill && `${c.skill}.kms`, head: c.head, state_layers: [SPLIT], ...(c.split ? { split: c.split } : {}),
    })),
  };
  for (const emulate of [true, false]) {
    fs.writeFileSync(file('cases.json'), JSON.stringify({ ...spec, emulate }));
    python([path.join(SCRIPTS, 'reference.py'), dir, file('cases.json'), file(emulate ? 'ref.json' : 'exact.json')]);
  }
  const ref = JSON.parse(fs.readFileSync(file('ref.json'))).cases;
  const exact = JSON.parse(fs.readFileSync(file('exact.json'))).cases;

  for (const variant of variants) {
    const threads = variant === 'simd' ? 1 : 3;
    const e = await makeEngine({ variant, threads, head: null });
    for (const [i, c] of cases.entries()) {
      e.slotUse(e.slotNew());
      if (c.head) e.loadHead(fs.readFileSync(file(c.head)));
      e.selectSkill(c.skill === null ? -1 : c.skill === 'alpha' ? 0 : 1);
      e.eval(c.tokens);
      const logits = Array.from(e.logits);
      const limit = TOL * peak(ref[i].logits);
      const err = maxAbs(logits, ref[i].logits), errExact = maxAbs(logits, exact[i].logits);
      check(`logits ${variant}/${threads}t, ${c.name}`, err <= limit,
        `max diff ${err.toExponential(2)}, limit ${limit.toExponential(2)}; against exact float ${errExact.toExponential(2)}`);
      const states = Array.from(e.states);
      const refStates = ref[i].states[String(SPLIT)].flat();
      check(`split-layer states ${variant}/${threads}t, ${c.name}`, states.length === refStates.length &&
        maxAbs(states, refStates) <= TOL_STATE * peak(refStates), `max diff ${maxAbs(states, refStates).toExponential(2)}`);
      if (c.head) {
        const control = Array.from(e.control());
        const diff = maxAbs(control, ref[i].control);
        check(`control head ${variant}/${threads}t, ${c.name}`, diff <= 1e-3 * Math.max(1, peak(ref[i].control)) + 0.01,
          `max diff ${diff.toExponential(2)}`);
      }
      e.slotFree(e.slotCurrent);
    }
    e.close();
  }
});

// ------------------------------------------------------------------ 2. truncate, split evaluation, skill switch

await group('cache', async () => {
  const toks = tokensOf(150, 11);
  for (const variant of variants) {
    const threads = variant === 'simd' ? 1 : 3;
    const e = await makeEngine({ variant, threads });
    const tag = `${variant}/${threads}t`;
    e.selectSkill(0);
    e.eval(toks);
    const whole = Array.from(e.logits);

    // Rows are computed independently of how tokens are grouped, so the results must be bit for bit the same.
    e.reset();
    e.eval(toks.slice(0, 100));
    check(`truncate returns the kept length ${tag}`, e.truncate(40) === 40 && e.nPast === 40);
    e.eval(toks.slice(40));
    check(`eval after truncate equals one pass ${tag}`, e.nPast === 150 && same(Array.from(e.logits), whole));

    e.reset();
    e.truncate(5);
    check(`truncate on an empty slot keeps nothing ${tag}`, e.nPast === 0);
    e.eval(toks.slice(0, 20));
    e.truncate(20);
    check(`truncate beyond the length changes nothing ${tag}`, e.nPast === 20);

    e.reset();
    e.evalLower(toks);
    check(`evalLower alone leaves the upper layers undone ${tag}`, e.nPast === 150 && e.nUpper === 0 && !e.logitsReady);
    e.evalUpper();
    check(`evalLower + evalUpper equals eval ${tag}`, e.logitsReady && same(Array.from(e.logits), whole));

    e.reset();
    e.evalLower(toks);
    let steps = 0, left;
    do { left = e.evalUpper(37); steps++; } while (left > 0);
    check(`evalUpper in pieces of 37 equals eval ${tag}`, steps === 5 && same(Array.from(e.logits), whole), `${steps} calls`);

    // Skill switch: the upper layers are recomputed from the stored split states.
    const fresh = async skill => {
      const f = await makeEngine({ variant, threads });
      f.selectSkill(skill);
      f.eval(toks);
      const out = Array.from(f.logits);
      f.close();
      return out;
    };
    e.reset();
    e.selectSkill(0);
    e.eval(toks);
    e.useSkill(1);
    check(`switch alpha -> beta equals a fresh pass with beta ${tag}`, same(Array.from(e.logits), await fresh(1)));
    e.useSkill(-1);
    check(`switch beta -> none equals a fresh pass without skill ${tag}`, same(Array.from(e.logits), await fresh(-1)));
    e.useSkill(0);
    check(`switch none -> alpha returns to the first result ${tag}`, same(Array.from(e.logits), whole));
    // Appending after a switch keeps working on the new skill.
    e.useSkill(1);
    e.eval([5, 6, 7]);
    const g = await makeEngine({ variant, threads });
    g.selectSkill(1);
    g.eval([...toks, 5, 6, 7]);
    check(`appending after a switch equals a fresh pass ${tag}`, same(Array.from(e.logits), Array.from(g.logits)));
    g.close();
    e.close();
  }

  // Different variants and thread counts agree, so a result does not depend on the machine's features.
  const outs = [];
  for (const [variant, threads] of [['simd', 1], ['simd-mt', 4], ['relaxed-mt', 2], ['relaxed-mt', 5]]) {
    if (!variants.includes(variant)) continue;
    const e = await makeEngine({ variant, threads });
    e.selectSkill(1);
    e.eval(toks);
    outs.push([`${variant}/${threads}t`, Array.from(e.logits)]);
    e.close();
  }
  const base = outs[0][1];
  check('logits agree across variants and thread counts', outs.every(([, o]) => maxAbs(o, base) <= 1e-3 * peak(base)),
    outs.map(([n, o]) => `${n} ${maxAbs(o, base).toExponential(1)}`).join(', '));

  // Errors are reported, not hidden.
  const e = await makeEngine({ variant: 'simd' });
  let message = '';
  try { e.eval([5, 300]); } catch (x) { message = x.message; }
  check('a token id outside the vocabulary is refused', /out of range/.test(message), message);
  message = '';
  try { e.eval(tokensOf(513, 1)); } catch (x) { message = x.message; }
  check('a context beyond its size is refused', /full/.test(message), message);
  e.close();
});

// ------------------------------------------------------------------ 3. slots

function greedy(e, n) {
  const ids = [];
  for (let i = 0; i < n; i++) { const t = e.sample({ temperature: 0 }); ids.push(t); e.eval([t]); }
  return ids;
}

await group('slots', async () => {
  const promptA = tokensOf(140, 21), promptB = tokensOf(61, 22), N = 40;
  for (const variant of variants) {
    const threads = variant === 'simd' ? 1 : 3;
    const tag = `${variant}/${threads}t`;
    // each conversation alone, on a fresh engine
    const alone = async (prompt, skill) => {
      const e = await makeEngine({ variant, threads });
      e.selectSkill(skill);
      e.eval(prompt);
      const ids = greedy(e, N), logits = Array.from(e.logits);
      e.close();
      return { ids, logits };
    };
    const soloA = await alone(promptA, 0), soloB = await alone(promptB, -1);

    // the same two, in one engine, prefill and decode taking turns in small pieces
    const e = await makeEngine({ variant, threads });
    const a = 0, b = e.slotNew();
    check(`a second slot is created ${tag}`, b === 1 && e.slotCount === 2);
    const slot = (id, fn) => { e.slotUse(id); return fn(); };
    slot(a, () => e.selectSkill(0));
    slot(b, () => e.selectSkill(-1));
    for (let i = 0; i < 140; i += 16) {
      slot(a, () => e.eval(promptA.slice(i, i + 16)));
      if (i < 61) slot(b, () => e.eval(promptB.slice(i, i + 16)));
    }
    const idsA = [], idsB = [];
    for (let i = 0; i < N; i++) {
      slot(a, () => { const t = e.sample({ temperature: 0 }); idsA.push(t); e.eval([t]); });
      slot(b, () => { const t = e.sample({ temperature: 0 }); idsB.push(t); e.eval([t]); });
    }
    check(`two greedy generations interleaved equal each alone ${tag}`, same(idsA, soloA.ids) && same(idsB, soloB.ids),
      `${N} tokens each, prompts of ${promptA.length} and ${promptB.length}`);
    check(`final logits of both slots are bit for bit the solo ones ${tag}`,
      same(slot(a, () => Array.from(e.logits)), soloA.logits) && same(slot(b, () => Array.from(e.logits)), soloB.logits));
    check(`the slots keep their own length and skill ${tag}`,
      e.slotNPast(a) === 140 + N && e.slotNPast(b) === 61 + N && e.slotSkill(a) === 0 && e.slotSkill(b) === -1);

    // a skill switch in one slot leaves the other alone
    const nUpperB = slot(b, () => e.nUpper), logitsB = slot(b, () => Array.from(e.logits));
    slot(a, () => e.useSkill(1));
    check(`a skill switch in one slot does not touch the other ${tag}`,
      slot(b, () => e.nUpper) === nUpperB && same(slot(b, () => Array.from(e.logits)), logitsB) && e.slotSkill(b) === -1 && e.slotSkill(a) === 1);
    const idsB2 = slot(b, () => greedy(e, 10));
    const longB = await (async () => {
      const f = await makeEngine({ variant, threads });
      f.eval(promptB);
      const ids = greedy(f, N + 10);
      f.close();
      return ids;
    })();
    check(`the other slot goes on as if nothing happened ${tag}`, same(idsB2, longB.slice(N, N + 10)));
    // and the switched slot now holds exactly what a fresh pass over the same tokens under the new skill holds
    const fresh1 = await makeEngine({ variant, threads });
    fresh1.selectSkill(1);
    fresh1.eval([...promptA, ...idsA]);
    check(`the switched slot equals a fresh pass under the new skill ${tag}`, same(slot(a, () => Array.from(e.logits)), Array.from(fresh1.logits)));
    fresh1.close();
    e.close();
  }

  // memory: a slot costs what its context uses, and replacing slots grows nothing
  const e = await makeEngine({ variant: 'simd' });
  const block = e.blockPositions;
  const first = e.slotNew();
  e.slotUse(first);
  e.eval(tokensOf(10, 1));
  const small = e.slotBytes(first);
  e.eval(tokensOf(2 * block + 10, 2));
  const large = e.slotBytes(first);
  const blockBytes = e.e.km_block_bytes();
  check('slot memory grows with the context in whole blocks', large - small === 2 * blockBytes,
    `${small} bytes at 10 tokens, ${large} at ${2 * block + 20}; one block of ${block} positions is ${blockBytes} bytes`);
  e.truncate(10);
  check('truncating gives the blocks back', e.slotBytes(first) === small);
  e.slotFree(first);
  e.slotUse(0);
  const sizes = [];
  for (let round = 0; round < 6; round++) {
    const id = e.slotNew();
    e.slotUse(id);
    e.eval(tokensOf(300, round));
    e.slotFree(id);
    e.slotUse(0);
    sizes.push(e.memory.byteLength);
  }
  check('freeing and replacing slots does not grow wasm memory', sizes.every(s => s === sizes[0]), `memory after each round: ${sizes.join(', ')}`);
  const many = Array.from({ length: e.maxSlots - e.slotCount }, () => e.slotNew());
  check('slots run out at the engine maximum', many.every(id => id >= 0) && e.slotNew() === -1, `${e.maxSlots} slots`);
  let refused = '';
  try { e.slotFree(many[0]); e.slotUse(many[0]); } catch (x) { refused = x.message; }
  check('a freed slot cannot be used', /not in use/.test(refused), refused);
  e.close();
});

// ------------------------------------------------------------------ 4. tokenizer

await group('tokenizer', async () => {
  const json = JSON.parse(fs.readFileSync(file('tokenizer.json'), 'utf8'));
  const tok = BpeTokenizer.fromJSON(json);
  const texts = ['hello world!', 'the quick brown fox', '<|im_start|>hello world!<|im_end|>', 'def add(a, b): return a + b',
    '  x  \n\n y', 'https://example.com/path?q=1 is the answer', "it's 42 o'clock", '{"name": "kiwi", "value": 42}'];
  fs.writeFileSync(file('texts.json'), JSON.stringify(texts));
  const script = [
    'import json, sys', `sys.path.insert(0, ${JSON.stringify(SCRIPTS)})`, 'import formats',
    `tok = json.load(open(${JSON.stringify(file('tokenizer.json'))}))`,
    `texts = json.load(open(${JSON.stringify(file('texts.json'))}))`,
    'print(json.dumps([formats.tiny_encode(tok, t) for t in texts]))'].join('\n');
  const expected = JSON.parse(python(['-c', script]));
  texts.forEach((t, i) => {
    const got = tok.encode(t);
    check(`tokenizer ids for ${JSON.stringify(t)}`, same(got, expected[i]), `${got.length} ids`);
    check(`tokenizer round trip for ${JSON.stringify(t)}`, tok.decode(got) === t);
  });
  // text is never split inside a multi-byte character when it is decoded piece by piece
  const text = 'türkçe İstanbul 😀 ✓';
  const ids = tok.encode(text), decoder = tok.createStreamDecoder();
  const pieces = ids.map(id => decoder.push(id));
  check('streamed decoding gives whole characters only', pieces.join('') + decoder.flush() === text && pieces.every(p => !p.includes('�')),
    `${ids.length} tokens for ${[...text].length} characters`);

  // the real Qwen3 file, when given (a download, not part of the repository)
  const qwen = process.env.KIWIMIND_QWEN_TOKENIZER;
  if (qwen && fs.existsSync(qwen)) {
    const q = BpeTokenizer.fromJSON(JSON.parse(fs.readFileSync(qwen, 'utf8')));
    check('qwen3 tokenizer: "Hello, world!" is [9707, 11, 1879, 0]', same(q.encode('Hello, world!'), [9707, 11, 1879, 0]));
    check('qwen3 tokenizer: chat markers are single ids', same(q.encode('<|im_start|><|im_end|><|endoftext|>'), [151644, 151645, 151643]));
    const mixed = 'def f(x):\n    return {"a": [1, 2.5]}  # türkçe 😀';
    check('qwen3 tokenizer: code and unicode round trip', q.decode(q.encode(mixed)) === mixed);
  } else {
    console.log('skip qwen3 tokenizer checks (set KIWIMIND_QWEN_TOKENIZER to a tokenizer.json)');
  }
});

// ------------------------------------------------------------------ 5. the host over HTTP

function startHost(extra = [], { slots = 3, threads = 2 } = {}) {
  const child = spawn('node', [HOST, '--model', file('tiny.fkm'), '--tokenizer', file('tokenizer.json'),
    '--skill', `alpha=${file('alpha.kms')}`, '--skill', `beta=${file('beta.kms')}`, '--head', file('head.kmh'),
    '--threads', String(threads), '--ctx', '512', '--slots', String(slots), '--port', '0', ...extra],
  { env: { ...process.env, KIWIMIND_KEY: KEY }, stdio: ['pipe', 'pipe', 'pipe'] });
  let stderr = '', stdout = '';
  child.stderr.on('data', d => { stderr += d; });
  child.stdout.on('data', d => { stdout += d; });
  const exited = new Promise(resolve => child.once('exit', (code, signal) => resolve({ code, signal, at: Date.now() })));
  const ready = new Promise((resolve, reject) => {
    const timer = setInterval(() => { if (stdout.includes('\n')) { clearInterval(timer); resolve(stdout.slice(0, stdout.indexOf('\n'))); } }, 5);
    child.once('exit', () => { clearInterval(timer); reject(new Error(`host exited before it was ready: ${stderr}`)); });
    setTimeout(() => { clearInterval(timer); reject(new Error('host not ready after 20 s')); }, 20000).unref();
  });
  return { child, ready, exited, stdout: () => stdout, stderr: () => stderr };
}

function request(port, method, url, body, { key = KEY, onData } = {}) {
  return new Promise((resolve, reject) => {
    const data = body === undefined ? undefined : JSON.stringify(body);
    const headers = { ...(key === null ? {} : { authorization: `Bearer ${key}` }), ...(data ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) } : {}) };
    const req = http.request({ host: '127.0.0.1', port, method, path: url, headers }, res => {
      const chunks = [];
      res.on('data', c => { chunks.push(c); onData?.(c, res); });
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        resolve({ status: res.statusCode, headers: res.headers, text, json: res.headers['content-type']?.includes('json') ? JSON.parse(text) : null });
      });
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

// Server-sent events of a response body: [{text}, ..., {done...}] and whether [DONE] closed it.
function parseSse(text) {
  const events = [];
  let closed = false;
  for (const block of text.split('\n\n')) {
    if (!block) continue;
    if (!block.startsWith('data: ')) throw new Error(`not an SSE line: ${block}`);
    const payload = block.slice(6);
    if (payload === '[DONE]') closed = true; else events.push(JSON.parse(payload));
  }
  return { events, closed };
}

const sseText = events => events.filter(p => 'text' in p).map(p => p.text).join('');

await group('host', async () => {
  const host = startHost();
  const line = await host.ready;
  const info = JSON.parse(line);
  check('one ready line of JSON with the announced fields', info.ready === true && Number.isInteger(info.port) && info.port > 0 &&
    info.pid === host.child.pid && info.threads === 2 && typeof info.variant === 'string' && same(info.skills, ['alpha', 'beta']) &&
    info.ctx === 512 && info.slots === 3, line);
  const port = info.port;
  const get = (url, opts) => request(port, 'GET', url, undefined, opts);
  const post = (url, body, opts) => request(port, 'POST', url, body, opts);

  // --- authorisation and plain errors
  const noKey = await get('/health', { key: null }), wrongKey = await get('/health', { key: 'nope' });
  const noKeyPost = await post('/generate', { prompt: 'hi' }, { key: null });
  check('requests without the key or with a wrong key get 403', noKey.status === 403 && wrongKey.status === 403 && noKeyPost.status === 403);
  const health = await get('/health');
  check('/health with the key', health.status === 200 && health.json.ok === true && health.json.ctx === 512 && same(health.json.skills, ['alpha', 'beta']) &&
    same(health.json.head, ['none', 'alpha', 'beta']) && Array.isArray(health.json.slots) && health.json.memory_bytes > 0 && health.json.threads === 2 &&
    typeof health.json.variant === 'string', JSON.stringify(health.json));
  check('unknown path is 404', (await get('/nothing')).status === 404);
  const badJson = await new Promise(resolve => {
    const req = http.request({ host: '127.0.0.1', port, method: 'POST', path: '/generate', headers: { authorization: `Bearer ${KEY}` } },
      res => { res.resume(); res.on('end', () => resolve(res.statusCode)); });
    req.end('{not json');
  });
  check('a body that is not JSON is 400', badJson === 400);

  // --- tokenize
  const tok = BpeTokenizer.fromJSON(JSON.parse(fs.readFileSync(file('tokenizer.json'), 'utf8')));
  const tokenized = await post('/tokenize', { text: 'the quick brown fox' });
  check('/tokenize', tokenized.status === 200 && same(tokenized.json.tokens, tok.encode('the quick brown fox')) && tokenized.json.count === tokenized.json.tokens.length);

  // --- generate, not streamed, then streamed
  const prompt = 'the quick brown fox jumps over the lazy dog. hello world! what is the weather today?';
  const promptIds = tok.encode(prompt);
  const base = { prompt, temperature: 0, max_tokens: 30, slot: 'one' };
  const plain = await post('/generate', { ...base, skill: 'alpha' });
  const j = plain.json;
  check('/generate (not streamed) returns text, usage, timings and the control logits', plain.status === 200 && typeof j.text === 'string' && j.text.length > 0 &&
    j.finish === 'length' && j.skill === 'alpha' && j.usage.prompt_tokens === promptIds.length && j.usage.cached_tokens === 0 && j.usage.completion_tokens === 30 &&
    j.control.names.length === 3 && j.control.logits.length === 3 && ['lower_ms', 'upper_ms', 'switch_ms', 'decode_ms', 'prompt_per_second', 'predicted_per_second'].every(k => typeof j.timings[k] === 'number'),
  JSON.stringify(j).slice(0, 400));

  // the same generation computed with the engine directly
  const e = await makeEngine({ variant: 'simd' });
  e.selectSkill(0);
  e.eval(promptIds);
  const direct = greedy(e, 30);
  check('the host generates the same tokens as the engine called directly (greedy)', tok.decode(direct, { skipSpecial: true }) === j.text);
  e.close();

  const streamed = await post('/generate', { ...base, slot: 'two', skill: 'alpha', stream: true });
  const sse = parseSse(streamed.text);
  const doneEvent = sse.events.at(-1), pieces = sse.events.slice(0, -1);
  check('/generate streamed: text events, then done, then [DONE]', streamed.headers['content-type'].startsWith('text/event-stream') && sse.closed &&
    pieces.length > 1 && pieces.every(p => typeof p.text === 'string' && Object.keys(p).length === 1) && doneEvent.done === true && doneEvent.finish === 'length' &&
    sseText(pieces) === j.text, `${pieces.length} text events`);
  check('the done event carries skill, control, usage and timings', doneEvent.skill === 'alpha' && doneEvent.usage.completion_tokens === 30 &&
    doneEvent.usage.cached_tokens === 0 && doneEvent.control.logits.length === 3 && typeof doneEvent.timings.decode_ms === 'number');

  // --- /control: the head names and logits for a prompt, the same numbers the engine gives and /generate returns
  const control = await post('/control', { prompt, slot: 'one' });
  const e2 = await makeEngine({ variant: 'simd' });
  e2.eval(promptIds);
  const controlEngine = Array.from(e2.control());
  e2.close();
  check('/control returns the head names and logits (the engine\'s own numbers)', control.status === 200 && same(control.json.names, ['none', 'alpha', 'beta']) &&
    maxAbs(control.json.logits, controlEngine) < 1e-6, JSON.stringify(control.json.logits));
  check('/generate returns the same control logits for the same prompt', maxAbs(doneEvent.control.logits, controlEngine) < 1e-6);

  // --- prefix cache
  const again = await post('/generate', { ...base, skill: 'alpha' });
  check('the identical prompt on the same slot reuses all but its last token and answers the same', again.json.usage.cached_tokens === promptIds.length - 1 &&
    again.json.text === j.text, `cached ${again.json.usage.cached_tokens} of ${promptIds.length}`);
  const longer = await post('/generate', { ...base, prompt: `${prompt} the user asks about the weather in paris`, skill: 'alpha', max_tokens: 5 });
  check('a longer prompt on the same slot reuses the shared prefix', longer.json.usage.cached_tokens >= promptIds.length,
    `cached ${longer.json.usage.cached_tokens}, shared prefix ${promptIds.length}`);
  const switched = await post('/generate', { ...base, skill: 'beta', max_tokens: 5 });
  check('another skill on the same slot recomputes the upper layers (switch time reported)', switched.json.skill === 'beta' && switched.json.timings.switch_ms > 0 &&
    switched.json.usage.cached_tokens > 0, `switch ${switched.json.timings.switch_ms} ms`);
  const reference = await post('/generate', { ...base, slot: 'fresh', skill: 'beta', max_tokens: 5 });
  check('the answer after a skill switch equals the answer on a fresh slot', switched.json.text === reference.json.text);

  // --- stop strings, on sampled text (a seed makes it repeatable) so that the output is not one repeated letter
  // The random tiny model picks bytes at random; banning the 128 high bytes and the three special tokens leaves
  // ASCII text that can be read and compared.
  const vocab = JSON.parse(fs.readFileSync(file('tokenizer.json'), 'utf8')).model.vocab;
  const byteOf = new Map(); // the printable stand-in character of each byte (GPT-2 table) -> the byte
  const keep = [...range(0x21, 0x7e), ...range(0xa1, 0xac), ...range(0xae, 0xff)];
  keep.forEach(b => byteOf.set(String.fromCharCode(b), b));
  range(0, 255).filter(b => !keep.includes(b)).forEach((b, i) => byteOf.set(String.fromCharCode(256 + i), b));
  const highBytes = Object.keys(vocab).filter(text => byteOf.get(text) >= 128);
  const ascii = [...highBytes, '<|endoftext|>', '<|im_start|>', '<|im_end|>'];
  const varied = { prompt, temperature: 2, top_k: 60, top_p: 1, repeat_penalty: 1.6, seed: 1234, max_tokens: 40, skill: 'alpha', ban: ascii };
  const full = (await post('/generate', { ...varied, slot: 's0' })).json.text;
  const at = Math.floor(full.length / 2), stop = full.slice(at, at + 3), cut = full.slice(0, full.indexOf(stop));
  check('the sampled text has some variety', new Set(full).size >= 8 && cut.length > 0, JSON.stringify(full));
  const stopped = await post('/generate', { ...varied, slot: 's1', stop: [stop, 'zzzz-not-there'] });
  check('a stop string ends the text before it and finish is "stop"', stopped.json.text === cut && stopped.json.finish === 'stop', `stop ${JSON.stringify(stop)}`);
  const stoppedStream = parseSse((await post('/generate', { ...varied, slot: 's2', stop: [stop], stream: true })).text);
  check('streamed text never contains the stop string or a part of it', sseText(stoppedStream.events) === cut && stoppedStream.events.at(-1).finish === 'stop');
  const nearly = full.slice(at, at + 2) + '\u0001';
  const partial = parseSse((await post('/generate', { ...varied, slot: 's3', stop: [nearly], stream: true })).text);
  check('text held back for a stop string that does not complete is released at the end', sseText(partial.events) === full && partial.events.at(-1).finish === 'length');
  const unicode = await post('/generate', { ...varied, slot: 's4', stream: true });
  check('streamed chunks are whole characters', parseSse(unicode.text).events.filter(p => 'text' in p).every(p => !p.text.includes('�')));

  // Two-byte characters arrive as two tokens. Only the two bytes of "\u00e9" are allowed here, so the model produces
  // them (and, by chance, broken pairs); a chunk must never carry half of a character.
  const charOfByte = new Map([...byteOf].map(([c, b]) => [b, c]));
  const allowed = new Set([vocab[charOfByte.get(0xc3)], vocab[charOfByte.get(0xa9)]]);
  const onlyE = { prompt, temperature: 100, top_k: 0, top_p: 1, seed: 3, max_tokens: 60, skill: 'none', slot: 's5',
    ban: [...Object.keys(vocab).filter(k => !allowed.has(vocab[k])), '<|endoftext|>', '<|im_start|>', '<|im_end|>'] };
  const eAcc = await makeEngine({ variant: 'simd' });
  eAcc.eval(promptIds);
  eAcc.bias.fill(-1e30);
  for (const id of allowed) eAcc.bias[id] = 0;
  eAcc.seed(3);
  const accIds = [];
  for (let i = 0; i < 60; i++) { const t = eAcc.sample({ temperature: 100, topK: 0, topP: 1 }); accIds.push(t); eAcc.eval([t]); }
  eAcc.close();
  const accStream = parseSse((await post('/generate', { ...onlyE, stream: true })).text);
  const accChunks = accStream.events.filter(p => 'text' in p).map(p => p.text);
  const accText = accChunks.join('');
  check('a character split over two tokens is sent whole, in one chunk', accText === tok.decode(accIds, { skipSpecial: true }) && (accText.match(/\u00e9/g) ?? []).length >= 2, `${(accText.match(/\u00e9/g) ?? []).length} of "\u00e9" in ${accText.length} characters, ${accChunks.length} chunks`);

  // --- ban: the first token the model would pick is refused, and the ban ends with the request
  const e3 = await makeEngine({ variant: 'simd' });
  e3.eval(promptIds);
  const top = e3.sample({ temperature: 0 });
  e3.reset();
  e3.bias[top] = -1e30;
  e3.eval(promptIds);
  const expectBanned = tok.decode(greedy(e3, 12), { skipSpecial: true });
  e3.close();
  const topText = tok.decode([top]);
  const sameToken = same(tok.encode(topText), [top]);
  const none = { ...base, slot: 'ban', skill: 'none', max_tokens: 12 };
  const plainNone = await post('/generate', none);
  const bannedRun = sameToken ? await post('/generate', { ...none, ban: [topText] }) : null;
  const unbanned = await post('/generate', none);
  check('a banned token is not chosen, as the engine with the same bias confirms', sameToken && bannedRun.json.text === expectBanned && bannedRun.json.text !== plainNone.json.text,
    `banned token ${JSON.stringify(topText)}`);
  check('the ban ends with the request', unbanned.json.text === plainNone.json.text);

  // --- /warm, auto, skills, errors
  const warmPrompt = 'x '.repeat(40) + prompt;
  const warm = await post('/warm', { prompt: warmPrompt, slot: 'warm' });
  const warmed = await post('/generate', { ...base, prompt: warmPrompt, slot: 'warm', skill: 'none', max_tokens: 4 });
  check('/warm fills the cache so that the first answer finds its prompt', warm.status === 200 && warm.json.usage.cached_tokens === 0 && warmed.json.usage.cached_tokens === warmed.json.usage.prompt_tokens - 1 &&
    warm.json.timings.lower_ms > 0, `warm ${JSON.stringify(warm.json)}`);
  const auto = await post('/generate', { ...base, slot: 'auto', skill: 'auto', max_tokens: 4 });
  const loaded = auto.json.control.names.map((n, i) => [n, auto.json.control.logits[i]]).filter(([n]) => ['alpha', 'beta'].includes(n)).sort((a, b) => b[1] - a[1]);
  check('skill "auto" uses the best loaded skill among the head outputs ("none" is ignored)', auto.json.skill === loaded[0][0], `${auto.json.skill} from ${JSON.stringify(auto.json.control)}`);
  check('unknown skill is 400', (await post('/generate', { prompt: 'hi', skill: 'gamma' })).status === 400);
  check('missing prompt is 400', (await post('/generate', { skill: 'none' })).status === 400);
  check('a wrongly typed parameter is 400', (await post('/generate', { prompt: 'hi', max_tokens: 'many' })).status === 400);
  check('a prompt longer than the context is 400', (await post('/generate', { prompt: 'the '.repeat(600) })).status === 400);
  check('a ban entry that is not one token is 400', (await post('/generate', { prompt: 'hi', ban: ['the quick brown'] })).status === 400);
  const unload = await post('/skills', { unload: 'beta' });
  const afterUnload = await post('/generate', { prompt: 'hi', skill: 'beta' });
  const reload = await post('/skills', { load: { name: 'beta', path: file('beta.kms') } });
  const afterReload = await post('/generate', { ...base, slot: 'reload', skill: 'beta', max_tokens: 5 });
  check('/skills unloads and loads again', same(unload.json.skills, ['alpha']) && afterUnload.status === 400 && same(reload.json.skills, ['alpha', 'beta']) &&
    afterReload.json.text === reference.json.text);
  check('/skills with a missing file is 400', (await post('/skills', { load: { name: 'x', path: '/nonexistent/x.kms' } })).status === 400);

  // --- several conversations at once
  const promptB = 'hello world! the user asks about the weather in paris';
  const solo = {};
  for (const [name, p, skill] of [['A', prompt, 'alpha'], ['B', promptB, 'none']]) {
    solo[name] = (await post('/generate', { ...varied, prompt: p, skill, max_tokens: 200, slot: `solo-${name}`, seed: name === 'A' ? 5 : 6 })).json.text;
  }
  const arrivals = [];
  const live = (name, p, skill) => post('/generate', { ...varied, prompt: p, skill, max_tokens: 200, slot: `live-${name}`, seed: name === 'A' ? 5 : 6, stream: true },
    { onData: () => arrivals.push(name) });
  const [ra, rb] = await Promise.all([live('A', prompt, 'alpha'), live('B', promptB, 'none')]);
  const textA = sseText(parseSse(ra.text).events), textB = sseText(parseSse(rb.text).events);
  check('two conversations at the same time give each the answer it gets alone', textA === solo.A && textB === solo.B && textA.length > 50 && textB.length > 50,
    `${textA.length} and ${textB.length} characters`);
  const order = arrivals.join('');
  const turns = (order.match(/AB|BA/g) ?? []).length;
  check('both conversations make progress in turns', turns >= 4, `${turns} changes of turn in ${order.length} writes`);

  // --- slots: at most --slots conversations, the least recently used idle one is replaced, memory stays put
  const filler = 'the quick brown fox jumps over the lazy dog. '.repeat(8);
  const sized = async key => {
    const r = await post('/generate', { prompt: filler, temperature: 0, max_tokens: 3, slot: key, skill: 'none' });
    if (r.status !== 200) throw new Error(`slot ${key}: ${r.text}`);
    return r;
  };
  for (const k of ['m1', 'm2', 'm3']) await sized(k);
  const afterThree = (await get('/health')).json;
  check('at most --slots conversations are kept', afterThree.slots.length === 3 && same(afterThree.slots.map(s => s.key).sort(), ['m1', 'm2', 'm3']),
    afterThree.slots.map(s => s.key).join(','));
  const bytesOf = afterThree.slots[0].bytes;
  await sized('m1'); // m1 is now the most recent, so m2 is the one that has been idle longest
  await sized('m4');
  const afterFour = (await get('/health')).json;
  check('the least recently used idle conversation is replaced', same(afterFour.slots.map(s => s.key).sort(), ['m1', 'm3', 'm4']), afterFour.slots.map(s => s.key).join(','));
  check('replacing a conversation does not grow the engine memory', afterFour.memory_bytes === afterThree.memory_bytes, `${afterThree.memory_bytes} -> ${afterFour.memory_bytes}`);
  for (const k of ['m5', 'm6', 'm7', 'm8']) await sized(k);
  const later = (await get('/health')).json;
  check('memory is still the same after more replacements', later.memory_bytes === afterThree.memory_bytes && later.slots.every(s => s.bytes >= bytesOf && !s.busy),
    `${later.memory_bytes} bytes, each slot ${later.slots.map(s => s.bytes).join('/')}`);
  const kept = await sized('m8');
  const evicted = await sized('m1');
  check('a slot that was kept answers from its cache, an evicted one starts over', kept.json.usage.cached_tokens > 0 && evicted.json.usage.cached_tokens === 0);

  // --- a client that leaves stops the work; a request that left while it waited never runs
  const longRun = { prompt, temperature: 0, max_tokens: 5000, slot: 'abort', stream: true, skill: 'none' };
  await new Promise(resolve => {
    const req = http.request({ host: '127.0.0.1', port, method: 'POST', path: '/generate', headers: { authorization: `Bearer ${KEY}` } }, res => {
      res.once('data', () => { req.destroy(); resolve(); });
    });
    req.end(JSON.stringify(longRun));
  });
  await new Promise(r => setTimeout(r, 300));
  const afterAbort = (await get('/health')).json.slots.find(s => s.key === 'abort');
  check('a client that disconnects stops the generation', afterAbort && !afterAbort.busy && afterAbort.n_past < 512 - 20, `n_past ${afterAbort?.n_past} of 512`);
  const resumed = await post('/generate', { prompt, temperature: 0, max_tokens: 3, slot: 'abort', skill: 'none' });
  check('the slot is usable after an abort', resumed.status === 200 && resumed.json.usage.cached_tokens > 0);

  const queue = Array.from({ length: 6 }, () => post('/generate', { prompt, temperature: 0, max_tokens: 400, slot: 'queue', skill: 'none' }));
  await new Promise(r => setTimeout(r, 20));
  const leaver = http.request({ host: '127.0.0.1', port, method: 'POST', path: '/generate', headers: { authorization: `Bearer ${KEY}` } });
  leaver.on('error', () => {});
  leaver.end(JSON.stringify({ prompt: 'zzz yyy '.repeat(30), temperature: 0, max_tokens: 5, slot: 'queue', skill: 'none' }));
  await new Promise(r => setTimeout(r, 30));
  leaver.destroy();
  const results = await Promise.all(queue);
  const afterQueue = await post('/generate', { prompt, temperature: 0, max_tokens: 3, slot: 'queue', skill: 'none' });
  check('requests of one slot are answered in order and a waiting request that left never runs',
    results.every(r => r.status === 200) && afterQueue.json.usage.cached_tokens === promptIds.length - 1, `cached ${afterQueue.json.usage.cached_tokens}`);

  // --- the process goes away with its parent, and only the ready line was ever written to stdout
  const t0 = Date.now();
  host.child.stdin.end();
  const gone = await host.exited;
  check('the host exits within 2 s when stdin reaches EOF', gone.code === 0 && gone.at - t0 < 2000, `${gone.at - t0} ms, code ${gone.code}`);
  check('stdout holds the ready line and nothing else', host.stdout() === `${line}\n`);
});

await group('host signals and arguments', async () => {
  const host = startHost([], { threads: 1, slots: 1 });
  const info = JSON.parse(await host.ready);
  check('a single-slot host with one thread starts', info.slots === 1 && info.threads === 1);
  const post = body => request(info.port, 'POST', '/generate', body);
  const a1 = await post({ prompt: 'hello world!', slot: 'a', temperature: 0, max_tokens: 3 });
  const b1 = await post({ prompt: 'the quick brown fox', slot: 'b', temperature: 0, max_tokens: 3 });
  const a2 = await post({ prompt: 'hello world!', slot: 'a', temperature: 0, max_tokens: 3 });
  check('with one slot a second key replaces the first', a1.status === 200 && b1.status === 200 && a2.json.usage.cached_tokens === 0 && a2.json.text === a1.json.text);
  const t0 = Date.now();
  host.child.kill('SIGTERM');
  const gone = await host.exited;
  check('the host exits within 2 s on SIGTERM', gone.at - t0 < 2000 && (gone.code === 0 || gone.signal === 'SIGTERM'), `${gone.at - t0} ms`);

  const noKey = spawnSync('node', [HOST, '--model', file('tiny.fkm'), '--tokenizer', file('tokenizer.json')], { env: { ...process.env, KIWIMIND_KEY: '' }, encoding: 'utf8', timeout: 10000 });
  check('without KIWIMIND_KEY the host refuses to start', noKey.status !== 0 && /KIWIMIND_KEY/.test(noKey.stderr), noKey.stderr.trim().split('\n')[0]);
  const bogus = spawnSync('node', [HOST, '--bogus', '1'], { env: { ...process.env, KIWIMIND_KEY: KEY }, encoding: 'utf8', timeout: 10000 });
  check('an unknown option is refused with the usage text', bogus.status === 2 && /usage:/.test(bogus.stderr));
});

console.log(failed === 0 ? 'all checks passed' : `${failed} check(s) failed`);
process.exit(failed === 0 ? 0 : 1);
