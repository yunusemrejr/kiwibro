// Kiwi Mind scoped skills and the action unit, on the tiny random model of scripts/mind/formats.py.
//
//   node tests/mind_scoped.mjs DIR     exit code 0 when every check passed, 1 otherwise; one line per check
//
// DIR is prepared by tests/test_mind_scoped.py: the tiny set, units.kmh and plan.json (the prompts, their tokens and
// the logits a NumPy forward pass gives when every position has its own skill). The helpers below are the ones of
// mind_engine.mjs.
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { MindEngine, VARIANTS, wasmFeatures } from '../kiwibro/mind/host/engine.mjs';
import { BpeTokenizer } from '../kiwibro/mind/host/tokenizer.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const HOST = path.join(ROOT, 'kiwibro/mind/host/host.mjs');
const KEY = 'test-secret-key';
const dir = process.argv[2];
if (!dir) { console.error('usage: node tests/mind_scoped.mjs DIR'); process.exit(2); }
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

const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const peak = a => Math.max(...Array.from(a, Math.abs));
const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
const common = (a, b) => { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i; };

const file = name => path.join(dir, name);
const plan = JSON.parse(fs.readFileSync(file('plan.json'), 'utf8'));
const tok = BpeTokenizer.fromJSON(JSON.parse(fs.readFileSync(file('tokenizer.json'), 'utf8')));
const ONE = plan.tokens_one, TWO = plan.tokens_two, SYSTEM = plan.system_end;
const THREE_TEXT = `${plan.prompt_two}fine.<|im_end|>\n<|im_start|>user\nthe weather in paris<|im_end|>\n<|im_start|>assistant\n`;
const THREE = tok.encode(THREE_TEXT);
const ALPHA = 0, BETA = 1, NONE = -1;

async function makeEngine({ variant = 'simd', threads = 1, scoped = false } = {}) {
  const e = await MindEngine.create({ variant, threads });
  e.loadTrunk(file('tiny.fkm'), 512);
  e.loadSkill(ALPHA, fs.readFileSync(file('alpha.kms')));
  e.loadSkill(BETA, fs.readFileSync(file('beta.kms')));
  if (scoped) e.scope(true);
  return e;
}

const variants = VARIANTS.filter(v => {
  const f = wasmFeatures();
  return f.simd && (v === 'simd' || (f.threads && (v === 'simd-mt' || f.relaxed)));
});

// The same tolerance and reason as in mind_engine.mjs: the reference repeats the engine's roundings, so only
// summation order is left; a wrong skill on a position moves the logits by whole units.
const TOL = 0.02;
function matches(e, caseName) {
  const want = plan.cases[caseName].logits, got = Array.from(e.logits);
  const diff = maxAbs(got, want);
  return [diff <= TOL * peak(want), `max diff ${diff.toExponential(2)}, limit ${(TOL * peak(want)).toExponential(2)}`];
}

// ------------------------------------------------------------------ 1. the engine

await group('prompts', async () => {
  check('the host tokenizer reads the chat prompts as the Python reference does',
    same(tok.encode(plan.prompt_one), ONE) && same(tok.encode(plan.prompt_two), TWO));
  const imEnd = tok.tokenToId('<|im_end|>');
  check('the system turn ends two positions after the first <|im_end|>', ONE.indexOf(imEnd) + 2 === SYSTEM && ONE[SYSTEM - 1] === TWO[SYSTEM - 1]);
});

await group('scoped engine', async () => {
  for (const variant of variants) {
    const threads = variant === 'simd' ? 1 : 3, tag = `${variant}/${threads}t`;
    const plain = await makeEngine({ variant, threads });
    check(`scoped skills are off until asked ${tag}`, plain.e.km_scoped() === 0);
    plain.close();

    const e = await makeEngine({ variant, threads, scoped: true });
    check(`km_scope(1) turns them on ${tag}`, e.e.km_scoped() === 1);
    e.evalLower(ONE);
    e.selectSkill(NONE);
    check(`the system turn alone: its upper layers are done, the turn waits ${tag}`, e.evalUpper(SYSTEM) === ONE.length - SYSTEM && e.nUpper === SYSTEM);
    e.selectSkill(ALPHA);
    check(`first turn under alpha: nothing waits ${tag}`, e.evalUpper() === 0 && e.nUpper === ONE.length);
    let [ok, detail] = matches(e, 'scoped, first turn');
    check(`logits: system turn under no skill, first turn under alpha, as the reference ${tag}`, ok, detail);

    // the next request: the first turn stays as it was computed, only the new positions take the new skill
    e.evalLower(TWO.slice(ONE.length));
    check(`new tokens leave the upper cache alone ${tag}`, e.nUpper === ONE.length && e.nPast === TWO.length);
    e.selectSkill(BETA);
    check(`another skill does not drop the upper cache ${tag}`, e.nUpper === ONE.length);
    check(`only the new positions are computed (nothing earlier waits) ${tag}`, e.evalUpper(TWO.length - ONE.length) === 0 && e.nUpper === TWO.length);
    [ok, detail] = matches(e, 'scoped, second turn under another skill');
    check(`logits: system under none, turn 1 under alpha, turn 2 under beta, as the reference ${tag}`, ok, detail);

    // a truncation inside the second turn keeps the cache up to there, and the recomputation gives the same bits
    const before = Array.from(e.logits);
    e.truncate(ONE.length + 3);
    check(`truncate keeps the upper cache up to the cut ${tag}`, e.nUpper === ONE.length + 3);
    e.evalLower(TWO.slice(ONE.length + 3));
    e.selectSkill(BETA);
    e.evalUpper();
    check(`recomputing the cut part under the same skill gives the same logits ${tag}`, same(Array.from(e.logits), before));

    // A skill that is reloaded may hide in any position of any cache: all of them are dropped.
    e.loadSkill(BETA, fs.readFileSync(file('beta.kms')));
    check(`reloading a skill drops the upper cache (any positions may be of it) ${tag}`, e.nUpper === 0 && !e.logitsReady);
    e.close();
  }
});

await group('unscoped engine', async () => {
  for (const variant of variants) {
    const threads = variant === 'simd' ? 1 : 3, tag = `${variant}/${threads}t`;
    const e = await makeEngine({ variant, threads });
    e.evalLower(ONE);
    e.selectSkill(ALPHA);
    e.evalUpper();
    let [ok, detail] = matches(e, 'unscoped, first turn');
    check(`the system turn runs under the skill too ${tag}`, ok, detail);
    e.evalLower(TWO.slice(ONE.length));
    e.selectSkill(BETA);
    const left = e.evalUpper(TWO.length - ONE.length);
    check(`another skill drops the upper cache: the earlier positions wait again ${tag}`, left === ONE.length && e.nUpper === TWO.length - ONE.length, `${left} wait`);
    e.evalUpper();
    [ok, detail] = matches(e, 'unscoped, second turn');
    check(`and the result is the whole prompt under the new skill ${tag}`, ok, detail);
    e.loadSkill(BETA, fs.readFileSync(file('beta.kms')));
    check(`reloading the skill the cache was built under drops it ${tag}`, e.nUpper === 0);
    e.selectSkill(ALPHA);
    e.evalUpper();
    e.loadSkill(BETA, fs.readFileSync(file('beta.kms')));
    check(`reloading another skill leaves it ${tag}`, e.nUpper === TWO.length);
    e.close();
  }
});

await group('state at a position', async () => {
  const e = await makeEngine();
  e.evalLower(ONE);
  const state = e.stateAt(ONE.length - 1);
  check('the state of an evaluated position is a copy of dim values', state instanceof Float32Array && state.length === e.dim && peak(state) > 0);
  check('outside what was evaluated there is none', e.stateAt(-1) === null && e.stateAt(ONE.length) === null);
  e.truncate(5);
  check('a truncated position has none either', e.stateAt(5) === null && e.stateAt(4) !== null);
  e.close();
});

// ------------------------------------------------------------------ 2. the host

function startHost({ units = false, scoped = false, unitsFile = 'units.kmh', slots = 6 } = {}) {
  const args = [HOST, '--model', file('tiny.fkm'), '--tokenizer', file('tokenizer.json'),
    '--skill', `alpha=${file('alpha.kms')}`, '--skill', `beta=${file('beta.kms')}`, '--head', file('head.kmh'),
    '--threads', '2', '--ctx', '512', '--slots', String(slots), '--port', '0',
    ...(units ? ['--units', file(unitsFile)] : []), ...(scoped ? ['--scoped', '1'] : [])];
  const child = spawn('node', args, { env: { ...process.env, KIWIMIND_KEY: KEY }, stdio: ['pipe', 'pipe', 'pipe'] });
  let stderr = '', stdout = '';
  child.stderr.on('data', d => { stderr += d; });
  child.stdout.on('data', d => { stdout += d; });
  const ready = new Promise((resolve, reject) => {
    const timer = setInterval(() => { if (stdout.includes('\n')) { clearInterval(timer); resolve(JSON.parse(stdout.slice(0, stdout.indexOf('\n')))); } }, 5);
    child.once('exit', () => { clearInterval(timer); reject(new Error(`host exited before it was ready: ${stderr}`)); });
    setTimeout(() => { clearInterval(timer); reject(new Error('host not ready after 20 s')); }, 20000).unref();
  });
  const stop = () => new Promise(resolve => { child.once('exit', resolve); child.stdin.end(); });
  return { child, ready, stop };
}

function request(port, method, url, body) {
  return new Promise((resolve, reject) => {
    const data = body === undefined ? undefined : JSON.stringify(body);
    const headers = { authorization: `Bearer ${KEY}`, ...(data ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) } : {}) };
    const req = http.request({ host: '127.0.0.1', port, method, path: url, headers }, res => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        resolve({ status: res.statusCode, text, json: res.headers['content-type']?.includes('json') ? JSON.parse(text) : null });
      });
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

// What the host should answer, computed by the engine directly: `layout` lists [skill, end position] for the upper
// layers in order (the last end is the prompt's length), then the greedy continuation. The random tiny model puts one
// token (a line break) far above all others, which would give the same text under any skill; so only the runner-up
// tokens are allowed (all others are banned, in the request too), and then the skills decide the text. `ban` is
// given to repeat what an earlier call chose; without it the runners-up are those of this layout.
const RUNNERS = 60;
const SPECIAL = ['<|endoftext|>', '<|im_start|>', '<|im_end|>'];
const vocab = JSON.parse(fs.readFileSync(file('tokenizer.json'), 'utf8')).model.vocab;
const words = Object.keys(vocab);
async function emulate({ tokens, layout, scoped, count, ban }) {
  const e = await makeEngine({ scoped });
  e.evalLower(tokens);
  let at = 0;
  for (const [skill, end] of layout) {
    e.selectSkill(skill);
    if (end > at) { e.evalUpper(end - at); at = end; }
  }
  if (!ban) {
    const logits = Array.from(e.logits);
    const runnersUp = words.slice().sort((x, y) => logits[vocab[y]] - logits[vocab[x]]).slice(2, RUNNERS);
    ban = [...words.filter(w => !runnersUp.includes(w)), ...SPECIAL];
  }
  for (const word of ban) e.bias[tok.tokenToId(word)] = -1e30;
  const ids = [];
  for (let i = 0; i < count; i++) {
    const t = e.sample({ temperature: 0 });
    ids.push(t);
    e.eval([t]);
  }
  e.close();
  return { text: tok.decode(ids, { skipSpecial: true }), ban };
}

// The unit's two layers over the normed state, in float64 from the weights the Python reader took from the file.
function unitOutputs(state) {
  const { w1, b1, w2, b2, eps } = plan.unit;
  const scale = 1 / Math.sqrt(state.reduce((s, v) => s + v * v, 0) / state.length + eps);
  const mid = w1.map((row, j) => Math.max(0, row.reduce((s, w, i) => s + w * state[i], 0) * scale + b1[j]));
  return w2.map((row, k) => row.reduce((s, w, j) => s + w * mid[j], b2[k]));
}

const COUNT = 16;
const generate = (port, body) => request(port, 'POST', '/generate', { temperature: 0, max_tokens: COUNT, ...body });

await group('scoped host', async () => {
  const host = startHost({ units: true, scoped: true });
  const port = (await host.ready).port;
  const p2 = Math.min(common(ONE, TWO), TWO.length - 1);
  const p3 = Math.min(common(TWO, THREE), THREE.length - 1);
  const layouts = {
    one: [[NONE, SYSTEM], [ALPHA, ONE.length]],
    two: [[NONE, SYSTEM], [ALPHA, p2], [BETA, TWO.length]],
    three: [[NONE, SYSTEM], [ALPHA, p2], [BETA, p3], [ALPHA, THREE.length]],
  };
  const want = {
    one: await emulate({ tokens: ONE, layout: layouts.one, scoped: true, count: COUNT }),
    two: await emulate({ tokens: TWO, layout: layouts.two, scoped: true, count: COUNT }),
    three: await emulate({ tokens: THREE, layout: layouts.three, scoped: true, count: COUNT }),
  };
  const ask = (name, prompt, skill, slot) => generate(port, { prompt, skill, slot, ban: want[name].ban });

  const r1 = await ask('one', plan.prompt_one, 'alpha', 'chat');
  const r2 = await ask('two', plan.prompt_two, 'beta', 'chat');
  const r3 = await ask('three', THREE_TEXT, 'alpha', 'chat');
  check('three requests of one slot, each under another skill, are answered', [r1, r2, r3].every(r => r.status === 200 && r.json.text.length > 0),
    [r1, r2, r3].map(r => r.status).join(','));
  check('the second request finds the whole first turn in the cache', r2.json.usage.cached_tokens === p2 && p2 === ONE.length, `cached ${r2.json.usage.cached_tokens} of ${TWO.length}`);
  check('the third finds the first two turns', r3.json.usage.cached_tokens === p3 && p3 === TWO.length, `cached ${r3.json.usage.cached_tokens} of ${THREE.length}`);
  check('a change of skill costs no switch time', [r2, r3].every(r => r.json.timings.switch_ms === 0) && r2.json.skill === 'beta' && r3.json.skill === 'alpha');
  check('only the new tokens go through the lower and the upper layers',
    r2.json.timings.lower_ms > 0 && r2.json.timings.upper_ms > 0 && r2.json.usage.prompt_tokens - r2.json.usage.cached_tokens === TWO.length - ONE.length);
  check('the first answer is that of one fresh pass: system turn under none, the turn under alpha', r1.json.text === want.one.text, JSON.stringify(r1.json.text));
  check('the second answer is that of one fresh pass: none, alpha, beta', r2.json.text === want.two.text, JSON.stringify(r2.json.text));
  check('the third answer is that of one fresh pass: none, alpha, beta, alpha', r3.json.text === want.three.text, JSON.stringify(r3.json.text));
  const wrong = await emulate({ tokens: TWO, layout: [[BETA, TWO.length]], scoped: false, count: COUNT, ban: want.two.ban });
  const noSystem = await emulate({ tokens: ONE, layout: [[ALPHA, ONE.length]], scoped: true, count: COUNT, ban: want.one.ban });
  check('other layouts (all under beta; a system turn under a skill) would have given other texts, so the checks above can fail',
    wrong.text !== want.two.text && noSystem.text !== want.one.text, `${JSON.stringify(wrong.text)} ${JSON.stringify(noSystem.text)} against ${JSON.stringify(want.two.text)} ${JSON.stringify(want.one.text)}`);

  // replayed on a slot of its own: the same requests in the same order give the same answers
  const q1 = await ask('one', plan.prompt_one, 'alpha', 'replay');
  const q2 = await ask('two', plan.prompt_two, 'beta', 'replay');
  const q3 = await ask('three', THREE_TEXT, 'alpha', 'replay');
  check('the same three requests on a new slot give the same answers and the same reuse', q1.json.text === r1.json.text && q2.json.text === r2.json.text &&
    q3.json.text === r3.json.text && q2.json.usage.cached_tokens === r2.json.usage.cached_tokens && q3.json.usage.cached_tokens === r3.json.usage.cached_tokens);

  // /warm with skill none computes only the system turn: the turns are left to the skill of their reply
  const wantWarm = await emulate({ tokens: TWO, layout: [[NONE, SYSTEM], [ALPHA, TWO.length]], scoped: true, count: COUNT });
  const allNone = await emulate({ tokens: TWO, layout: [[NONE, TWO.length]], scoped: true, count: COUNT, ban: wantWarm.ban });
  const warm = await request(port, 'POST', '/warm', { prompt: plan.prompt_two, slot: 'warm', skill: 'none' });
  const afterWarm = await generate(port, { prompt: plan.prompt_two, skill: 'alpha', slot: 'warm', ban: wantWarm.ban });
  check('/warm none finds nothing cached and answers', warm.status === 200 && warm.json.usage.cached_tokens === 0 && warm.json.timings.lower_ms > 0);
  check('after /warm none the turn still runs under the skill of its reply', afterWarm.json.usage.cached_tokens === TWO.length - 1 && afterWarm.json.text === wantWarm.text, JSON.stringify(afterWarm.json.text));
  check('(the same prompt entirely under no skill would have given another text)', allNone.text !== wantWarm.text);
  const warmNamed = await request(port, 'POST', '/warm', { prompt: plan.prompt_two, slot: 'named', skill: 'beta' });
  const afterNamed = await generate(port, { prompt: plan.prompt_two, skill: 'beta', slot: 'named', ban: wantWarm.ban });
  const wantNamed = await emulate({ tokens: TWO, layout: [[NONE, SYSTEM], [BETA, TWO.length]], scoped: true, count: COUNT, ban: wantWarm.ban });
  check('/warm with a skill computes the turns under it, the system turn still under none', warmNamed.status === 200 && afterNamed.json.text === wantNamed.text);

  // /control: the head, the action unit, the flag
  const control = await request(port, 'POST', '/control', { prompt: plan.prompt_two, slot: 'ctl' });
  const again = await request(port, 'POST', '/control', { prompt: plan.prompt_two, slot: 'ctl' });
  const fresh = await request(port, 'POST', '/control', { prompt: plan.prompt_two, slot: 'ctl2' });
  const j = control.json;
  check('/control names the head outputs and the unit outputs', control.status === 200 && same(j.names, ['none', 'alpha', 'beta']) && j.logits.length === 3 &&
    same(j.units.names, plan.unit.names) && j.units.logits.length === 3 && j.units.logits.every(Number.isFinite), JSON.stringify(j.units));
  check('/control says that the host runs scoped skills', j.scoped === true);
  const e = await makeEngine();
  e.evalLower(TWO);
  const expected = unitOutputs(Array.from(e.stateAt(TWO.length - 1)));
  e.close();
  check('the unit outputs are its two layers over the state of the last position', maxAbs(j.units.logits, expected) <= 1e-4 * Math.max(1, peak(expected)),
    `host ${j.units.logits.map(v => v.toFixed(4))} vs ${expected.map(v => v.toFixed(4))}`);
  check('they are the same when the prompt is asked again (cached) and on a new slot', same(again.json.units.logits, j.units.logits) && same(fresh.json.units.logits, j.units.logits) &&
    again.json.usage.cached_tokens === TWO.length - 1);
  check('/generate reports the unit with its control', same(r2.json.control.units.names, j.units.names) && r2.json.control.units.logits.length === 3);
  await host.stop();
});

await group('unscoped host', async () => {
  const host = startHost({ units: true, scoped: false });
  const port = (await host.ready).port;
  const p2 = Math.min(common(ONE, TWO), TWO.length - 1);
  const want1 = await emulate({ tokens: ONE, layout: [[ALPHA, ONE.length]], scoped: false, count: COUNT });
  const want2 = await emulate({ tokens: TWO, layout: [[BETA, TWO.length]], scoped: false, count: COUNT });
  const r1 = await generate(port, { prompt: plan.prompt_one, skill: 'alpha', slot: 'chat', ban: want1.ban });
  const r2 = await generate(port, { prompt: plan.prompt_two, skill: 'beta', slot: 'chat', ban: want2.ban });
  check('without --scoped a change of skill rebuilds the upper layers (switch time is reported)', r2.json.timings.switch_ms > 0 && r2.json.usage.cached_tokens === p2, `switch ${r2.json.timings.switch_ms} ms`);
  check('the first answer is the whole prompt under alpha, the system turn too', r1.json.text === want1.text, JSON.stringify(r1.json.text));
  check('the second is the whole prompt under beta', r2.json.text === want2.text, JSON.stringify(r2.json.text));
  const scopedWant = await emulate({ tokens: TWO, layout: [[NONE, SYSTEM], [ALPHA, p2], [BETA, TWO.length]], scoped: true, count: COUNT, ban: want2.ban });
  check('(and it is not the scoped answer)', want2.text !== scopedWant.text);
  const wantWarm = await emulate({ tokens: TWO, layout: [[ALPHA, TWO.length]], scoped: false, count: COUNT });
  const warm = await request(port, 'POST', '/warm', { prompt: plan.prompt_two, slot: 'warm', skill: 'none' });
  const afterWarm = await generate(port, { prompt: plan.prompt_two, skill: 'alpha', slot: 'warm', ban: wantWarm.ban });
  check('/warm none computes the whole prompt; another skill afterwards rebuilds it', warm.status === 200 && afterWarm.json.timings.switch_ms > 0 && afterWarm.json.text === wantWarm.text);
  const control = await request(port, 'POST', '/control', { prompt: plan.prompt_two, slot: 'ctl' });
  check('/control still reads the unit and says that it is not scoped', control.json.scoped === false && same(control.json.units.names, plan.unit.names) && same(control.json.names, ['none', 'alpha', 'beta']));
  await host.stop();
});

await group('host without a unit', async () => {
  const host = startHost({ units: false, scoped: true });
  const port = (await host.ready).port;
  const control = await request(port, 'POST', '/control', { prompt: plan.prompt_two, slot: 'ctl' });
  check('without --units /control answers units null and keeps the head', control.status === 200 && control.json.units === null && control.json.logits.length === 3 &&
    same(control.json.names, ['none', 'alpha', 'beta']));
  check('--scoped does not need a unit', control.json.scoped === true);
  const r = await generate(port, { prompt: plan.prompt_one, skill: 'alpha', slot: 'chat' });
  check('/generate answers and its control has no unit', r.status === 200 && r.json.control.units === null);
  await host.stop();
});

await group('host arguments', async () => {
  const run = extra => spawnSync('node', [HOST, '--model', file('tiny.fkm'), '--tokenizer', file('tokenizer.json'), '--head', file('head.kmh'), ...extra],
    { env: { ...process.env, KIWIMIND_KEY: KEY }, encoding: 'utf8', timeout: 10000 });
  const usage = spawnSync('node', [HOST, '--bogus', '1'], { env: { ...process.env, KIWIMIND_KEY: KEY }, encoding: 'utf8', timeout: 10000 });
  check('the usage text names --units and --scoped', /--units <file\.kmh>/.test(usage.stderr) && /--scoped 1/.test(usage.stderr), usage.stderr.trim().split('\n').join(' | '));
  const missing = run(['--units', file('nothing.kmh')]);
  check('a unit file that is not there stops the host', missing.status !== 0 && /nothing\.kmh/.test(missing.stderr), missing.stderr.trim().split('\n')[0]);
  const notAUnit = run(['--units', file('alpha.kms')]);
  check('a file that is not a head file stops the host', notAUnit.status !== 0 && /not a head file/.test(notAUnit.stderr), notAUnit.stderr.trim().split('\n')[0]);
  const bytes = fs.readFileSync(file('units.kmh'));
  fs.writeFileSync(file('units_cut.kmh'), bytes.subarray(0, bytes.length - 700));
  const cut = run(['--units', file('units_cut.kmh')]);
  check('a unit file that is cut short stops the host', cut.status !== 0 && /truncated/.test(cut.stderr), cut.stderr.trim().split('\n')[0]);
  const narrow = run(['--units', file('units_narrow.kmh')]);
  check('a unit made for another width of state stops the host at start, not at the first question', narrow.status !== 0 && /unit/.test(narrow.stderr), narrow.stderr.trim().split('\n')[0]);
  const flag = run(['--scoped', 'yes']);
  check('--scoped takes 0 or 1 and nothing else', flag.status === 2 && /--scoped/.test(flag.stderr) && /usage:/.test(flag.stderr), flag.stderr.trim().split('\n')[0]);
});

console.log(failed === 0 ? 'all checks passed' : `${failed} check(s) failed`);
process.exit(failed === 0 ? 0 : 1);
