// Kiwi Mind host: one Node.js process that holds the model once and serves several conversations ("slots") over
// HTTP on 127.0.0.1. No dependencies. See the usage text below for the command line.
//
// How work is shared: the engine has one set of weights and one thread pool, and every engine call finishes before
// the next begins. A request becomes a job that runs as a generator; each `yield` ends one small piece of work
// (a prefill piece of up to PREFILL_PIECE tokens, or up to DECODE_TURN generated tokens). A round-robin loop gives
// every slot that has a job one piece per turn and lets the HTTP server run in between, so two conversations
// both move forward at the same thread count and stream, abort and queue while others compute.
import http from 'node:http';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { MindEngine, parseSkillHeader, parseHeadHeader, pickVariant, defaultThreads } from './engine.mjs';
import { BpeTokenizer } from './tokenizer.mjs';

const PREFILL_PIECE = 64;
const DECODE_TURN = 4;
const MAX_BODY = 8 << 20;
const USAGE = 'usage: node host.mjs --model <trunk.fkm> --tokenizer <tokenizer.json> [--skill name=<file.kms> ...] ' +
  '[--head <file.kmh>] [--threads N] [--ctx 4096] [--slots 3] [--port 0]   (secret in env KIWIMIND_KEY)';

class HttpError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}
const bad = message => new HttpError(400, message);

// ------------------------------------------------------------------ command line

function parseArgs(argv) {
  const opts = { skills: [], threads: defaultThreads(), ctx: 4096, slots: 3, port: 0 };
  const intOption = (name, value, min, max) => {
    const n = Number(value);
    if (!Number.isInteger(n) || n < min || n > max) throw new Error(`${name} must be an integer from ${min} to ${max}`);
    return n;
  };
  for (let i = 0; i < argv.length; i += 2) {
    const flag = argv[i], value = argv[i + 1];
    if (value === undefined) throw new Error(`${flag} needs a value`);
    switch (flag) {
      case '--model': opts.model = value; break;
      case '--tokenizer': opts.tokenizer = value; break;
      case '--head': opts.head = value; break;
      case '--skill': {
        const at = value.indexOf('=');
        if (at < 1 || at === value.length - 1) throw new Error('--skill needs name=<file.kms>');
        opts.skills.push({ name: value.slice(0, at), path: value.slice(at + 1) });
        break;
      }
      case '--threads': opts.threads = intOption(flag, value, 1, 16); break;
      case '--ctx': opts.ctx = intOption(flag, value, 64, 1 << 20); break;
      case '--slots': opts.slots = intOption(flag, value, 1, 16); break;
      case '--port': opts.port = intOption(flag, value, 0, 65535); break;
      default: throw new Error(`unknown option ${flag}`);
    }
  }
  if (!opts.model || !opts.tokenizer) throw new Error('--model and --tokenizer are required');
  return opts;
}

// The trunk's own end-of-text id (header field eos_id), read without loading the file.
function readEosId(file) {
  const fd = fs.openSync(file, 'r');
  try {
    const head = Buffer.alloc(200);
    fs.readSync(fd, head, 0, 200, 0);
    return head.readUInt32LE(56);
  } finally {
    fs.closeSync(fd);
  }
}

// ------------------------------------------------------------------ request validation (once, at the edge)

function field(body, name, type, fallback) {
  const v = body[name];
  if (v === undefined || v === null) return fallback;
  if (typeof v !== type) throw bad(`"${name}" must be a ${type}`);
  return v;
}

function number(body, name, fallback, min, max, integer = false) {
  const v = field(body, name, 'number', fallback);
  if (!Number.isFinite(v) || v < min || v > max || (integer && !Number.isInteger(v))) {
    throw bad(`"${name}" must be ${integer ? 'an integer' : 'a number'} from ${min} to ${max}`);
  }
  return v;
}

// ------------------------------------------------------------------ stop strings

// Text that could still turn into a stop string is held back; nothing of a stop string is ever released.
class StopFilter {
  constructor(stops) { this.stops = stops; this.held = ''; this.hit = false; }

  push(text) {
    const buf = this.held + text;
    let cut = -1;
    for (const s of this.stops) {
      const at = buf.indexOf(s);
      if (at >= 0 && (cut < 0 || at < cut)) cut = at;
    }
    if (cut >= 0) { this.hit = true; this.held = ''; return buf.slice(0, cut); }
    let keep = 0;
    for (const s of this.stops) {
      for (let k = Math.min(s.length - 1, buf.length); k > keep; k--) {
        if (buf.endsWith(s.slice(0, k))) { keep = k; break; }
      }
    }
    this.held = buf.slice(buf.length - keep);
    return buf.slice(0, buf.length - keep);
  }

  flush() { const rest = this.held; this.held = ''; return rest; }
}

// ------------------------------------------------------------------ the host

async function main(opts) {
  const key = process.env.KIWIMIND_KEY;
  if (!key) throw new Error('env KIWIMIND_KEY must hold the secret every request has to carry');
  const keyDigest = crypto.createHash('sha256').update(`Bearer ${key}`).digest();

  const started = performance.now();
  const engine = await MindEngine.create({ variant: pickVariant(), threads: opts.threads });
  engine.loadTrunk(opts.model, opts.ctx);
  const tokenizer = BpeTokenizer.fromJSON(JSON.parse(fs.readFileSync(opts.tokenizer, 'utf8')));
  const maxSlots = Math.min(opts.slots, engine.maxSlots);
  const eosIds = new Set([readEosId(opts.model)]);
  for (const name of ['<|im_end|>', '<|endoftext|>']) {
    const id = tokenizer.tokenToId(name);
    if (id !== undefined) eosIds.add(id);
  }

  // ---- skills and head
  const skills = new Map(); // name -> engine skill slot
  const skillNames = () => [...skills.keys()];
  const skillName = index => [...skills].find(([, i]) => i === index)?.[0] ?? null;

  const loadSkill = (name, file) => {
    const bytes = fs.readFileSync(file);
    parseSkillHeader(bytes);
    let index = skills.get(name);
    if (index === undefined) {
      index = Array.from({ length: engine.skillSlots }, (_, i) => i).find(i => skillName(i) === null);
      if (index === undefined) throw bad(`at most ${engine.skillSlots} skills can be loaded`);
    }
    try {
      engine.loadSkill(index, bytes);
    } catch (e) {
      // A skill that starts at another layer moves the split: every cache is rebuilt, so only when none is in use.
      if (e.code !== -8) throw e;
      if ([...contexts.values()].some(c => c.job || c.queue.length)) throw new HttpError(409, 'conversations are running; try again when they are idle');
      for (const c of contexts.values()) { engine.slotUse(c.id); engine.reset(); c.tokens = []; }
      engine.loadSkill(index, bytes);
    }
    skills.set(name, index);
  };

  const unloadSkill = name => {
    const index = skills.get(name);
    if (index === undefined) throw bad(`no skill named "${name}"`);
    engine.unloadSkill(index);
    for (const c of contexts.values()) if (c.upperSkill === index) c.upperSkill = -1;
    skills.delete(name);
  };

  let head = null; // class names of the control head
  if (opts.head) {
    const bytes = fs.readFileSync(opts.head);
    head = parseHeadHeader(bytes).names;
    engine.loadHead(bytes);
  }
  for (const s of opts.skills) loadSkill(s.name, s.path);

  // ---- conversations
  const contexts = new Map(); // key -> { key, id, tokens, upperSkill, job, queue, used }
  const waiting = []; // jobs whose slot key has no conversation yet and no free or idle one to take
  let tick = 0;

  const idle = c => !c.job && c.queue.length === 0;

  const takeContext = key => {
    if (contexts.size >= maxSlots) {
      let oldest = null;
      for (const c of contexts.values()) if (idle(c) && (!oldest || c.used < oldest.used)) oldest = c;
      if (!oldest) return null;
      engine.slotFree(oldest.id);
      contexts.delete(oldest.key);
    }
    const id = engine.slotNew();
    if (id < 0) throw new HttpError(503, 'no free conversation slot');
    const ctx = { key, id, tokens: [], upperSkill: -1, job: null, queue: [], used: ++tick, biasDirty: false };
    contexts.set(key, ctx);
    return ctx;
  };

  const submit = job => {
    const ctx = contexts.get(job.key);
    if (ctx) { ctx.queue.push(job); ctx.used = ++tick; }
    else if (waiting.some(w => w.key === job.key)) waiting.push(job);
    else {
      const fresh = takeContext(job.key);
      if (fresh) fresh.queue.push(job); else waiting.push(job);
    }
    kick();
  };

  const promoteWaiting = () => {
    for (let i = 0; i < waiting.length;) {
      const job = waiting[i];
      const ctx = contexts.get(job.key) ?? takeContext(job.key);
      if (!ctx) { i++; continue; }
      ctx.queue.push(job);
      waiting.splice(i, 1);
    }
  };

  // ---- the round-robin loop
  let looping = false, cursor = 0;
  const kick = () => { if (!looping) { looping = true; setImmediate(loop); } };

  function loop() {
    const active = [...contexts.values()].filter(c => c.job || c.queue.length);
    if (active.length === 0) { looping = false; return; }
    const ctx = active[cursor++ % active.length];
    if (!ctx.job) {
      ctx.job = ctx.queue.shift();
      ctx.job.gen = ctx.job.run(ctx);
    }
    const job = ctx.job;
    let finished = false;
    try {
      finished = job.gen.next().done;
    } catch (e) {
      finished = true;
      // the cache may be half-changed: start this conversation over rather than trust it
      engine.slotUse(ctx.id);
      engine.reset();
      ctx.tokens = [];
      job.fail(e);
    }
    if (finished) {
      ctx.job = null;
      ctx.used = ++tick;
      promoteWaiting();
    }
    setImmediate(loop);
  }

  // ---- pieces of work, shared by the three kinds of job
  const now = () => performance.now();

  const encode = text => tokenizer.encode(text);

  const fit = (tokens, reserve = 0) => {
    if (tokens.length === 0) throw bad('the prompt is empty');
    if (tokens.length + reserve > engine.maxCtx) {
      throw bad(`the prompt has ${tokens.length} tokens but the context holds ${engine.maxCtx}`);
    }
  };

  // Keeps the cached tokens that agree with the new prompt (never all of it: the last token is always evaluated, so
  // there are logits and a head state for the position the answer follows). Returns how many were kept.
  const reuse = (ctx, tokens) => {
    const limit = Math.min(ctx.tokens.length, tokens.length - 1);
    let p = 0;
    while (p < limit && ctx.tokens[p] === tokens[p]) p++;
    engine.slotUse(ctx.id);
    if (p < ctx.tokens.length) { engine.truncate(p); ctx.tokens.length = p; }
    return p;
  };

  function* lowerLayers(job, ctx, tokens, from) {
    for (let i = from; i < tokens.length; i += PREFILL_PIECE) {
      if (job.aborted) return;
      const piece = tokens.slice(i, i + PREFILL_PIECE);
      const t0 = now();
      engine.slotUse(ctx.id);
      engine.evalLower(piece);
      job.timings.lower_ms += now() - t0;
      for (const t of piece) ctx.tokens.push(t);
      yield;
    }
  }

  const readControl = ctx => {
    if (!head) return null;
    engine.slotUse(ctx.id);
    const out = engine.control();
    return out && { names: head, logits: Array.from(out) };
  };

  // "auto": the loaded skill the head likes best; outputs of the head that name no loaded skill are ignored.
  const chooseSkill = control => {
    let best = null, bestValue = -Infinity;
    control.names.forEach((name, i) => {
      if (skills.has(name) && control.logits[i] > bestValue) { best = name; bestValue = control.logits[i]; }
    });
    return best;
  };

  // Upper layers for everything that still lacks them. Positions that were built under another skill are redone
  // first and counted as switch time.
  function* upperLayers(job, ctx, skill) {
    const index = skill === null ? -1 : skills.get(skill);
    if (index === undefined) throw new HttpError(409, `skill "${skill}" was unloaded`);
    engine.slotUse(ctx.id);
    engine.selectSkill(index);
    const redo = engine.nUpper > 0 && ctx.upperSkill !== index ? engine.nUpper : 0;
    let at = redo ? 0 : engine.nUpper;
    for (;;) {
      if (job.aborted) return false;
      engine.slotUse(ctx.id);
      const t0 = now();
      const left = engine.evalUpper(Math.min(PREFILL_PIECE, at < redo ? redo - at : PREFILL_PIECE));
      const dt = now() - t0;
      if (at < redo) job.timings.switch_ms += dt; else job.timings.upper_ms += dt;
      ctx.upperSkill = index;
      at = engine.nUpper;
      if (left === 0) return true;
      yield;
    }
  }

  const finishTimings = (job, newTokens) => {
    const t = job.timings, prefill = t.lower_ms + t.upper_ms;
    t.prompt_per_second = prefill > 0 ? newTokens / (prefill / 1000) : 0;
    t.predicted_per_second = t.decode_ms > 0 ? job.completion / (t.decode_ms / 1000) : 0;
    for (const k of ['lower_ms', 'upper_ms', 'switch_ms', 'decode_ms']) t[k] = Math.round(t[k] * 100) / 100;
    t.prompt_per_second = Math.round(t.prompt_per_second * 100) / 100;
    t.predicted_per_second = Math.round(t.predicted_per_second * 100) / 100;
    return t;
  };

  const newJob = key => ({
    key, aborted: false, completion: 0, gen: null,
    timings: { lower_ms: 0, upper_ms: 0, switch_ms: 0, decode_ms: 0 },
  });

  // Resolves the requested skill after the lower layers ran: a name, null for none, or the head's pick for "auto".
  const resolveSkill = (spec, control) => {
    if (spec === 'auto') return control ? chooseSkill(control) : null;
    return spec === 'none' ? null : spec;
  };

  const checkSkillSpec = spec => {
    if (spec === 'auto') {
      if (!head) throw bad('skill "auto" needs a control head');
    } else if (spec !== 'none' && !skills.has(spec)) {
      throw bad(`no skill named "${spec}"`);
    }
  };

  // ---- job: /generate
  function generateJob(body, res) {
    const prompt = field(body, 'prompt', 'string');
    if (prompt === undefined) throw bad('"prompt" is required');
    const spec = field(body, 'skill', 'string', 'none');
    checkSkillSpec(spec);
    const stream = field(body, 'stream', 'boolean', false);
    const maxTokens = number(body, 'max_tokens', 256, 1, 1 << 20, true);
    const sampling = {
      temperature: number(body, 'temperature', 0.7, 0, 100),
      topK: number(body, 'top_k', 20, 0, 1 << 20, true),
      topP: number(body, 'top_p', 0.9, 0, 1),
      minP: number(body, 'min_p', 0, 0, 1),
      repeatPenalty: number(body, 'repeat_penalty', 1, 0.01, 100),
      window: 256,
    };
    const seed = field(body, 'seed', 'number');
    const stops = field(body, 'stop', 'object', []);
    if (!Array.isArray(stops) || stops.some(s => typeof s !== 'string' || s === '')) throw bad('"stop" must be a list of non-empty strings');
    const banList = field(body, 'ban', 'object', []);
    if (!Array.isArray(banList) || banList.some(s => typeof s !== 'string')) throw bad('"ban" must be a list of strings');
    // a ban entry is a token of the vocabulary (as written in tokenizer.json) or a text that is exactly one token
    const banned = banList.map(text => {
      const id = tokenizer.tokenToId(text);
      if (id !== undefined) return id;
      const ids = encode(text);
      if (ids.length !== 1) throw bad(`ban entry ${JSON.stringify(text)} is not a single token`);
      return ids[0];
    });
    const tokens = encode(prompt);
    fit(tokens, 1);

    const job = newJob(field(body, 'slot', 'string', 'default'));
    job.fail = e => {
      const status = e instanceof HttpError ? e.status : 500;
      if (!res.headersSent) send(res, status, { error: e.message });
      else if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ done: true, finish: 'error', error: e.message })}\n\ndata: [DONE]\n\n`);
        res.end();
      }
      if (!(e instanceof HttpError)) console.error(e);
    };
    res.on('close', () => { if (!res.writableEnded) job.aborted = true; });

    job.run = function* (ctx) {
      if (job.aborted) return;
      const cached = reuse(ctx, tokens);
      yield* lowerLayers(job, ctx, tokens, cached);
      if (job.aborted) return;
      const control = readControl(ctx);
      const skill = resolveSkill(spec, control);
      const ready = yield* upperLayers(job, ctx, skill);
      if (!ready) return;

      engine.slotUse(ctx.id);
      const bias = engine.bias;
      if (ctx.biasDirty) bias.fill(0);
      for (const id of banned) bias[id] = -1e30;
      ctx.biasDirty = banned.length > 0;
      engine.seed(seed === undefined ? crypto.randomInt(0, 2 ** 32) : seed);

      if (stream) res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache' });
      const decoder = tokenizer.createStreamDecoder({ skipSpecial: true });
      const filter = new StopFilter(stops);
      let text = '', finish = null;
      const emit = piece => {
        if (!piece) return;
        if (stream) res.write(`data: ${JSON.stringify({ text: piece })}\n\n`);
        else text += piece;
      };
      const room = engine.maxCtx - tokens.length;
      while (finish === null) {
        for (let n = 0; n < DECODE_TURN && finish === null; n++) {
          if (job.aborted) { finish = 'abort'; break; }
          const t0 = now();
          engine.slotUse(ctx.id);
          const token = engine.sample(sampling);
          if (eosIds.has(token)) { finish = 'stop'; job.timings.decode_ms += now() - t0; break; }
          job.completion++;
          emit(filter.push(decoder.push(token)));
          if (filter.hit) finish = 'stop';
          else if (job.completion >= maxTokens || job.completion >= room) finish = 'length';
          else {
            engine.eval([token]);
            ctx.tokens.push(token);
          }
          job.timings.decode_ms += now() - t0;
        }
        if (finish === null) yield;
      }
      if (finish === 'abort') return;
      if (!filter.hit) emit(filter.flush() + decoder.flush());

      const done = {
        finish, skill,
        control,
        usage: { prompt_tokens: tokens.length, cached_tokens: cached, completion_tokens: job.completion },
        timings: finishTimings(job, tokens.length - cached),
      };
      if (stream) {
        res.write(`data: ${JSON.stringify({ done: true, ...done })}\n\ndata: [DONE]\n\n`);
        res.end();
      } else {
        send(res, 200, { text, ...done });
      }
    };
    return job;
  }

  // ---- job: /control (lower layers only) and /warm (whole prompt into the cache)
  function promptJob(body, res, kind) {
    const prompt = field(body, 'prompt', 'string');
    if (prompt === undefined) throw bad('"prompt" is required');
    let spec = 'none';
    if (kind === 'warm') { spec = field(body, 'skill', 'string', 'none'); checkSkillSpec(spec); }
    if (kind === 'control' && !head) throw bad('no control head loaded');
    const tokens = encode(prompt);
    fit(tokens);
    const job = newJob(field(body, 'slot', 'string', 'default'));
    job.fail = e => {
      send(res, e instanceof HttpError ? e.status : 500, { error: e.message });
      if (!(e instanceof HttpError)) console.error(e);
    };
    res.on('close', () => { if (!res.writableEnded) job.aborted = true; });
    job.run = function* (ctx) {
      if (job.aborted) return;
      const cached = reuse(ctx, tokens);
      yield* lowerLayers(job, ctx, tokens, cached);
      if (job.aborted) return;
      const control = readControl(ctx);
      const usage = { prompt_tokens: tokens.length, cached_tokens: cached, completion_tokens: 0 };
      if (kind === 'control') {
        send(res, 200, { names: control?.names ?? head, logits: control?.logits ?? null, usage });
        return;
      }
      const skill = resolveSkill(spec, control);
      if (!(yield* upperLayers(job, ctx, skill))) return;
      send(res, 200, { skill, usage, timings: finishTimings(job, tokens.length - cached) });
    };
    return job;
  }

  // ---- HTTP
  function send(res, status, object) {
    const data = JSON.stringify(object);
    res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'content-length': Buffer.byteLength(data) });
    res.end(data);
  }

  const readJson = req => new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', chunk => {
      size += chunk.length;
      if (size > MAX_BODY) { reject(new HttpError(413, 'request body is too large')); req.destroy(); return; }
      chunks.push(chunk);
    });
    req.on('end', () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
        if (body === null || typeof body !== 'object' || Array.isArray(body)) throw new Error();
        resolve(body);
      } catch { reject(bad('the body must be a JSON object')); }
    });
    req.on('error', reject);
  });

  const health = () => ({
    ok: true,
    variant: engine.variant,
    threads: engine.threads,
    ctx: engine.maxCtx,
    skills: skillNames(),
    head,
    slots: [...contexts.values()].map(c => ({
      key: c.key, n_past: engine.slotNPast(c.id), bytes: engine.slotBytes(c.id),
      busy: !idle(c), skill: skillName(engine.slotSkill(c.id)),
    })),
    memory_bytes: engine.memory.byteLength,
    rss_bytes: process.memoryUsage().rss,
  });

  const routes = {
    'GET /health': async () => health(),
    'POST /tokenize': async body => {
      const text = field(body, 'text', 'string');
      if (text === undefined) throw bad('"text" is required');
      const tokens = encode(text);
      return { tokens, count: tokens.length };
    },
    'POST /skills': async body => {
      const load = field(body, 'load', 'object'), unload = field(body, 'unload', 'string');
      if ((load === undefined) === (unload === undefined)) throw bad('give either "load" or "unload"');
      if (load !== undefined) {
        if (load === null || typeof load.name !== 'string' || !load.name || typeof load.path !== 'string') {
          throw bad('"load" needs a name and a path');
        }
        try { loadSkill(load.name, load.path); } catch (e) { throw e instanceof HttpError ? e : bad(e.message); }
      } else {
        unloadSkill(unload);
      }
      return { skills: skillNames() };
    },
  };

  const server = http.createServer(async (req, res) => {
    try {
      const given = crypto.createHash('sha256').update(String(req.headers.authorization ?? '')).digest();
      if (!crypto.timingSafeEqual(given, keyDigest)) throw new HttpError(403, 'forbidden');
      const route = `${req.method} ${new URL(req.url, 'http://localhost').pathname}`;
      const body = req.method === 'POST' ? await readJson(req) : {};
      if (route === 'POST /generate') return submit(generateJob(body, res));
      if (route === 'POST /control') return submit(promptJob(body, res, 'control'));
      if (route === 'POST /warm') return submit(promptJob(body, res, 'warm'));
      const handler = routes[route];
      if (!handler) throw new HttpError(404, 'not found');
      send(res, 200, await handler(body));
    } catch (e) {
      if (!res.headersSent) send(res, e instanceof HttpError ? e.status : 500, { error: e.message });
      if (!(e instanceof HttpError)) console.error(e);
    }
  });

  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(opts.port, '127.0.0.1', resolve); });

  // The parent keeps a pipe open to our stdin; when it closes, the app is gone and so are we.
  const shutdown = () => { engine.close(); process.exit(0); };
  process.stdin.on('end', shutdown);
  process.stdin.on('close', shutdown);
  process.stdin.resume();
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);

  console.log(JSON.stringify({
    ready: true, port: server.address().port, pid: process.pid, threads: engine.threads, variant: engine.variant,
    skills: skillNames(), ctx: engine.maxCtx, slots: maxSlots,
  }));
  console.error(`kiwi mind ready in ${Math.round(performance.now() - started)} ms`);
}

let opts;
try {
  opts = parseArgs(process.argv.slice(2));
} catch (e) {
  console.error(`${e.message}\n${USAGE}`);
  process.exit(2);
}
main(opts).catch(e => {
  console.error(e.message ?? e);
  process.exit(1);
});
