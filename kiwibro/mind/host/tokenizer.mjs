// Derived from the Flyingkiwi tokenizer by the same author.
//
// Dependency-free port of the Hugging Face `tokenizers` pipeline used by Qwen2/Qwen3 tokenizer.json files
// (ByteLevel BPE):
//
//   added-token split -> normalizer (NFC) -> Split(Isolated, regex) -> ByteLevel -> BPE
//
// plus the matching ByteLevel decoder. No imports, no DOM, no eval. Anything outside this pipeline (template
// post-processing, dropout, byte fallback, fuse_unk, truncation, non-ByteLevel decoders) is rejected by
// fromJSON instead of being approximated.
//
// Known limitation: NFC and the \p{..} classes come from the JavaScript engine's Unicode data.
// For a small set of combining marks, tokenizers' bundled data differs (154 of about 1.1 million
// code points in a three-context scan; no single character differs). Text that combines those
// marks can tokenize differently from the Python reference.

const CACHE_CAPACITY = 20000;
const PRIVATE = Symbol('mind-tokenizer');
const ENCODER = new TextEncoder();
const NORMALIZER_FORMS = new Set(['NFC', 'NFD', 'NFKC', 'NFKD']);

// Rust regex / Oniguruma `\s` is Unicode White_Space. JavaScript `\s` differs: it adds
// U+FEFF and omits U+0085. Every whitespace class is spelled out explicitly.
const WS_BODY = '\\t\\n\\x0B\\f\\r \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000';
const WS_CHAR = new RegExp(`^[${WS_BODY}]$`, 'u');
// Rust `\w`, used by added-token `single_word`. Checked against tokenizers 0.22.2:
// U+00B2 (No) is not a word character, `_` and combining marks are.
const WORD_CHAR = /^[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]$/u;

// GPT-2 bytes_to_unicode: every byte maps to one printable character.
function buildByteTables() {
  const bytes = [];
  for (let b = 0x21; b <= 0x7e; b++) bytes.push(b);
  for (let b = 0xa1; b <= 0xac; b++) bytes.push(b);
  for (let b = 0xae; b <= 0xff; b++) bytes.push(b);
  const codes = bytes.slice();
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      codes.push(256 + extra);
      extra++;
    }
  }
  const byteToChar = new Array(256);
  const charToByte = new Map();
  for (let i = 0; i < 256; i++) {
    const ch = String.fromCodePoint(codes[i]);
    byteToChar[bytes[i]] = ch;
    charToByte.set(ch, bytes[i]);
  }
  return { byteToChar, charToByte };
}

const { byteToChar: BYTE_TO_CHAR, charToByte: CHAR_TO_BYTE } = buildByteTables();

function unsupported(what) {
  return new Error(`mind-tokenizer: unsupported tokenizer feature: ${what}`);
}

function byteLevelString(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += BYTE_TO_CHAR[bytes[i]];
  return s;
}

// Bytes of a token string for decoding. Characters outside the byte table pass through
// as their UTF-8 bytes, which is what tokenizers' ByteLevel decoder does.
function tokenBytes(token) {
  const out = [];
  for (const ch of token) {
    const b = CHAR_TO_BYTE.get(ch);
    if (b !== undefined) {
      out.push(b);
    } else {
      for (const x of ENCODER.encode(ch)) out.push(x);
    }
  }
  return out;
}

// Translate the Oniguruma/Rust Split pattern into an equivalent JavaScript source.
// Supported: (?i:...) over ASCII letters, (?:..), (?=..), (?!..), (?<=..), (?<!..),
// \s \S \p{..} \P{..} \r \n \t \f \v and escaped punctuation, classes, quantifiers, |.
// Anything else throws so an unknown construct can never be silently mistranslated.
function translateEscape(source, i, inClass) {
  const d = source[i + 1];
  if (d === undefined) throw new Error('mind-tokenizer: dangling backslash in Split pattern');
  switch (d) {
    case 's':
      return { text: inClass ? WS_BODY : `[${WS_BODY}]`, next: i + 2 };
    case 'S':
      if (inClass) throw unsupported('\\S inside a character class');
      return { text: `[^${WS_BODY}]`, next: i + 2 };
    case 'p':
    case 'P': {
      if (source[i + 2] !== '{') throw unsupported(`\\${d} without {name}`);
      const close = source.indexOf('}', i + 3);
      if (close < 0) throw unsupported(`unterminated \\${d}{`);
      return { text: source.slice(i, close + 1), next: close + 1 };
    }
    case 'r':
    case 'n':
    case 't':
    case 'f':
    case 'v':
      return { text: `\\${d}`, next: i + 2 };
    default:
      if ('\\.^$*+?()[]{}|/-'.includes(d)) return { text: `\\${d}`, next: i + 2 };
      throw unsupported(`escape \\${d} in Split pattern`);
  }
}

function translateSplitPattern(source) {
  let out = '';
  let i = 0;
  let inClass = false;
  let caseDepth = 0;
  const groups = []; // one entry per open group: true when the group is (?i:
  while (i < source.length) {
    const c = source[i];
    if (c === '\\') {
      const esc = translateEscape(source, i, inClass);
      out += esc.text;
      i = esc.next;
      continue;
    }
    if (inClass) {
      if (c === ']') {
        inClass = false;
      } else if (c === '[' || source.startsWith('&&', i)) {
        throw unsupported('nested character class or set operation in Split pattern');
      } else if (caseDepth > 0 && /[A-Za-z]/.test(c)) {
        throw unsupported('letters inside a case-insensitive character class');
      }
      out += c;
      i++;
      continue;
    }
    if (c === '[') {
      inClass = true;
      out += '[';
      i++;
      if (source[i] === '^') {
        out += '^';
        i++;
      }
      continue;
    }
    if (c === '(') {
      if (source.startsWith('(?i:', i)) {
        groups.push(true);
        caseDepth++;
        out += '(?:';
        i += 4;
        continue;
      }
      const m = /^\(\?(?::|=|!|<=|<!)/.exec(source.slice(i, i + 4));
      if (m) {
        groups.push(false);
        out += m[0];
        i += m[0].length;
      } else if (source[i + 1] === '?') {
        throw unsupported(`group syntax ${source.slice(i, i + 4)} in Split pattern`);
      } else {
        groups.push(false);
        out += '(';
        i++;
      }
      continue;
    }
    if (c === ')') {
      if (groups.length === 0) throw new Error('mind-tokenizer: unbalanced ) in Split pattern');
      if (groups.pop()) caseDepth--;
      out += ')';
      i++;
      continue;
    }
    if (c === '.') throw unsupported("'.' (line terminators differ between Oniguruma and JavaScript)");
    if (caseDepth > 0 && /[A-Za-z]/.test(c)) {
      out += `[${c.toLowerCase()}${c.toUpperCase()}]`;
    } else if (caseDepth > 0 && c.charCodeAt(0) > 0x7f) {
      throw unsupported('non-ASCII literal inside a case-insensitive group');
    } else {
      out += c;
    }
    i++;
  }
  if (inClass || groups.length > 0) throw new Error('mind-tokenizer: unbalanced Split pattern');
  return out;
}

// Split with behavior "Isolated": every match becomes a piece and so does every gap.
function splitIsolated(regex, text) {
  const pieces = [];
  let last = 0;
  for (const m of text.matchAll(regex)) {
    if (m[0].length === 0) continue;
    if (m.index > last) pieces.push(text.slice(last, m.index));
    pieces.push(m[0]);
    last = m.index + m[0].length;
  }
  if (last < text.length) pieces.push(text.slice(last));
  return pieces;
}

function codePointBefore(s, index) {
  const lo = s.charCodeAt(index - 1);
  if (lo >= 0xdc00 && lo <= 0xdfff && index >= 2) {
    const hi = s.charCodeAt(index - 2);
    if (hi >= 0xd800 && hi <= 0xdbff) return s.slice(index - 2, index);
  }
  return s.slice(index - 1, index);
}

function codePointAt(s, index) {
  return String.fromCodePoint(s.codePointAt(index));
}

// Added tokens in one raw sentence, matched leftmost-longest (alternatives are sorted by
// length). Returns segments of { text } or { id }. Honours single_word, lstrip and rstrip
// as tokenizers does, verified against tokenizers 0.22.2.
function splitOnAddedTokens(sentence, pass) {
  if (sentence.length === 0) return [];
  if (pass === null) return [{ text: sentence }];
  const { matcher, byContent } = pass;
  const segments = [];
  matcher.lastIndex = 0;
  let startOffset = 0;
  let m;
  while ((m = matcher.exec(sentence)) !== null) {
    const token = byContent.get(m[0]);
    let start = m.index;
    let stop = m.index + m[0].length;
    if (token.singleWord) {
      const startOk = start === 0 || !WORD_CHAR.test(codePointBefore(sentence, start));
      const stopOk = stop === sentence.length || !WORD_CHAR.test(codePointAt(sentence, stop));
      if (!startOk || !stopOk) continue;
    }
    if (token.lstrip) {
      while (start > startOffset && WS_CHAR.test(sentence[start - 1])) start--;
    }
    if (token.rstrip) {
      while (stop < sentence.length && WS_CHAR.test(sentence[stop])) stop++;
    }
    if (startOffset < start) segments.push({ text: sentence.slice(startOffset, start) });
    segments.push({ id: token.id });
    startOffset = Math.max(startOffset, stop);
  }
  if (startOffset < sentence.length) segments.push({ text: sentence.slice(startOffset) });
  return segments;
}

function buildAddedPass(tokens) {
  if (tokens.length === 0) return null;
  const byContent = new Map();
  for (const token of tokens) if (!byContent.has(token.content)) byContent.set(token.content, token);
  const alternatives = [...byContent.values()]
    .sort((a, b) => b.content.length - a.content.length)
    .map((token) => token.content.replace(/[\\^$.*+?()[\]{}|/]/g, '\\$&'));
  return { matcher: new RegExp(alternatives.join('|'), 'gu'), byContent };
}

// Binary min-heap ordered by (rank, position), as tokenizers' BPE merge queue.
function before(a, b) {
  return a.rank < b.rank || (a.rank === b.rank && a.pos < b.pos);
}

class MergeQueue {
  constructor() {
    this.items = [];
  }

  get size() {
    return this.items.length;
  }

  push(rank, pos, newId) {
    const items = this.items;
    const item = { rank, pos, newId };
    let i = items.length;
    items.push(item);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!before(item, items[parent])) break;
      items[i] = items[parent];
      i = parent;
    }
    items[i] = item;
  }

  pop() {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length > 0) {
      const n = items.length;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        let best = last;
        if (l < n && before(items[l], best)) {
          m = l;
          best = items[l];
        }
        if (r < n && before(items[r], best)) {
          m = r;
          best = items[r];
        }
        if (m === i) break;
        items[i] = items[m];
        i = m;
      }
      items[i] = last;
    }
    return top;
  }
}

// Mirrors tokenizers' Word::merge_all: lowest rank first, then leftmost position, with the
// same stale-entry validation. Symbols that cannot be mapped are dropped before merging.
function mergeSymbols(symbols, table) {
  const n = symbols.length;
  if (n < 2) return symbols.slice();
  const { pairRank, rankNewId, base } = table;
  const c = symbols.slice();
  const prev = new Int32Array(n);
  const next = new Int32Array(n);
  const alive = new Uint8Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    prev[i] = i - 1;
    next[i] = i + 1 < n ? i + 1 : -1;
  }
  const queue = new MergeQueue();
  for (let i = 0; i + 1 < n; i++) {
    const rank = pairRank.get(c[i] * base + c[i + 1]);
    if (rank !== undefined) queue.push(rank, i, rankNewId[rank]);
  }
  while (queue.size > 0) {
    const top = queue.pop();
    const pos = top.pos;
    if (alive[pos] === 0) continue;
    const right = next[pos];
    if (right === -1) continue;
    const rank = pairRank.get(c[pos] * base + c[right]);
    if (rank === undefined || rankNewId[rank] !== top.newId) continue;
    c[pos] = top.newId;
    alive[right] = 0;
    const after = next[right];
    next[pos] = after;
    if (after !== -1) prev[after] = pos;
    const left = prev[pos];
    if (left !== -1) {
      const r = pairRank.get(c[left] * base + c[pos]);
      if (r !== undefined) queue.push(r, left, rankNewId[r]);
    }
    const nxt = next[pos];
    if (nxt !== -1) {
      const r = pairRank.get(c[pos] * base + c[nxt]);
      if (r !== undefined) queue.push(r, pos, rankNewId[r]);
    }
  }
  const out = [];
  for (let i = 0; i !== -1; i = next[i]) out.push(c[i]);
  return out;
}

function parseMergeEntry(entry, rank) {
  if (typeof entry === 'string') {
    const parts = entry.split(' ');
    if (parts.length === 2) return parts;
  } else if (Array.isArray(entry) && entry.length === 2 && typeof entry[0] === 'string' && typeof entry[1] === 'string') {
    return entry;
  }
  throw new Error(`mind-tokenizer: merge ${rank} is neither "left right" nor [left, right]`);
}

function parseNormalizer(normalizer) {
  if (normalizer === null || normalizer === undefined) return null;
  if (NORMALIZER_FORMS.has(normalizer.type)) return normalizer.type;
  throw unsupported(`normalizer ${normalizer.type}`);
}

function parsePreTokenizer(preTokenizer) {
  const items = preTokenizer?.type === 'Sequence' ? preTokenizer.pretokenizers : [preTokenizer];
  if (!Array.isArray(items) || items.length === 0) throw unsupported('pre_tokenizer');
  const last = items[items.length - 1];
  if (last?.type !== 'ByteLevel') throw unsupported('pre_tokenizer must end with ByteLevel');
  if (last.use_regex || last.add_prefix_space) throw unsupported('ByteLevel use_regex / add_prefix_space');
  return items.slice(0, -1).map((item) => {
    if (item?.type !== 'Split') throw unsupported(`pre_tokenizer ${item?.type}`);
    if (item.behavior !== 'Isolated') throw unsupported(`Split behavior ${item.behavior}`);
    if (item.invert) throw unsupported('Split.invert');
    if (typeof item.pattern?.Regex !== 'string') throw unsupported('Split pattern other than Regex');
    return new RegExp(translateSplitPattern(item.pattern.Regex), 'gu');
  });
}

function parsePostProcessor(postProcessor) {
  if (postProcessor === null || postProcessor === undefined) return;
  if (postProcessor.type !== 'ByteLevel' || postProcessor.add_prefix_space) {
    throw unsupported(`post_processor ${postProcessor.type}`);
  }
}

function parseDecoder(decoder) {
  if (decoder?.type !== 'ByteLevel') throw unsupported(`decoder ${decoder?.type}`);
}

function normalizeAddedToken(raw) {
  if (!raw || typeof raw.content !== 'string' || raw.content.length === 0) {
    throw new Error('mind-tokenizer: added token needs a non-empty content string');
  }
  if (!Number.isSafeInteger(raw.id) || raw.id < 0) {
    throw new Error(`mind-tokenizer: added token ${JSON.stringify(raw.content)} has no valid id`);
  }
  const special = raw.special === true;
  return {
    id: raw.id,
    content: raw.content,
    special,
    lstrip: raw.lstrip === true,
    rstrip: raw.rstrip === true,
    singleWord: raw.single_word === true,
    normalized: typeof raw.normalized === 'boolean' ? raw.normalized : !special,
  };
}

export class BpeTokenizer {
  #vocab;
  #idToToken;
  #addedById;
  #addedByContent;
  #vocabSize;
  #normForm;
  #splits;
  #rawPass;
  #normPass;
  #byteIds;
  #unkId;
  #ignoreMerges;
  #merge;
  #cache = new Map();

  constructor(token, state) {
    if (token !== PRIVATE) throw new TypeError('use BpeTokenizer.fromJSON(tokenizerJson)');
    this.#vocab = state.vocab;
    this.#idToToken = state.idToToken;
    this.#addedById = state.addedById;
    this.#addedByContent = state.addedByContent;
    this.#vocabSize = state.vocabSize;
    this.#normForm = state.normForm;
    this.#splits = state.splits;
    this.#rawPass = state.rawPass;
    this.#normPass = state.normPass;
    this.#byteIds = state.byteIds;
    this.#unkId = state.unkId;
    this.#ignoreMerges = state.ignoreMerges;
    this.#merge = state.merge;
  }

  // Build from a parsed tokenizer.json. Throws for features outside the supported pipeline.
  static fromJSON(obj) {
    if (obj === null || typeof obj !== 'object') throw new TypeError('fromJSON expects a parsed tokenizer.json object');
    const model = obj.model;
    if (model?.type !== 'BPE') throw unsupported(`model.type ${JSON.stringify(model?.type)} (only BPE)`);
    if (model.dropout) throw unsupported('model.dropout');
    if (model.byte_fallback) throw unsupported('model.byte_fallback');
    if (model.fuse_unk) throw unsupported('model.fuse_unk');
    if (model.continuing_subword_prefix || model.end_of_word_suffix) throw unsupported('subword prefix/suffix');
    if (obj.truncation) throw unsupported('truncation');

    const normForm = parseNormalizer(obj.normalizer);
    const splits = parsePreTokenizer(obj.pre_tokenizer);
    parsePostProcessor(obj.post_processor);
    parseDecoder(obj.decoder);

    if (!model.vocab || typeof model.vocab !== 'object') throw new Error('mind-tokenizer: model.vocab missing');
    const vocab = new Map();
    const idToToken = new Map();
    let maxId = -1;
    for (const token of Object.keys(model.vocab)) {
      const id = model.vocab[token];
      if (!Number.isSafeInteger(id) || id < 0) throw new Error(`mind-tokenizer: vocab id of ${JSON.stringify(token)} is invalid`);
      if (idToToken.has(id)) throw new Error(`mind-tokenizer: duplicate vocab id ${id}`);
      vocab.set(token, id);
      idToToken.set(id, token);
      if (id > maxId) maxId = id;
    }

    const base = maxId + 1;
    if (base * base > Number.MAX_SAFE_INTEGER) throw new Error('mind-tokenizer: vocab ids too large');
    const merges = model.merges ?? [];
    const pairRank = new Map();
    const rankNewId = new Array(merges.length);
    merges.forEach((entry, rank) => {
      const [left, right] = parseMergeEntry(entry, rank);
      const a = vocab.get(left);
      const b = vocab.get(right);
      const merged = vocab.get(left + right);
      if (a === undefined || b === undefined || merged === undefined) {
        throw new Error(`mind-tokenizer: merge ${rank} references a token missing from the vocabulary`);
      }
      pairRank.set(a * base + b, rank);
      rankNewId[rank] = merged;
    });

    let unkId = null;
    if (model.unk_token !== null && model.unk_token !== undefined) {
      unkId = vocab.get(model.unk_token);
      if (unkId === undefined) throw new Error('mind-tokenizer: unk_token is not in the vocabulary');
    }
    const byteIds = new Int32Array(256);
    for (let b = 0; b < 256; b++) {
      const id = vocab.get(BYTE_TO_CHAR[b]);
      byteIds[b] = id === undefined ? -1 : id;
    }

    const addedList = [];
    const addedById = new Map();
    const addedByContent = new Map();
    for (const raw of obj.added_tokens ?? []) {
      const token = normalizeAddedToken(raw);
      if (addedById.has(token.id)) throw new Error(`mind-tokenizer: duplicate added token id ${token.id}`);
      addedById.set(token.id, token);
      if (!addedByContent.has(token.content)) addedByContent.set(token.content, token);
      addedList.push(token);
      if (token.id > maxId) maxId = token.id;
    }

    const state = {
      vocab,
      idToToken,
      addedById,
      addedByContent,
      vocabSize: maxId + 1,
      normForm,
      splits,
      rawPass: buildAddedPass(addedList.filter((t) => !t.normalized)),
      normPass: buildAddedPass(addedList.filter((t) => t.normalized)),
      byteIds,
      unkId,
      ignoreMerges: model.ignore_merges === true,
      merge: { pairRank, rankNewId, base },
    };
    return new BpeTokenizer(PRIVATE, state);
  }

  get vocabSize() {
    return this.#vocabSize;
  }

  tokenToId(token) {
    const added = this.#addedByContent.get(token);
    if (added !== undefined) return added.id;
    return this.#vocab.get(token);
  }

  idToToken(id) {
    const added = this.#addedById.get(id);
    if (added !== undefined) return added.content;
    return this.#idToToken.get(id);
  }

  encode(text, { addSpecial = true } = {}) {
    if (typeof text !== 'string') throw new TypeError('encode expects a string');
    const ids = [];
    for (const segment of this.#segments(text, addSpecial)) {
      if (segment.id !== undefined) {
        ids.push(segment.id);
        continue;
      }
      for (const piece of this.#preTokenize(segment.text)) {
        for (const id of this.#pieceIds(piece)) ids.push(id);
      }
    }
    return ids;
  }

  decode(ids, { skipSpecial = false } = {}) {
    const bytes = [];
    for (const id of ids) {
      const part = this.#idBytes(id, skipSpecial);
      if (part !== null) for (const b of part) bytes.push(b);
    }
    return new TextDecoder('utf-8', { ignoreBOM: true }).decode(Uint8Array.from(bytes));
  }

  // Raw bytes of one token, or null for a special (control) token. Used for byte-level grammar checks.
  tokenBytes(id) {
    const bytes = this.#idBytes(id, true);
    return bytes === null ? null : Uint8Array.from(bytes);
  }

  createStreamDecoder({ skipSpecial = false } = {}) {
    const decoder = new TextDecoder('utf-8', { ignoreBOM: true });
    return {
      push: (id) => {
        const part = this.#idBytes(id, skipSpecial);
        if (part === null || part.length === 0) return '';
        return decoder.decode(Uint8Array.from(part), { stream: true });
      },
      flush: () => decoder.decode(),
    };
  }

  #idBytes(id, skipSpecial) {
    if (!Number.isInteger(id)) throw new TypeError(`token id must be an integer, got ${id}`);
    const added = this.#addedById.get(id);
    if (added !== undefined) {
      if (skipSpecial && added.special) return null;
      return tokenBytes(added.content);
    }
    const token = this.#idToToken.get(id);
    return token === undefined ? null : tokenBytes(token);
  }

  #normalize(text) {
    return this.#normForm === null ? text : text.normalize(this.#normForm);
  }

  #segments(text, addSpecial) {
    if (!addSpecial) return [{ text: this.#normalize(text) }];
    const out = [];
    for (const segment of splitOnAddedTokens(text, this.#rawPass)) {
      if (segment.id !== undefined) {
        out.push(segment);
        continue;
      }
      const normalized = this.#normalize(segment.text);
      if (this.#normPass === null) {
        if (normalized.length > 0) out.push({ text: normalized });
      } else {
        for (const sub of splitOnAddedTokens(normalized, this.#normPass)) out.push(sub);
      }
    }
    return out;
  }

  #preTokenize(text) {
    let pieces = text.length > 0 ? [text] : [];
    for (const regex of this.#splits) {
      const next = [];
      for (const piece of pieces) for (const sub of splitIsolated(regex, piece)) next.push(sub);
      pieces = next;
    }
    return pieces;
  }

  #pieceIds(piece) {
    const cached = this.#cache.get(piece);
    if (cached !== undefined) {
      this.#cache.delete(piece);
      this.#cache.set(piece, cached);
      return cached;
    }
    const ids = this.#bpe(piece);
    this.#cache.set(piece, ids);
    if (this.#cache.size > CACHE_CAPACITY) this.#cache.delete(this.#cache.keys().next().value);
    return ids;
  }

  #bpe(piece) {
    const bytes = ENCODER.encode(piece);
    if (bytes.length === 0) return [];
    if (this.#ignoreMerges) {
      const direct = this.#vocab.get(byteLevelString(bytes));
      if (direct !== undefined) return [direct];
    }
    const symbols = [];
    for (let i = 0; i < bytes.length; i++) {
      const id = this.#byteIds[bytes[i]];
      if (id >= 0) symbols.push(id);
      else if (this.#unkId !== null) symbols.push(this.#unkId);
    }
    return mergeSymbols(symbols, this.#merge);
  }
}
