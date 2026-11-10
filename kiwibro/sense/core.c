/* Kiwi Sense core: a small reading model that runs beside the language model.
 *
 * Freestanding C, compiled to WebAssembly (see build.sh). The same file drives the
 * server (through a WASM runtime) and the browser. kiwibro/sense/pyengine.py is its
 * line-for-line twin in Python and must give bit-identical results.
 *
 * The host hands in text that is already normalised: lower case, accents removed,
 * tokens (words and single punctuation marks) separated by exactly one space.
 *
 * What lives here:
 *   embed    static word-piece embeddings summed into one unit vector (meaning)
 *   probe    hashed word n-grams + that vector -> a small network -> behaviour classes
 *   repeat   how much of a text is the same four words coming round again
 *   lexsim   byte-trigram overlap of two texts (works in any language)
 */

#include <stddef.h>
#include <stdint.h>

#define EXPORT(name) __attribute__((export_name(#name)))
#define PAGE 65536u
#define FNV_SEED 2166136261u
#define FNV_PRIME 16777619u
#define MAX_DIMS 256
#define MAX_PDIM 64
#define MAX_HIDDEN 128
#define MAX_CLASSES 16
#define MAX_PIECES 100
#define MAX_WORDS 256
#define REPEAT_WORDS 1024
#define REPEAT_SLOTS 4096
#define LEX_BINS 1024

extern unsigned char __heap_base;

void *memset(void *dest, int value, size_t count) {
    unsigned char *p = dest;
    while (count--) *p++ = (unsigned char)value;
    return dest;
}

void *memcpy(void *dest, const void *src, size_t count) {
    unsigned char *a = dest;
    const unsigned char *b = src;
    while (count--) *a++ = *b++;
    return dest;
}

/* ---- memory: one bump allocator; the model data stays, scratch is reset ---- */

static uint32_t heap_top, heap_mark;

static uint32_t align8(uint32_t value) { return (value + 7u) & ~7u; }

EXPORT(ks_alloc) uint32_t ks_alloc(uint32_t size) {
    if (!heap_top) heap_top = align8((uint32_t)(uintptr_t)&__heap_base);
    uint32_t at = heap_top, end = align8(at + size);
    if (end < at) return 0;
    uint32_t have = (uint32_t)__builtin_wasm_memory_size(0) * PAGE;
    if (end > have) {
        uint32_t pages = (end - have + PAGE - 1) / PAGE;
        if (__builtin_wasm_memory_grow(0, pages) == (size_t)-1) return 0;
    }
    heap_top = end;
    return at;
}

EXPORT(ks_reset) void ks_reset(void) {
    if (heap_mark) heap_top = heap_mark;
}

/* ---- reading the data blob ---- */

static uint32_t rd16(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static double rdf32(const uint8_t *p) {
    union { uint32_t bits; float value; } cell;
    cell.bits = rd32(p);
    return (double)cell.value;
}

static const uint8_t *v_slots, *v_offsets, *v_pool;
static uint32_t v_size, v_count, v_pool_len;

static const uint8_t *e_scale;
static const int8_t *e_rows;
static uint32_t e_count, e_dims;

static const int8_t *p_table;
static const uint8_t *p_w1, *p_b1, *p_w2, *p_b2;
static uint32_t p_buckets, p_dim, p_hidden, p_classes;
static double p_scale;

static int tag_is(const uint8_t *p, const char *tag) {
    return p[0] == (uint8_t)tag[0] && p[1] == (uint8_t)tag[1] && p[2] == (uint8_t)tag[2] && p[3] == (uint8_t)tag[3];
}

static int load_vocab(const uint8_t *s, uint32_t len) {
    if (len < 12) return -10;
    uint32_t size = rd32(s), count = rd32(s + 4), pool = rd32(s + 8);
    if (!size || (size & (size - 1)) || count >= size) return -11;
    uint64_t need = 12ull + 4ull * size + 4ull * ((uint64_t)count + 1) + pool;
    if (need > len) return -12;
    v_size = size;
    v_count = count;
    v_pool_len = pool;
    v_slots = s + 12;
    v_offsets = v_slots + 4u * size;
    v_pool = v_offsets + 4u * (count + 1);
    return 0;
}

static int load_embeddings(const uint8_t *s, uint32_t len) {
    if (len < 8) return -20;
    uint32_t count = rd32(s), dims = rd32(s + 4);
    if (!dims || dims > MAX_DIMS) return -21;
    uint64_t need = 8ull + 2ull * count + (uint64_t)count * dims;
    if (need > len) return -22;
    e_count = count;
    e_dims = dims;
    e_scale = s + 8;
    e_rows = (const int8_t *)(e_scale + 2u * count);
    return 0;
}

static int load_probe(const uint8_t *s, uint32_t len) {
    if (len < 24) return -30;
    uint32_t buckets = rd32(s), dim = rd32(s + 4), edim = rd32(s + 8);
    uint32_t hidden = rd32(s + 12), classes = rd32(s + 16);
    if (!buckets || !dim || dim > MAX_PDIM || !hidden || hidden > MAX_HIDDEN) return -31;
    if (!classes || classes > MAX_CLASSES || edim != e_dims) return -32;
    uint64_t inputs = (uint64_t)edim + dim;
    uint64_t need = 24ull + (uint64_t)buckets * dim + 4ull * (hidden * inputs + hidden + (uint64_t)classes * hidden + classes);
    if (need > len) return -33;
    p_buckets = buckets;
    p_dim = dim;
    p_hidden = hidden;
    p_classes = classes;
    p_scale = rdf32(s + 20);
    p_table = (const int8_t *)(s + 24);
    p_w1 = s + 24 + buckets * dim;
    p_b1 = p_w1 + 4u * hidden * (uint32_t)inputs;
    p_w2 = p_b1 + 4u * hidden;
    p_b2 = p_w2 + 4u * classes * hidden;
    return 0;
}

/* Blob: "KSNS", version, section count, then (tag, offset, length) per section. */
EXPORT(ks_init) int32_t ks_init(const uint8_t *blob, uint32_t len) {
    v_size = e_count = e_dims = p_classes = 0;
    if (len < 12 || !tag_is(blob, "KSNS")) return -1;
    if (rd32(blob + 4) != 1) return -2;
    uint32_t sections = rd32(blob + 8);
    if (sections > 16 || 12ull + 12ull * sections > len) return -3;
    for (uint32_t pass = 0; pass < 2; pass++) { /* the probe needs the embedding size first */
        for (uint32_t i = 0; i < sections; i++) {
            const uint8_t *entry = blob + 12 + 12 * i;
            uint32_t offset = rd32(entry + 4), size = rd32(entry + 8);
            if ((uint64_t)offset + size > len) return -4;
            int status = 0;
            if (pass == 0 && tag_is(entry, "VOCB")) status = load_vocab(blob + offset, size);
            else if (pass == 0 && tag_is(entry, "EMBD")) status = load_embeddings(blob + offset, size);
            else if (pass == 1 && tag_is(entry, "PROB")) status = load_probe(blob + offset, size);
            if (status) return status;
        }
    }
    if (!v_size || !e_dims || v_count != e_count) return -5;
    heap_mark = heap_top;
    return 0;
}

EXPORT(ks_version) uint32_t ks_version(void) { return 1; }
EXPORT(ks_dims) uint32_t ks_dims(void) { return e_dims; }
EXPORT(ks_classes) uint32_t ks_classes(void) { return p_classes; }

/* ---- hashing ---- */

static uint32_t fnv(const uint8_t *p, uint32_t n, uint32_t h) {
    for (uint32_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

static uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t h = a ^ (b + 0x9E3779B9u + (a << 6) + (a >> 2));
    h *= FNV_PRIME;
    h ^= h >> 15;
    return h;
}

/* ---- word pieces ---- */

static int32_t lookup(const uint8_t *piece, uint32_t n, int continued) {
    uint32_t h = FNV_SEED;
    if (continued) {
        h = (h ^ '#') * FNV_PRIME;
        h = (h ^ '#') * FNV_PRIME;
    }
    h = fnv(piece, n, h);
    uint32_t mask = v_size - 1, want = n + (continued ? 2u : 0u);
    uint32_t at = h & mask;
    for (uint32_t tries = 0; tries < v_size; tries++, at = (at + 1) & mask) {
        uint32_t slot = rd32(v_slots + 4 * at);
        if (!slot) return -1;
        uint32_t id = slot - 1;
        if (id >= v_count) return -1;
        uint32_t from = rd32(v_offsets + 4 * id), to = rd32(v_offsets + 4 * (id + 1));
        if (to > v_pool_len || from > to || to - from != want) continue;
        const uint8_t *token = v_pool + from;
        if (continued) {
            if (token[0] != '#' || token[1] != '#') continue;
            token += 2;
        }
        uint32_t k = 0;
        while (k < n && token[k] == piece[k]) k++;
        if (k == n) return (int32_t)id;
    }
    return -1;
}

/* Greedy longest-match-first split of one word; 0 pieces means the word is unknown. */
static uint32_t wordpiece(const uint8_t *word, uint32_t n, int32_t *ids) {
    uint32_t chars = 0;
    for (uint32_t i = 0; i < n; i++) {
        if ((word[i] & 0xC0) != 0x80) chars++;
    }
    if (!chars || chars > 100) return 0;
    uint32_t start = 0, count = 0;
    while (start < n) {
        uint32_t end = n;
        int32_t found = -1;
        while (end > start) {
            found = lookup(word + start, end - start, start > 0);
            if (found >= 0) break;
            end--;
            while (end > start && (word[end] & 0xC0) == 0x80) end--;
        }
        if (found < 0 || count >= MAX_PIECES) return 0;
        ids[count++] = found;
        start = end;
    }
    return count;
}

/* ---- meaning: one unit vector per text ---- */

static int64_t e_sum[MAX_DIMS];
static uint32_t last_words, last_known;

static void embed_into(const uint8_t *text, uint32_t len, double *out) {
    int32_t ids[MAX_PIECES];
    for (uint32_t d = 0; d < e_dims; d++) e_sum[d] = 0;
    last_words = last_known = 0;
    uint32_t i = 0;
    while (i < len) {
        while (i < len && text[i] == ' ') i++;
        uint32_t start = i;
        while (i < len && text[i] != ' ') i++;
        if (i == start) break;
        last_words++;
        uint32_t pieces = wordpiece(text + start, i - start, ids);
        if (!pieces) continue;
        last_known++;
        for (uint32_t k = 0; k < pieces; k++) {
            uint32_t id = (uint32_t)ids[k];
            int64_t scale = (int64_t)rd16(e_scale + 2 * id);
            if (!scale) continue;
            const int8_t *row = e_rows + (size_t)id * e_dims;
            for (uint32_t d = 0; d < e_dims; d++) e_sum[d] += (int64_t)row[d] * scale;
        }
    }
    double norm = 0.0;
    for (uint32_t d = 0; d < e_dims; d++) {
        double v = (double)e_sum[d];
        norm += v * v;
    }
    if (norm > 0.0) {
        norm = __builtin_sqrt(norm);
        for (uint32_t d = 0; d < e_dims; d++) out[d] = (double)e_sum[d] / norm;
    } else {
        for (uint32_t d = 0; d < e_dims; d++) out[d] = 0.0;
    }
}

/* Writes ks_dims() doubles to out; returns how many words were recognised. */
EXPORT(ks_embed) uint32_t ks_embed(const uint8_t *text, uint32_t len, double *out) {
    embed_into(text, len, out);
    return last_known;
}

/* Number of words in the text last embedded (known or not). */
EXPORT(ks_words) uint32_t ks_words(void) { return last_words; }

static double vec_a[MAX_DIMS], vec_b[MAX_DIMS];

/* Cosine of the two texts' vectors; 0 when either has no known word. */
EXPORT(ks_similarity) double ks_similarity(const uint8_t *a, uint32_t alen, const uint8_t *b, uint32_t blen) {
    embed_into(a, alen, vec_a);
    embed_into(b, blen, vec_b);
    double dot = 0.0;
    for (uint32_t d = 0; d < e_dims; d++) dot += vec_a[d] * vec_b[d];
    return dot;
}

/* ---- behaviour probe ---- */

static uint32_t word_hash[MAX_WORDS];
static int32_t p_sum[MAX_PDIM];
static double p_input[MAX_DIMS + MAX_PDIM], p_mid[MAX_HIDDEN];
static uint32_t p_features;

static void feature(uint32_t h) {
    const int8_t *row = p_table + (size_t)(h % p_buckets) * p_dim;
    for (uint32_t d = 0; d < p_dim; d++) p_sum[d] += row[d];
    p_features++;
}

/* Writes ks_classes() raw scores (logits) to out; returns the winning class, -1 without a probe. */
EXPORT(ks_probe) int32_t ks_probe(const uint8_t *text, uint32_t len, double *out) {
    if (!p_classes) return -1;
    embed_into(text, len, p_input);
    uint32_t n = 0, i = 0;
    while (i < len && n < MAX_WORDS) {
        while (i < len && text[i] == ' ') i++;
        uint32_t start = i;
        while (i < len && text[i] != ' ') i++;
        if (i == start) break;
        word_hash[n++] = fnv(text + start, i - start, FNV_SEED);
    }
    for (uint32_t d = 0; d < p_dim; d++) p_sum[d] = 0;
    p_features = 0;
    for (uint32_t k = 0; k < n; k++) {
        feature(mix(1, word_hash[k]));
        if (k + 1 < n) feature(mix(2, mix(word_hash[k], word_hash[k + 1])));
        if (k + 2 < n) feature(mix(3, mix(mix(word_hash[k], word_hash[k + 1]), word_hash[k + 2])));
    }
    if (n) {
        feature(mix(4, word_hash[0]));
        feature(mix(6, word_hash[n - 1]));
    }
    if (n > 1) {
        feature(mix(5, mix(word_hash[0], word_hash[1])));
        feature(mix(7, mix(word_hash[n - 2], word_hash[n - 1])));
    }
    for (uint32_t d = 0; d < p_dim; d++) {
        p_input[e_dims + d] = p_features ? (double)p_sum[d] * p_scale / (double)p_features : 0.0;
    }
    uint32_t inputs = e_dims + p_dim;
    for (uint32_t j = 0; j < p_hidden; j++) {
        double total = rdf32(p_b1 + 4 * j);
        const uint8_t *w = p_w1 + 4u * j * inputs;
        for (uint32_t d = 0; d < inputs; d++) total += rdf32(w + 4 * d) * p_input[d];
        p_mid[j] = total > 0.0 ? total : 0.0;
    }
    int32_t best = 0;
    for (uint32_t c = 0; c < p_classes; c++) {
        double total = rdf32(p_b2 + 4 * c);
        const uint8_t *w = p_w2 + 4u * c * p_hidden;
        for (uint32_t j = 0; j < p_hidden; j++) total += rdf32(w + 4 * j) * p_mid[j];
        out[c] = total;
        if (total > out[best]) best = (int32_t)c;
    }
    return best;
}

/* ---- loops: share of four-word runs already seen in the same text ---- */

static uint32_t r_words[REPEAT_WORDS], r_seen[REPEAT_SLOTS];

EXPORT(ks_repeat) double ks_repeat(const uint8_t *text, uint32_t len) {
    uint32_t total = 0, i = 0;
    while (i < len) {
        while (i < len && text[i] == ' ') i++;
        uint32_t start = i;
        while (i < len && text[i] != ' ') i++;
        if (i == start) break;
        r_words[total % REPEAT_WORDS] = fnv(text + start, i - start, FNV_SEED);
        total++;
    }
    uint32_t n = total < REPEAT_WORDS ? total : REPEAT_WORDS;
    if (n < 12) return 0.0;
    uint32_t first = total < REPEAT_WORDS ? 0 : total % REPEAT_WORDS; /* oldest word kept */
    for (uint32_t s = 0; s < REPEAT_SLOTS; s++) r_seen[s] = 0;
    uint32_t repeated = 0, runs = n - 3;
    for (uint32_t k = 0; k < runs; k++) {
        uint32_t a = r_words[(first + k) % REPEAT_WORDS], b = r_words[(first + k + 1) % REPEAT_WORDS];
        uint32_t c = r_words[(first + k + 2) % REPEAT_WORDS], d = r_words[(first + k + 3) % REPEAT_WORDS];
        uint32_t h = mix(mix(mix(a, b), c), d) | 1u;
        uint32_t at = h & (REPEAT_SLOTS - 1);
        for (;;) {
            if (!r_seen[at]) {
                r_seen[at] = h;
                break;
            }
            if (r_seen[at] == h) {
                repeated++;
                break;
            }
            at = (at + 1) & (REPEAT_SLOTS - 1);
        }
    }
    return (double)repeated / (double)runs;
}

/* ---- wording overlap, any language: cosine of hashed byte-trigram counts ---- */

static int32_t lex_a[LEX_BINS], lex_b[LEX_BINS];

static void trigrams(const uint8_t *text, uint32_t len, int32_t *bins) {
    for (uint32_t s = 0; s < LEX_BINS; s++) bins[s] = 0;
    for (uint32_t i = 0; i + 3 <= len; i++) bins[fnv(text + i, 3, FNV_SEED) & (LEX_BINS - 1)]++;
}

EXPORT(ks_lexsim) double ks_lexsim(const uint8_t *a, uint32_t alen, const uint8_t *b, uint32_t blen) {
    trigrams(a, alen, lex_a);
    trigrams(b, blen, lex_b);
    int64_t dot = 0, na = 0, nb = 0;
    for (uint32_t s = 0; s < LEX_BINS; s++) {
        dot += (int64_t)lex_a[s] * lex_b[s];
        na += (int64_t)lex_a[s] * lex_a[s];
        nb += (int64_t)lex_b[s] * lex_b[s];
    }
    if (!na || !nb) return 0.0;
    return (double)dot / (__builtin_sqrt((double)na) * __builtin_sqrt((double)nb));
}
