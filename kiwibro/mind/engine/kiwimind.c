// Kiwi Mind inference engine.
//
// Derived from the Flyingkiwi engine by the same author: a dependency-free C implementation of a Qwen3-style
// decoder (per-head q/k RMS norm, RoPE, grouped-query full attention, SwiGLU, tied embedding) for
// WebAssembly (SIMD128, optional relaxed SIMD, optional shared-memory threads) and, for testing, native code.
//
// What is different from Flyingkiwi, and why:
//  * Split trunk. Layers below `split` never see a skill, so their KV cache and their output are shared by
//    every skill. The hidden state at the input of layer `split` is kept for every cached position. Switching
//    skills recomputes only layers split..n-1 from those states. A skill file names the split layer; all
//    loaded skills agree on it.
//  * Skills. A skill is a set of low-rank deltas y = W x + scale * B (A x) on any of the seven linear layers
//    of the upper layers (file layout: scripts/mind/formats.py). Up to KM_SLOTS skills stay loaded; loading
//    one touches neither the trunk nor the caches.
//  * Control head. A small MLP reads the RMS-normalised hidden state of the last evaluated position at the
//    input of one layer (normally the split layer), so the host can choose a skill before the upper layers run.
//  * Batched prefill. Weights are stored in 4-row tiles (rows of one group of 32 columns interleaved by four
//    bytes) so one 128-bit load feeds four output rows and one int8 activation word feeds all four. A chunk of
//    tokens is processed per weight tile, four tokens at a time, so weights are read once per chunk and the
//    kernels are bound by arithmetic, not by memory. Gate and up projections are fused with SiLU, and the
//    residual additions are fused into the output projections.
//  * Slots. Independent conversations ("contexts") share the one copy of the weights, the skills and the head.
//    A slot owns its token history, split-layer states, KV cache, active skill, logits and sampler state. Its
//    memory is a list of blocks of KM_BLOCK positions, taken when the context grows and returned to a shared pool
//    when it shrinks or the slot is freed, so a slot costs what it uses and a replaced slot costs nothing new.
//    Work areas (activations, the thread pool) are shared: one call works on the current slot and finishes
//    before another begins, which is what lets the host interleave slots in small pieces.
//  * Full attention only, any truncation distance, no sliding window, no recall head, no KV snapshots.
//
// Weights: Q4 or Q8 (one fp16 scale per 32 inputs, repacked in place at load) or f32. Activations are
// quantised to int8 per 32 values before each integer matrix product. The KV cache is int8 with one scale per
// 32 values; the first KM_SINK positions are kept exactly in float (Qwen3 attention sinks have huge values).
//
// Threads (KM_THREADS builds): helper threads share one WebAssembly.Memory and sleep on memory.atomic.wait32.
// A job is a list of tasks handed out through an atomic counter; every output element is computed by exactly
// one thread with the same instruction sequence, so results do not depend on the thread count.
#include <stddef.h>
#include <stdint.h>

#if defined(__wasm__)
#if !defined(__wasm_simd128__)
#error "the WebAssembly build needs -msimd128"
#endif
#include <wasm_simd128.h>
#define KM_API(name) __attribute__((export_name(#name)))
#else
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#define KM_API(name)
#endif

#define KM_MAGIC 0x314d4b46u       /* "FKM1" */
#define KM_SKILL_MAGIC 0x31534d4bu /* "KMS1" */
#define KM_HEAD_MAGIC 0x31484d4bu  /* "KMH1" */
#define KM_GROUP 32
#define KM_CHUNK 64 /* tokens per prefill chunk */
#define KM_MAX_LAYERS 64
#define KM_SKILLS 8        /* skills that stay loaded at once */
#define KM_MAX_SLOTS 16    /* conversations (contexts) at once */
#define KM_BLOCK_SHIFT 7
#define KM_BLOCK (1 << KM_BLOCK_SHIFT) /* positions per memory block of a slot */
#define KM_MAX_BLOCKS 256
#define KM_MAX_CTX (KM_BLOCK * KM_MAX_BLOCKS)
#define KM_MAX_RANK 128
#define KM_MAX_COLS 8192
#define KM_MAX_THREADS 16
#define KM_MAX_HEAD_DIM 256
#define KM_KVG 32  /* KV cache values per scale */
#define KM_SINK 4  /* positions kept exactly in float */
#define KM_NO_LAYER 0xffffffffu

enum { KM_F32 = 0, KM_Q8 = 2, KM_Q4 = 3 };
enum { MAT_Q, MAT_K, MAT_V, MAT_O, MAT_GATE, MAT_UP, MAT_DOWN, MAT_COUNT };

typedef struct {
  uint32_t magic, version, header_bytes, n_tensors;
  uint32_t vocab, dim, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim, window, max_ctx;
  uint32_t bos_id, eos_id, flags;
  float rope_theta, norm_eps;
  uint8_t layer_global[64];
  uint32_t recall_layer, recall_bits, reserved[14];
} km_header;

typedef struct {
  char name[40];
  uint32_t type, rows, cols, pad;
  uint64_t offset, bytes;
} km_tensor;

typedef struct {
  uint32_t magic, version;
  char name[64];
  uint32_t first_layer, n_entries, n_layers, reserved[3];
} km_skill_header;

typedef struct {
  uint32_t layer, matrix, rank;
  float scale;
  uint64_t a_offset, b_offset;
} km_skill_entry;

typedef struct {
  uint32_t magic, version, layer, in_dim, hidden, n_out;
  float eps;
  uint32_t json_offset, json_length, reserved[7];
} km_head_header;

/* A weight matrix. Q4/Q8: tiles of 4 rows x 32 columns, tile (quad, group) at (quad * groups + group) * 64
   (Q4) or * 128 (Q8) bytes, scales as f32 [quad][group][row of the quad]. F32: plain rows. */
typedef struct {
  uint8_t *w;
  float *s;
  int type, rows, cols;
} km_mat;

typedef struct {
  const float *a;  /* [rank][in] */
  const float *bt; /* [rank][out]: B transposed so one load feeds four output rows */
  int rank;
  float scale;
} km_delta;

typedef struct {
  int used, first_layer;
  char name[65];
  km_delta d[KM_MAX_LAYERS][MAT_COUNT];
  void *block;
} km_skill;

typedef struct {
  int used, layer, in_dim, hidden, n_out; /* layer < 0: the split layer */
  float eps;
  const float *w1, *b1, *w2, *b2;
  void *block;
} km_head;

typedef struct {
  const float *attn_norm, *q_norm, *k_norm, *ffn_norm;
  km_mat w[MAT_COUNT];
} km_layer;

/* One conversation. Memory block b holds positions [b * KM_BLOCK, (b + 1) * KM_BLOCK) for every layer:
   per layer kc, vc int8 [KM_BLOCK][kv_heads][head_dim] then ks, vs float [KM_BLOCK][kv_heads][head_dim / KM_KVG];
   after the layers, the split-layer states float [KM_BLOCK][dim] and the token ids int32 [KM_BLOCK]. */
typedef struct {
  int used, n_blocks;
  int n_lower, n_upper, logits_ok, head_pos;
  int active;      /* skill used for new upper evaluations, -1 = none */
  int upper_skill; /* skill the upper cache [0, n_upper) was computed under, -2 = none */
  uint64_t rng;
  float *logits, *bias, *head_state, *sink; /* sink: [layer][k, v][KM_SINK][kv_heads][head_dim], kept exactly */
  uint8_t *block[KM_MAX_BLOCKS];
} km_slot;

typedef struct {
  km_header h;
  km_mat emb;
  const float *final_norm;
  km_layer layers[KM_MAX_LAYERS];
  km_skill skills[KM_SKILLS];
  km_head head;
  float *rope; /* [max_ctx][head_dim/2] cos, then the same for sin */
  int max_ctx, ready, threads;
  int split;   /* first layer that may carry a skill; n_layers while no skill is loaded */
  /* geometry of a slot's memory block (bytes unless noted) */
  size_t blk_layer, blk_kvq, blk_kvs /* floats */, off_states, off_hist, blk_bytes, aux_bytes;
  km_slot slots[KM_MAX_SLOTS];
  float *x, *xb, *q, *k, *v, *att, *ffa, *work, *scores;
  float *lora_u[3];
  int8_t *aq;
  float *as;
  int32_t *ac;
  size_t act_cols;
  uint8_t *flag;
} km_model;

static km_model M;
static km_slot none_slot; /* what the current slot is after the current slot was freed: used = 0 */
static km_slot *S = &none_slot; /* the current slot: every call that touches a conversation works on it */
static int cur_slot = -1;

/* ------------------------------------------------------------------ memory */
#if defined(__wasm__)
extern unsigned char __heap_base;
static uintptr_t heap_top;
void *memcpy(void *restrict d, const void *restrict s, size_t n) { __builtin_memcpy(d, s, n); return d; }
void *memset(void *d, int c, size_t n) { __builtin_memset(d, c, n); return d; }
void *memmove(void *d, const void *s, size_t n) { __builtin_memmove(d, s, n); return d; }
/* New WebAssembly pages are zero and the heap only grows, so this memory is already cleared. */
static void *raw_alloc(size_t n) {
  if (!heap_top) heap_top = (uintptr_t)&__heap_base;
  uintptr_t p = (heap_top + 63) & ~(uintptr_t)63, end = p + n;
  size_t have = __builtin_wasm_memory_size(0) * 65536u;
  if (end > have && __builtin_wasm_memory_grow(0, (end - have + 65535u) / 65536u) == (size_t)-1) return 0;
  heap_top = end;
  return (void *)p;
}
#else
static void *raw_alloc(size_t n) {
  void *p = NULL;
  if (posix_memalign(&p, 64, n ? n : 64)) return NULL;
  memset(p, 0, n);
  return p;
}
#endif

KM_API(km_alloc) void *km_alloc(uint32_t n) { return raw_alloc(n); }

/* A reusable staging area for files the host hands over (skills, heads): grows, never shrinks. */
KM_API(km_staging) void *km_staging(uint32_t n) {
  static void *area;
  static uint32_t cap;
  if (n > cap) {
    void *p = raw_alloc(n);
    if (!p) return 0;
    area = p;
    cap = n;
  }
  return area;
}

/* Blocks for things that come and go (skills, the head): freed blocks are reused by later requests. */
#define ARENA_BLOCKS 64
static struct { void *p; size_t cap; int used; } arena[ARENA_BLOCKS];

static void *arena_alloc(size_t n) {
  int best = -1, free_slot = -1;
  for (int i = 0; i < ARENA_BLOCKS; i++) {
    if (!arena[i].p) { if (free_slot < 0) free_slot = i; continue; }
    if (!arena[i].used && arena[i].cap >= n && (best < 0 || arena[i].cap < arena[best].cap)) best = i;
  }
  if (best < 0) {
    if (free_slot < 0) return NULL;
    void *p = raw_alloc(n);
    if (!p) return NULL;
    arena[free_slot].p = p;
    arena[free_slot].cap = n;
    best = free_slot;
  }
  arena[best].used = 1;
  return arena[best].p;
}

static void arena_free(void *p) {
  for (int i = 0; i < ARENA_BLOCKS; i++)
    if (arena[i].p == p) arena[i].used = 0;
}

/* Memory blocks of the slots. A block that is no longer needed goes to a pool and is handed out again before any
   new memory is asked for, so freeing and replacing slots never grows the heap. */
static void *free_blocks;

static void *block_get(void) {
  void *p = free_blocks;
  if (p) {
    free_blocks = *(void **)p;
    return p;
  }
  return raw_alloc(M.blk_bytes);
}

static void block_put(void *p) {
  *(void **)p = free_blocks;
  free_blocks = p;
}

/* ------------------------------------------------------------------ vectors of four floats */
#if defined(__wasm_simd128__)
typedef v128_t vf;
static inline vf vf_zero(void) { return wasm_f32x4_splat(0.0f); }
static inline vf vf_splat(float x) { return wasm_f32x4_splat(x); }
static inline vf vf_load(const float *p) { return wasm_v128_load(p); }
static inline void vf_store(float *p, vf v) { wasm_v128_store(p, v); }
static inline vf vf_add(vf a, vf b) { return wasm_f32x4_add(a, b); }
static inline vf vf_mul(vf a, vf b) { return wasm_f32x4_mul(a, b); }
static inline float vf_hsum(vf a) {
  return wasm_f32x4_extract_lane(a, 0) + wasm_f32x4_extract_lane(a, 1) + wasm_f32x4_extract_lane(a, 2) +
         wasm_f32x4_extract_lane(a, 3);
}
/* exp with the same clamps and polynomial as km_expf, four lanes at once */
static inline vf vf_exp(vf x) {
  x = wasm_f32x4_max(wasm_f32x4_min(x, vf_splat(88.0f)), vf_splat(-87.0f));
  vf n = wasm_f32x4_floor(wasm_f32x4_add(wasm_f32x4_mul(x, vf_splat(1.44269504088896341f)), vf_splat(0.5f)));
  vf r = wasm_f32x4_sub(wasm_f32x4_sub(x, wasm_f32x4_mul(n, vf_splat(0.693145751953125f))),
                        wasm_f32x4_mul(n, vf_splat(1.42860682030941723e-6f)));
  vf p = vf_splat(0.00138888892f);
  p = wasm_f32x4_add(vf_splat(0.00833333377f), wasm_f32x4_mul(r, p));
  p = wasm_f32x4_add(vf_splat(0.0416666679f), wasm_f32x4_mul(r, p));
  p = wasm_f32x4_add(vf_splat(0.166666672f), wasm_f32x4_mul(r, p));
  p = wasm_f32x4_add(vf_splat(0.5f), wasm_f32x4_mul(r, p));
  p = wasm_f32x4_add(vf_splat(1.0f), wasm_f32x4_mul(r, p));
  p = wasm_f32x4_add(vf_splat(1.0f), wasm_f32x4_mul(r, p));
  v128_t bits = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(n), wasm_i32x4_splat(127)), 23);
  return wasm_f32x4_mul(p, bits);
}
static inline vf vf_silu_mul(vf g, vf u) {
  vf e = vf_exp(wasm_f32x4_neg(g));
  return wasm_f32x4_mul(wasm_f32x4_div(g, wasm_f32x4_add(vf_splat(1.0f), e)), u);
}
#else
typedef struct { float f[4]; } vf;
static inline vf vf_zero(void) { vf r = {{0, 0, 0, 0}}; return r; }
static inline vf vf_splat(float x) { vf r = {{x, x, x, x}}; return r; }
static inline vf vf_load(const float *p) { vf r = {{p[0], p[1], p[2], p[3]}}; return r; }
static inline void vf_store(float *p, vf v) { for (int i = 0; i < 4; i++) p[i] = v.f[i]; }
static inline vf vf_add(vf a, vf b) { for (int i = 0; i < 4; i++) a.f[i] += b.f[i]; return a; }
static inline vf vf_mul(vf a, vf b) { for (int i = 0; i < 4; i++) a.f[i] *= b.f[i]; return a; }
static inline float vf_hsum(vf a) { return a.f[0] + a.f[1] + a.f[2] + a.f[3]; }
static inline float km_expf(float x);
static inline vf vf_silu_mul(vf g, vf u) {
  for (int i = 0; i < 4; i++) g.f[i] = g.f[i] / (1.0f + km_expf(-g.f[i])) * u.f[i];
  return g;
}
#endif

/* ------------------------------------------------------------------ math */
static inline float km_f16(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, u;
  if (e == 0) {
    if (!m) u = s;
    else {
      int ee = 1;
      while (!(m & 0x400u)) { m <<= 1; ee--; }
      u = s | ((uint32_t)(ee + 112) << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (e == 31) u = s | 0x7f800000u | (m << 13);
  else u = s | ((e + 112) << 23) | (m << 13);
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

static inline float km_floorf(float x) {
#if defined(__wasm__)
  return __builtin_floorf(x);
#else
  return floorf(x);
#endif
}

static inline float km_expf(float x) {
  x = x > 88.0f ? 88.0f : (x < -87.0f ? -87.0f : x);
  float n = km_floorf(x * 1.44269504088896341f + 0.5f);
  float r = x - n * 0.693145751953125f - n * 1.42860682030941723e-6f;
  float p = 0.00138888892f;
  p = 0.00833333377f + r * p;
  p = 0.0416666679f + r * p;
  p = 0.166666672f + r * p;
  p = 0.5f + r * p;
  p = 1.0f + r * p;
  p = 1.0f + r * p;
  uint32_t bits = (uint32_t)((int32_t)n + 127) << 23;
  float scale;
  __builtin_memcpy(&scale, &bits, 4);
  return p * scale;
}

static inline float km_rsqrt(float x) { return 1.0f / __builtin_sqrtf(x); }

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
  vf acc = vf_zero();
  for (int i = 0; i < n; i += 4) acc = vf_add(acc, vf_mul(vf_load(x + i), vf_load(x + i)));
  const float r = km_rsqrt(vf_hsum(acc) / (float)n + eps);
  const vf rv = vf_splat(r);
  for (int i = 0; i < n; i += 4) vf_store(o + i, vf_mul(vf_mul(vf_load(x + i), rv), w ? vf_load(w + i) : vf_splat(1.0f)));
}

/* One group of 32 floats -> int8 codes, the scale, and -8 * (sum of the codes), the correction for the
   unsigned nibble operand of the relaxed dot product. */
static inline void quantize_group(const float *x, int8_t *q, float *scale, int32_t *corr) {
#if defined(__wasm_simd128__)
  v128_t v[8], amax = wasm_f32x4_splat(0.0f);
  for (int i = 0; i < 8; i++) {
    v[i] = wasm_v128_load(x + 4 * i);
    amax = wasm_f32x4_max(amax, wasm_f32x4_abs(v[i]));
  }
  amax = wasm_f32x4_max(amax, wasm_i32x4_shuffle(amax, amax, 2, 3, 0, 1));
  amax = wasm_f32x4_max(amax, wasm_i32x4_shuffle(amax, amax, 1, 0, 3, 2));
  const float a = wasm_f32x4_extract_lane(amax, 0);
  const float d = a / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
  *scale = d;
  const v128_t idv = wasm_f32x4_splat(id), half = wasm_i32x4_splat(0x3f000000), sign = wasm_i32x4_splat((int32_t)0x80000000u);
  v128_t r[8];
  for (int i = 0; i < 8; i++) {
    v128_t t = wasm_f32x4_mul(v[i], idv);
    r[i] = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_add(t, wasm_v128_or(wasm_v128_and(t, sign), half)));
  }
  v128_t b0 = wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(r[0], r[1]), wasm_i16x8_narrow_i32x4(r[2], r[3]));
  v128_t b1 = wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(r[4], r[5]), wasm_i16x8_narrow_i32x4(r[6], r[7]));
  wasm_v128_store(q, b0);
  wasm_v128_store(q + 16, b1);
  v128_t s = wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_add(wasm_i16x8_extadd_pairwise_i8x16(b0), wasm_i16x8_extadd_pairwise_i8x16(b1)));
  *corr = -8 * (wasm_i32x4_extract_lane(s, 0) + wasm_i32x4_extract_lane(s, 1) + wasm_i32x4_extract_lane(s, 2) +
                wasm_i32x4_extract_lane(s, 3));
#else
  float amax = 0.0f; /* round half away from zero, as the vector path and the converter do */
  for (int i = 0; i < KM_GROUP; i++) {
    float a = x[i] < 0 ? -x[i] : x[i];
    amax = a > amax ? a : amax;
  }
  const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
  *scale = d;
  int sum = 0;
  for (int i = 0; i < KM_GROUP; i++) {
    const float v = x[i] * id;
    sum += (q[i] = (int8_t)(v >= 0.0f ? (int)(v + 0.5f) : -(int)(-v + 0.5f)));
  }
  *corr = -8 * sum;
#endif
}

static void quantize_rows(const float *x, int T, int cols) {
  const int ng = cols / KM_GROUP;
  for (int t = 0; t < T; t++)
    for (int g = 0; g < ng; g++)
      quantize_group(x + (size_t)t * cols + g * KM_GROUP, M.aq + (size_t)t * cols + g * KM_GROUP,
                     M.as + (size_t)t * ng + g, M.ac + (size_t)t * ng + g);
}

/* ------------------------------------------------------------------ tile kernels */
/* tile_*: the dot products of one tile row-quad (4 output rows) with TB tokens, accumulated over all groups.
   acc[k] holds the four rows for token t0 + k. TB is a compile-time constant (4 for prefill, 1 otherwise). */
#define KM_INLINE static inline __attribute__((always_inline))

#if defined(__wasm_simd128__)
KM_INLINE void tile_q4(const uint8_t *w, const float *s, int ng, int cols, int t0, const int TB, vf *acc) {
  for (int k = 0; k < TB; k++) acc[k] = vf_zero();
  const v128_t m4 = wasm_i8x16_splat(0x0f);
#if !defined(__wasm_relaxed_simd__)
  const v128_t eight = wasm_i8x16_splat(8);
#endif
  for (int g = 0; g < ng; g++) {
    const v128_t sv = wasm_v128_load(s + g * 4);
    v128_t lo[4], hi[4];
    for (int j = 0; j < 4; j++) {
      const v128_t p = wasm_v128_load(w + g * 64 + 16 * j);
#if defined(__wasm_relaxed_simd__)
      lo[j] = wasm_v128_and(p, m4);
      hi[j] = wasm_u8x16_shr(p, 4);
#else
      lo[j] = wasm_i8x16_sub(wasm_v128_and(p, m4), eight);
      hi[j] = wasm_i8x16_sub(wasm_u8x16_shr(p, 4), eight);
#endif
    }
    for (int k = 0; k < TB; k++) {
      const int8_t *x = M.aq + (size_t)(t0 + k) * cols + g * KM_GROUP;
      const size_t gi = (size_t)(t0 + k) * ng + g;
#if defined(__wasm_relaxed_simd__)
      /* unsigned nibbles (code + 8) are the 7-bit operand; the +8 offset is removed by starting at -8 * sum(x) */
      v128_t a = wasm_v128_load32_splat(M.ac + gi);
      for (int j = 0; j < 4; j++) {
        a = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load32_splat(x + 4 * j), lo[j], a);
        a = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load32_splat(x + 16 + 4 * j), hi[j], a);
      }
#else
      v128_t pa = wasm_i32x4_splat(0), pb = wasm_i32x4_splat(0);
      for (int j = 0; j < 4; j++) {
        v128_t xl = wasm_v128_load32_splat(x + 4 * j), xh = wasm_v128_load32_splat(x + 16 + 4 * j);
        pa = wasm_i32x4_add(pa, wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extmul_low_i8x16(lo[j], xl)));
        pb = wasm_i32x4_add(pb, wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extmul_high_i8x16(lo[j], xl)));
        pa = wasm_i32x4_add(pa, wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extmul_low_i8x16(hi[j], xh)));
        pb = wasm_i32x4_add(pb, wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extmul_high_i8x16(hi[j], xh)));
      }
      v128_t a = wasm_i32x4_add(wasm_i32x4_shuffle(pa, pb, 0, 2, 4, 6), wasm_i32x4_shuffle(pa, pb, 1, 3, 5, 7));
#endif
      acc[k] = wasm_f32x4_add(acc[k], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(a),
                                                     wasm_f32x4_mul(sv, wasm_v128_load32_splat(M.as + gi))));
    }
  }
}

KM_INLINE void tile_q8(const uint8_t *w, const float *s, int ng, int cols, int t0, const int TB, vf *acc) {
  for (int k = 0; k < TB; k++) acc[k] = vf_zero();
  for (int g = 0; g < ng; g++) {
    const v128_t sv = wasm_v128_load(s + g * 4);
    v128_t wv[8];
    for (int j = 0; j < 8; j++) wv[j] = wasm_v128_load(w + g * 128 + 16 * j);
    for (int k = 0; k < TB; k++) {
      const int8_t *x = M.aq + (size_t)(t0 + k) * cols + g * KM_GROUP;
      v128_t pa = wasm_i32x4_splat(0), pb = wasm_i32x4_splat(0);
      for (int j = 0; j < 8; j++) {
        v128_t xv = wasm_v128_load32_splat(x + 4 * j);
        pa = wasm_i32x4_add(pa, wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extmul_low_i8x16(wv[j], xv)));
        pb = wasm_i32x4_add(pb, wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extmul_high_i8x16(wv[j], xv)));
      }
      v128_t a = wasm_i32x4_add(wasm_i32x4_shuffle(pa, pb, 0, 2, 4, 6), wasm_i32x4_shuffle(pa, pb, 1, 3, 5, 7));
      acc[k] = wasm_f32x4_add(acc[k], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(a),
                                                     wasm_f32x4_mul(sv, wasm_v128_load32_splat(M.as + (size_t)(t0 + k) * ng + g))));
    }
  }
}
#else
KM_INLINE void tile_q4(const uint8_t *w, const float *s, int ng, int cols, int t0, const int TB, vf *acc) {
  for (int k = 0; k < TB; k++) {
    acc[k] = vf_zero();
    const int8_t *x0 = M.aq + (size_t)(t0 + k) * cols;
    for (int g = 0; g < ng; g++) {
      const int8_t *x = x0 + g * KM_GROUP;
      for (int i = 0; i < 4; i++) {
        int sum = 0;
        for (int j = 0; j < 4; j++)
          for (int b = 0; b < 4; b++) {
            const uint8_t byte = w[g * 64 + 16 * j + 4 * i + b];
            sum += ((int)(byte & 15) - 8) * x[4 * j + b] + ((int)(byte >> 4) - 8) * x[16 + 4 * j + b];
          }
        acc[k].f[i] += (float)sum * (s[g * 4 + i] * M.as[(size_t)(t0 + k) * ng + g]);
      }
    }
  }
}

KM_INLINE void tile_q8(const uint8_t *w, const float *s, int ng, int cols, int t0, const int TB, vf *acc) {
  for (int k = 0; k < TB; k++) {
    acc[k] = vf_zero();
    const int8_t *x0 = M.aq + (size_t)(t0 + k) * cols;
    for (int g = 0; g < ng; g++) {
      const int8_t *x = x0 + g * KM_GROUP;
      for (int i = 0; i < 4; i++) {
        int sum = 0;
        for (int j = 0; j < 8; j++)
          for (int b = 0; b < 4; b++) sum += (int)(int8_t)w[g * 128 + 16 * j + 4 * i + b] * x[4 * j + b];
        acc[k].f[i] += (float)sum * (s[g * 4 + i] * M.as[(size_t)(t0 + k) * ng + g]);
      }
    }
  }
}
#endif

static inline float dot_f32(const float *a, const float *b, int n) {
  vf acc = vf_zero();
  int i = 0;
  for (; i + 4 <= n; i += 4) acc = vf_add(acc, vf_mul(vf_load(a + i), vf_load(b + i)));
  float s = vf_hsum(acc);
  for (; i < n; i++) s += a[i] * b[i];
  return s;
}

/* f32 weights (rare): four plain rows against the float activations */
KM_INLINE void tile_f32(const km_mat *W, size_t quad, const float *x, int t0, const int TB, vf *acc) {
  const float *w = (const float *)W->w + quad * 4 * (size_t)W->cols;
  for (int k = 0; k < TB; k++) {
    float r[4];
    for (int i = 0; i < 4; i++) r[i] = dot_f32(w + (size_t)i * W->cols, x + (size_t)(t0 + k) * W->cols, W->cols);
    acc[k] = vf_load(r);
  }
}

KM_INLINE void tile(const km_mat *W, size_t quad, const float *x, int t0, const int TB, vf *acc) {
  const int ng = W->cols / KM_GROUP;
  if (W->type == KM_Q4) tile_q4(W->w + quad * ng * 64, W->s + quad * ng * 4, ng, W->cols, t0, TB, acc);
  else if (W->type == KM_Q8) tile_q8(W->w + quad * ng * 128, W->s + quad * ng * 4, ng, W->cols, t0, TB, acc);
  else tile_f32(W, quad, x, t0, TB, acc);
}

/* ------------------------------------------------------------------ jobs and threads */
enum { MODE_STORE, MODE_ADD, MODE_SILU_MUL };
enum { JOB_MATMUL = 1, JOB_LORA = 2, JOB_ATTENTION = 3 };

typedef struct {
  int32_t generation, remaining, threads, kind, next;
  int nmat, mode, T, total, chunk;
  const km_mat *W[3];
  const km_delta *delta[3];
  float *out[3];
  const float *x;
  int layer;
  int pos0;
} km_job;
static km_job J = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 0, 0, 0};

/* out[t * rows + 4 * quad ...] for quads [qa, qb) of one matrix (or of a fused gate/up pair). */
static void matmul_quads(int m, int qa, int qb) {
  const km_mat *W = J.W[m];
  const km_delta *dl = J.delta[m];
  const int rows = W->rows, T = J.T;
  float *u = dl ? M.lora_u[m] : NULL;
  for (int quad = qa; quad < qb; quad++) {
    for (int t0 = 0; t0 < T;) {
      const int tb = T - t0 >= 4 ? 4 : 1;
      vf acc[4], up[4];
      if (tb == 4) tile(W, (size_t)quad, J.x, t0, 4, acc);
      else tile(W, (size_t)quad, J.x, t0, 1, acc);
      if (J.mode == MODE_SILU_MUL) {
        if (tb == 4) tile(J.W[1], (size_t)quad, J.x, t0, 4, up);
        else tile(J.W[1], (size_t)quad, J.x, t0, 1, up);
      }
      for (int k = 0; k < tb; k++) {
        vf v = acc[k];
        if (dl) {
          const float *uk = u + (size_t)(t0 + k) * dl->rank;
          vf d = vf_zero();
          for (int r = 0; r < dl->rank; r++)
            d = vf_add(d, vf_mul(vf_splat(uk[r]), vf_load(dl->bt + (size_t)r * rows + 4 * (size_t)quad)));
          v = vf_add(v, vf_mul(d, vf_splat(dl->scale)));
        }
        float *o = J.out[m] + (size_t)(t0 + k) * rows + 4 * (size_t)quad;
        if (J.mode == MODE_SILU_MUL) {
          vf g = v, uu = up[k];
          const km_delta *du = J.delta[1];
          if (du) {
            const float *uk = M.lora_u[1] + (size_t)(t0 + k) * du->rank;
            vf d = vf_zero();
            for (int r = 0; r < du->rank; r++)
              d = vf_add(d, vf_mul(vf_splat(uk[r]), vf_load(du->bt + (size_t)r * rows + 4 * (size_t)quad)));
            uu = vf_add(uu, vf_mul(d, vf_splat(du->scale)));
          }
          vf_store(o, vf_silu_mul(g, uu));
        } else if (J.mode == MODE_ADD) {
          vf_store(o, vf_add(vf_load(o), v));
        } else {
          vf_store(o, v);
        }
      }
      t0 += tb;
    }
  }
}

static void run_matmul(void) {
  for (;;) {
    int q0 = __atomic_fetch_add(&J.next, J.chunk, __ATOMIC_ACQ_REL);
    if (q0 >= J.total) break;
    const int q1 = q0 + J.chunk < J.total ? q0 + J.chunk : J.total;
    for (int m = 0, base = 0; m < J.nmat && q0 < q1; m++) {
      const int quads = J.W[m]->rows / 4, end = base + quads;
      if (q0 < end) {
        const int stop = q1 < end ? q1 : end;
        matmul_quads(m, q0 - base, stop - base);
        q0 = stop;
      }
      base = end;
    }
  }
}

/* u[m][t][i] = A_i . x[t] for the matrices of the job that carry a delta; tasks are (matrix, rank row). */
static void lora_row(int m, int i) {
  const km_delta *dl = J.delta[m];
  const int cols = J.W[m]->cols, T = J.T;
  const float *a = dl->a + (size_t)i * cols;
  float *u = M.lora_u[m];
  for (int t = 0; t < T; t++) u[(size_t)t * dl->rank + i] = dot_f32(a, J.x + (size_t)t * cols, cols);
}

static void run_lora(void) {
  for (;;) {
    const int task = __atomic_fetch_add(&J.next, 1, __ATOMIC_ACQ_REL);
    if (task >= J.total) break;
    int rest = task;
    for (int m = 0; m < J.nmat; m++) {
      const int r = J.delta[m] ? J.delta[m]->rank : 0;
      if (rest < r) { lora_row(m, rest); break; }
      rest -= r;
    }
  }
}

static void attention_task(int t, int h, int thread);

static void run_attention(int index) {
  const int tasks = J.T * (int)M.h.n_heads;
  for (;;) {
    const int task = __atomic_fetch_add(&J.next, 1, __ATOMIC_ACQ_REL);
    if (task >= tasks) break;
    attention_task(task / (int)M.h.n_heads, task % (int)M.h.n_heads, index);
  }
}

static void run_slice(int index) {
  if (J.kind == JOB_MATMUL) run_matmul();
  else if (J.kind == JOB_LORA) run_lora();
  else run_attention(index);
}

#if defined(KM_THREADS)
KM_API(km_thread_main) void km_thread_main(int index) {
  int32_t seen = __atomic_load_n(&J.generation, __ATOMIC_ACQUIRE);
  for (;;) {
    int32_t g;
    for (int spin = 0; (g = __atomic_load_n(&J.generation, __ATOMIC_ACQUIRE)) == seen; spin++)
      if (spin > 20000) __builtin_wasm_memory_atomic_wait32(&J.generation, seen, -1);
    seen = g;
    if (index < J.threads) {
      run_slice(index);
      if (__atomic_sub_fetch(&J.remaining, 1, __ATOMIC_ACQ_REL) == 0) __builtin_wasm_memory_atomic_notify(&J.remaining, 1);
    }
  }
}

static void dispatch(void) {
  J.next = 0;
  if (J.threads <= 1) { run_slice(0); return; }
  __atomic_store_n(&J.remaining, J.threads - 1, __ATOMIC_RELEASE);
  __atomic_add_fetch(&J.generation, 1, __ATOMIC_ACQ_REL);
  __builtin_wasm_memory_atomic_notify(&J.generation, (uint32_t)J.threads);
  run_slice(0);
  int32_t r;
  for (int spin = 0; (r = __atomic_load_n(&J.remaining, __ATOMIC_ACQUIRE)) != 0; spin++)
    if (spin > 20000) __builtin_wasm_memory_atomic_wait32(&J.remaining, r, -1);
}
#elif defined(_OPENMP)
static void dispatch(void) {
  J.next = 0;
  if (J.threads <= 1) { run_slice(0); return; }
#pragma omp parallel num_threads(J.threads)
  run_slice(omp_get_thread_num());
}
#else
static void dispatch(void) { J.next = 0; run_slice(0); }
#endif

KM_API(km_set_threads) void km_set_threads(int n) { J.threads = n < 1 ? 1 : (n > KM_MAX_THREADS ? KM_MAX_THREADS : n); }

/* Several matrices that share one input: outs[m][t * rows_m + r] = W_m[r] . x[t] (+ skill delta), as stored
   (MODE_STORE), added to the existing values (MODE_ADD) or, for a gate/up pair, SiLU(gate) * up (MODE_SILU_MUL,
   the result goes to outs[0]). */
static void matmul_multi(int n, const km_mat *const *Ws, const km_delta *const *deltas, float *const *outs,
                         const float *x, int T, int mode) {
  const int cols = Ws[0]->cols;
  int quantised = 0, total = 0, lora_rows = 0;
  for (int m = 0; m < n; m++) {
    quantised |= Ws[m]->type != KM_F32;
    J.W[m] = Ws[m];
    J.out[m] = outs[m];
    J.delta[m] = deltas ? deltas[m] : NULL;
    lora_rows += J.delta[m] ? J.delta[m]->rank : 0;
    total += Ws[m]->rows / 4;
  }
  J.x = x;
  J.T = T;
  J.nmat = n;
  if (quantised) quantize_rows(x, T, cols);
  if (lora_rows) {
    J.kind = JOB_LORA;
    J.total = lora_rows;
    if ((size_t)T * lora_rows * (size_t)cols < ((size_t)1 << 18)) {
      J.next = 0;
      run_lora();
    } else {
      dispatch();
    }
  }
  J.kind = JOB_MATMUL;
  J.mode = mode;
  J.total = mode == MODE_SILU_MUL ? Ws[0]->rows / 4 : total;
  const int fair = J.total / (J.threads * 8);  /* quads per fetch: small enough to balance, large enough to be cheap */
  J.chunk = fair < (T >= 16 ? 1 : 4) ? (T >= 16 ? 1 : 4) : (fair > (T >= 16 ? 4 : 64) ? (T >= 16 ? 4 : 64) : fair);
  dispatch();
}

static void matmul(float *out, const km_mat *W, const km_delta *delta, const float *x, int T, int mode) {
  const km_mat *Ws[1] = {W};
  const km_delta *ds[1] = {delta};
  float *outs[1] = {out};
  matmul_multi(1, Ws, ds, outs, x, T, mode);
}

/* ------------------------------------------------------------------ model setup */
static int bind_mat(km_mat *m, uint8_t *base, const km_tensor *t, uint32_t rows, uint32_t cols) {
  static uint8_t tmp[4 * KM_MAX_COLS];
  if (t->rows != rows || t->cols != cols || cols % KM_GROUP) return 0;
  m->type = (int)t->type;
  m->rows = (int)rows;
  m->cols = (int)cols;
  m->w = base + t->offset;
  m->s = NULL;
  if (t->type == KM_F32) return t->bytes == (uint64_t)rows * cols * 4;
  if ((t->type != KM_Q4 && t->type != KM_Q8) || rows % 4 || cols > KM_MAX_COLS) return 0;
  const size_t ng = cols / KM_GROUP, row_bytes = t->type == KM_Q4 ? cols / 2 : cols, code_bytes = rows * row_bytes;
  if (t->bytes != code_bytes + (size_t)rows * ng * 2) return 0;
  const uint16_t *s16 = (const uint16_t *)(m->w + code_bytes);
  m->s = (float *)raw_alloc((size_t)rows * ng * sizeof(float));
  if (!m->s) return 0;
  for (size_t quad = 0; quad < rows / 4; quad++) {
    uint8_t *dst = m->w + quad * 4 * row_bytes;
    __builtin_memcpy(tmp, dst, 4 * row_bytes);
    for (size_t g = 0; g < ng; g++) {
      for (int i = 0; i < 4; i++) m->s[(quad * ng + g) * 4 + i] = km_f16(s16[(quad * 4 + i) * ng + g]);
      if (t->type == KM_Q4)
        for (int j = 0; j < 4; j++)
          for (int i = 0; i < 4; i++) __builtin_memcpy(dst + g * 64 + 16 * j + 4 * i, tmp + i * row_bytes + g * 16 + 4 * j, 4);
      else
        for (int j = 0; j < 8; j++)
          for (int i = 0; i < 4; i++) __builtin_memcpy(dst + g * 128 + 16 * j + 4 * i, tmp + i * row_bytes + g * 32 + 4 * j, 4);
    }
  }
  return 1;
}

static const float *bind_vec(const uint8_t *base, const km_tensor *t, uint32_t n) {
  return (t->type == KM_F32 && t->rows == 1 && t->cols == n && t->bytes == (uint64_t)n * 4) ?
         (const float *)(base + t->offset) : NULL;
}

static int alloc_activations(size_t cols) {
  if (cols <= M.act_cols) return 1;
  M.aq = (int8_t *)raw_alloc((size_t)KM_CHUNK * cols);
  M.as = (float *)raw_alloc((size_t)KM_CHUNK * cols / KM_GROUP * 4);
  M.ac = (int32_t *)raw_alloc((size_t)KM_CHUNK * cols / KM_GROUP * 4);
  M.act_cols = cols;
  return M.aq && M.as && M.ac;
}

static int slot_create(void);

/* Returns 0, or a negative code: -1 bad header, -2 unsupported shape, -3 tensor table, -4 tensor bounds,
   -5 embedding, -6 layer vectors, -7 layer matrices, -8 out of memory, -9 final norm, -10 already initialised.
   The weights in [base, base + size) are repacked in place and must stay where they are. */
KM_API(km_init) int km_init(uint8_t *base, uint32_t size, uint32_t max_ctx) {
  if (M.ready) return -10;
  const km_header *h = (const km_header *)base;
  if (size < sizeof(km_header) || h->magic != KM_MAGIC || (h->version != 1 && h->version != 2)) return -1;
  if (h->window || h->flags || h->recall_layer || h->recall_bits) return -2;
  const uint32_t D = h->dim, HD = h->head_dim, NH = h->n_heads, KV = h->n_kv_heads, F = h->ffn_dim, V = h->vocab;
  if (!h->n_layers || h->n_layers > KM_MAX_LAYERS || !KV || !NH || NH % KV || HD % KM_KVG || HD > KM_MAX_HEAD_DIM ||
      D % KM_GROUP || F % KM_GROUP || !V || (NH * HD) % KM_GROUP) return -2;
  if (!max_ctx || max_ctx > h->max_ctx) max_ctx = h->max_ctx;
  if (max_ctx > KM_MAX_CTX) max_ctx = KM_MAX_CTX;
  if (h->n_tensors != 2 + 11 * h->n_layers || h->header_bytes != sizeof(km_header) + sizeof(km_tensor) * h->n_tensors ||
      (uint64_t)h->header_bytes > size) return -3;
  M.h = *h;
  M.max_ctx = (int)max_ctx;
  const km_tensor *t = (const km_tensor *)(base + sizeof(km_header));
  for (uint32_t i = 0; i < h->n_tensors; i++)
    if (t[i].offset < h->header_bytes || t[i].offset % 64 || t[i].offset > size || t[i].bytes > size - t[i].offset) return -4;
  if (!bind_mat(&M.emb, base, &t[0], V, D)) return -5;
  for (uint32_t l = 0; l < h->n_layers; l++) {
    const km_tensor *lt = t + 1 + 11 * l;
    km_layer *L = &M.layers[l];
    L->attn_norm = bind_vec(base, &lt[0], D);
    L->q_norm = bind_vec(base, &lt[1], HD);
    L->k_norm = bind_vec(base, &lt[2], HD);
    L->ffn_norm = bind_vec(base, &lt[7], D);
    if (!L->attn_norm || !L->q_norm || !L->k_norm || !L->ffn_norm) return -6;
    if (!bind_mat(&L->w[MAT_Q], base, &lt[3], NH * HD, D) || !bind_mat(&L->w[MAT_K], base, &lt[4], KV * HD, D) ||
        !bind_mat(&L->w[MAT_V], base, &lt[5], KV * HD, D) || !bind_mat(&L->w[MAT_O], base, &lt[6], D, NH * HD) ||
        !bind_mat(&L->w[MAT_GATE], base, &lt[8], F, D) || !bind_mat(&L->w[MAT_UP], base, &lt[9], F, D) ||
        !bind_mat(&L->w[MAT_DOWN], base, &lt[10], D, F))
      return -7;
  }
  M.final_norm = bind_vec(base, &t[1 + 11 * h->n_layers], D);
  if (!M.final_norm) return -9;
  const size_t C = KM_CHUNK, wide = F > NH * HD ? F : NH * HD;
  M.x = (float *)raw_alloc(C * D * 4);
  M.xb = (float *)raw_alloc(C * wide * 4);
  M.q = (float *)raw_alloc(C * NH * HD * 4);
  M.k = (float *)raw_alloc(C * KV * HD * 4);
  M.v = (float *)raw_alloc(C * KV * HD * 4);
  M.att = (float *)raw_alloc(C * NH * HD * 4);
  M.ffa = (float *)raw_alloc(C * F * 4);
  for (int i = 0; i < 3; i++) M.lora_u[i] = (float *)raw_alloc(C * KM_MAX_RANK * 4);
  M.scores = (float *)raw_alloc((size_t)KM_MAX_THREADS * max_ctx * 4);
  M.work = (float *)raw_alloc((size_t)V * 4);
  M.flag = (uint8_t *)raw_alloc(V);
  M.rope = (float *)raw_alloc((size_t)max_ctx * HD * 4);
  if (!M.x || !M.xb || !M.q || !M.k || !M.v || !M.att || !M.ffa || !M.lora_u[0] || !M.lora_u[1] || !M.lora_u[2] ||
      !M.scores || !M.work || !M.flag || !M.rope || !alloc_activations(wide > D ? wide : D))
    return -8;
  /* slot geometry: see km_slot */
  const size_t NG = HD / KM_KVG;
  M.blk_kvq = (size_t)KM_BLOCK * KV * HD;
  M.blk_kvs = (size_t)KM_BLOCK * KV * NG;
  M.blk_layer = 2 * M.blk_kvq + 2 * M.blk_kvs * 4;
  M.off_states = h->n_layers * M.blk_layer;
  M.off_hist = M.off_states + (size_t)KM_BLOCK * D * 4;
  M.blk_bytes = M.off_hist + (size_t)KM_BLOCK * 4;
  M.aux_bytes = ((size_t)V * 2 + D + (size_t)h->n_layers * 2 * KM_SINK * KV * HD) * 4;
#if !defined(__wasm__)
  /* Match Hugging Face: float32 inverse frequencies and float32 angles. (WebAssembly: the host fills the table.) */
  const int half = (int)HD / 2;
  for (int i = 0; i < half; i++) {
    float inv = (float)(1.0 / pow((double)h->rope_theta, (double)(2 * i) / (double)HD));
    for (int p = 0; p < (int)max_ctx; p++) {
      float ang = (float)p * inv;
      M.rope[(size_t)p * half + i] = (float)cos((double)ang);
      M.rope[(size_t)max_ctx * half + (size_t)p * half + i] = (float)sin((double)ang);
    }
  }
  if (J.threads <= 1) {
#if defined(_OPENMP)
    km_set_threads(omp_get_max_threads());
#endif
  }
#endif
  M.split = (int)h->n_layers;
  if (slot_create() != 0) return -8;
  M.ready = 1;
  return 0;
}

KM_API(km_rope_table) float *km_rope_table(void) { return M.rope; }
KM_API(km_max_ctx) int km_max_ctx(void) { return M.max_ctx; }
KM_API(km_vocab) int km_vocab(void) { return (int)M.h.vocab; }
KM_API(km_dim) int km_dim(void) { return (int)M.h.dim; }
KM_API(km_n_layers) int km_n_layers(void) { return (int)M.h.n_layers; }
KM_API(km_split) int km_split(void) { return M.split; }
KM_API(km_skill_slots) int km_skill_slots(void) { return KM_SKILLS; }

/* ------------------------------------------------------------------ slots (conversations) */
static inline size_t pos_in_block(int pos) { return (size_t)(pos & (KM_BLOCK - 1)); }
static inline int32_t *hist_ptr(const km_slot *s, int pos) {
  return (int32_t *)(s->block[pos >> KM_BLOCK_SHIFT] + M.off_hist) + pos_in_block(pos);
}
static inline float *state_ptr(const km_slot *s, int pos) {
  return (float *)(s->block[pos >> KM_BLOCK_SHIFT] + M.off_states) + pos_in_block(pos) * M.h.dim;
}

typedef struct { int8_t *kc, *vc; float *ks, *vs; } km_kvblock;
static inline km_kvblock kv_block(const km_slot *s, int layer, int b) {
  uint8_t *base = s->block[b] + (size_t)layer * M.blk_layer;
  km_kvblock r;
  r.kc = (int8_t *)base;
  r.vc = r.kc + M.blk_kvq;
  r.ks = (float *)(base + 2 * M.blk_kvq);
  r.vs = r.ks + M.blk_kvs;
  return r;
}

static int blocks_for(int positions) { return (positions + KM_BLOCK - 1) >> KM_BLOCK_SHIFT; }

/* Keep only the blocks that `positions` positions need. */
static void slot_trim(km_slot *s, int positions) {
  for (int need = blocks_for(positions); s->n_blocks > need;) {
    block_put(s->block[--s->n_blocks]);
    s->block[s->n_blocks] = NULL;
  }
}

static int slot_reserve(km_slot *s, int positions) {
  for (int need = blocks_for(positions); s->n_blocks < need; s->n_blocks++)
    if (!(s->block[s->n_blocks] = (uint8_t *)block_get())) return 0;
  return 1;
}

static void slot_clear(km_slot *s) {
  s->n_lower = s->n_upper = s->logits_ok = 0;
  s->head_pos = -1;
  s->active = -1;
  s->upper_skill = -2;
  s->rng = 0x9e3779b97f4a7c15ull;
}

/* The lowest free slot, with its fixed-size parts (kept for the next slot that takes this number). Returns its
   number, or -1. It becomes the current slot only when there is none. */
static int slot_create(void) {
  for (int id = 0; id < KM_MAX_SLOTS; id++) {
    km_slot *s = &M.slots[id];
    if (s->used) continue;
    if (!s->logits) {
      const size_t V = M.h.vocab, D = M.h.dim;
      float *aux = (float *)raw_alloc(M.aux_bytes);
      if (!aux) return -1;
      s->logits = aux;
      s->bias = aux + V;
      s->head_state = aux + 2 * V;
      s->sink = aux + 2 * V + D;
    }
    __builtin_memset(s->bias, 0, (size_t)M.h.vocab * 4);
    slot_clear(s);
    s->used = 1;
    if (cur_slot < 0) {
      S = s;
      cur_slot = id;
    }
    return id;
  }
  return -1;
}

static km_slot *slot_of(int id) { return id >= 0 && id < KM_MAX_SLOTS && M.slots[id].used ? &M.slots[id] : NULL; }

/* A new empty slot: returns its number, or -1 when all KM_MAX_SLOTS are taken. */
KM_API(km_slot_new) int km_slot_new(void) { return M.ready ? slot_create() : -1; }

/* Gives the slot's blocks back to the pool. If it was the current slot, no slot is current until km_slot_use. */
KM_API(km_slot_free) int km_slot_free(int id) {
  km_slot *s = slot_of(id);
  if (!s) return -1;
  slot_trim(s, 0);
  s->used = 0;
  if (id == cur_slot) {
    S = &none_slot;
    cur_slot = -1;
  }
  return 0;
}

KM_API(km_slot_use) int km_slot_use(int id) {
  km_slot *s = slot_of(id);
  if (!s) return -1;
  S = s;
  cur_slot = id;
  return 0;
}

KM_API(km_slot_current) int km_slot_current(void) { return cur_slot; }
KM_API(km_slot_max) int km_slot_max(void) { return KM_MAX_SLOTS; }
KM_API(km_slot_count) int km_slot_count(void) {
  int n = 0;
  for (int id = 0; id < KM_MAX_SLOTS; id++) n += M.slots[id].used;
  return n;
}
/* Memory the slot holds now: its blocks and its fixed-size part. */
KM_API(km_slot_bytes) uint32_t km_slot_bytes(int id) {
  const km_slot *s = slot_of(id);
  return s ? (uint32_t)((size_t)s->n_blocks * M.blk_bytes + M.aux_bytes) : 0;
}
KM_API(km_slot_n_past) int km_slot_n_past(int id) { const km_slot *s = slot_of(id); return s ? s->n_lower : -1; }
KM_API(km_slot_skill) int km_slot_skill(int id) { const km_slot *s = slot_of(id); return s ? s->active : -2; }
KM_API(km_block_bytes) uint32_t km_block_bytes(void) { return (uint32_t)M.blk_bytes; }
KM_API(km_block_positions) int km_block_positions(void) { return KM_BLOCK; }

/* The current slot's logits (valid after an evaluation reached the last position) and its additive bias on them. */
KM_API(km_logits) float *km_logits(void) { return S->logits; }
KM_API(km_bias) float *km_bias(void) { return S->bias; }
KM_API(km_token_at) int km_token_at(int pos) { return pos >= 0 && pos < S->n_lower ? *hist_ptr(S, pos) : -1; }
/* The hidden state at the input of the split layer for one cached position of the current slot. */
KM_API(km_state_at) float *km_state_at(int pos) { return pos >= 0 && pos < S->n_lower ? state_ptr(S, pos) : NULL; }
KM_API(km_n_past) int km_n_past(void) { return S->n_lower; }
KM_API(km_n_upper) int km_n_upper(void) { return S->n_upper; }

KM_API(km_reset) void km_reset(void) {
  if (!S->used) return;
  slot_trim(S, 0);
  S->n_lower = S->n_upper = 0;
  S->logits_ok = 0;
  S->head_pos = -1;
}

/* True when the logits belong to the last cached position and every cached position has its upper part. */
KM_API(km_logits_ready) int km_logits_ready(void) { return S->logits_ok && S->n_upper == S->n_lower && S->n_lower > 0; }

/* Keep the first n positions (any n; layers use full attention, so nothing else has to be undone). Returns the
   number of positions kept. */
KM_API(km_truncate) int km_truncate(int n) {
  if (n < 0) n = 0;
  if (n >= S->n_lower) return S->n_lower;
  S->n_lower = n;
  if (S->n_upper > n) S->n_upper = n;
  S->logits_ok = 0;
  slot_trim(S, n);
  return n;
}

/* ------------------------------------------------------------------ skills and head */
static int check_range(uint64_t off, uint64_t len, uint64_t size) { return off <= size && len <= size - off; }

static void skill_release(km_skill *s) {
  if (s->block) arena_free(s->block);
  __builtin_memset(s, 0, sizeof(*s));
}

/* A loaded skill is replaced or removed: no slot may keep using it or keep an upper cache computed under it. When
   it is removed, slots that selected it fall back to no skill. */
static void forget_skill(int skill, int removed) {
  for (int id = 0; id < KM_MAX_SLOTS; id++) {
    km_slot *s = &M.slots[id];
    if (s->upper_skill == skill) { s->n_upper = 0; s->logits_ok = 0; }
    if (removed && s->active == skill) s->active = -1;
  }
  skill_release(&M.skills[skill]);
}

/* Returns 0 or a negative code: -1 bad slot, -2 not a skill file, -3 truncated, -4 bad entry, -5 does not fit the
   trunk, -6 another loaded skill has a different first layer, -7 out of memory, -8 the context is not empty and the
   split would move (km_reset first). */
KM_API(km_skill_load) int km_skill_load(int slot, const uint8_t *p, uint32_t size) {
  if (!M.ready || slot < 0 || slot >= KM_SKILLS) return -1;
  km_skill_header hd;
  if (size < sizeof(hd)) return -2;
  __builtin_memcpy(&hd, p, sizeof(hd));
  if (hd.magic != KM_SKILL_MAGIC || hd.version != 1) return -2;
  const uint32_t n_layers = M.h.n_layers;
  if (hd.n_layers != n_layers || hd.first_layer >= n_layers) return -5;
  if (hd.n_entries > n_layers * MAT_COUNT || !check_range(sizeof(hd), (uint64_t)hd.n_entries * sizeof(km_skill_entry), size)) return -3;
  for (int i = 0; i < KM_SKILLS; i++)
    if (i != slot && M.skills[i].used && M.skills[i].first_layer != (int)hd.first_layer) return -6;
  int others = 0;
  for (int i = 0; i < KM_SKILLS; i++) others |= i != slot && M.skills[i].used;
  if (!others && M.split != (int)hd.first_layer)
    for (int id = 0; id < KM_MAX_SLOTS; id++)
      if (M.slots[id].used && M.slots[id].n_lower > 0) return -8;

  static const int in_dim_of[MAT_COUNT] = {0, 0, 0, 1, 0, 0, 2};  /* 0: dim, 1: n_heads * head_dim, 2: ffn_dim */
  const int dims[3] = {(int)M.h.dim, (int)(M.h.n_heads * M.h.head_dim), (int)M.h.ffn_dim};
  size_t floats = 0;
  uint8_t seen[KM_MAX_LAYERS][MAT_COUNT];
  __builtin_memset(seen, 0, sizeof(seen));
  for (uint32_t i = 0; i < hd.n_entries; i++) {
    km_skill_entry e;
    __builtin_memcpy(&e, p + sizeof(hd) + (size_t)i * sizeof(e), sizeof(e));
    if (e.layer < hd.first_layer || e.layer >= n_layers || e.matrix >= MAT_COUNT || e.rank < 1 || e.rank > KM_MAX_RANK ||
        seen[e.layer][e.matrix])
      return -4;
    seen[e.layer][e.matrix] = 1;
    const km_mat *W = &M.layers[e.layer].w[e.matrix];
    const uint64_t a_bytes = (uint64_t)e.rank * W->cols * 2, b_bytes = (uint64_t)W->rows * e.rank * 2;
    if (W->cols != dims[in_dim_of[e.matrix]]) return -5;
    if (!check_range(e.a_offset, a_bytes, size) || !check_range(e.b_offset, b_bytes, size)) return -3;
    floats += (size_t)e.rank * (W->cols + W->rows);
  }
  float *block = (float *)arena_alloc(floats * 4 + 64);
  if (!block) return -7;
  km_skill fresh;
  __builtin_memset(&fresh, 0, sizeof(fresh));
  fresh.used = 1;
  fresh.first_layer = (int)hd.first_layer;
  fresh.block = block;
  __builtin_memcpy(fresh.name, hd.name, 64);
  float *at = block;
  for (uint32_t i = 0; i < hd.n_entries; i++) {
    km_skill_entry e;
    __builtin_memcpy(&e, p + sizeof(hd) + (size_t)i * sizeof(e), sizeof(e));
    const km_mat *W = &M.layers[e.layer].w[e.matrix];
    const size_t in = (size_t)W->cols, out = (size_t)W->rows, r = e.rank;
    float *a = at, *bt = at + r * in;
    at = bt + r * out;
    const uint16_t *a16 = (const uint16_t *)(p + e.a_offset), *b16 = (const uint16_t *)(p + e.b_offset);
    for (size_t k = 0; k < r * in; k++) a[k] = km_f16(a16[k]);
    for (size_t o = 0; o < out; o++)
      for (size_t k = 0; k < r; k++) bt[k * out + o] = km_f16(b16[o * r + k]);
    km_delta *d = &fresh.d[e.layer][e.matrix];
    d->a = a;
    d->bt = bt;
    d->rank = (int)r;
    d->scale = e.scale;
  }
  if (M.skills[slot].used) forget_skill(slot, 0);
  M.skills[slot] = fresh;
  if (!others && M.split != (int)hd.first_layer) {
    M.split = (int)hd.first_layer;
    for (int id = 0; id < KM_MAX_SLOTS; id++) slot_clear(&M.slots[id]); /* all empty, checked above */
  }
  return 0;
}

KM_API(km_skill_unload) int km_skill_unload(int slot) {
  if (slot < 0 || slot >= KM_SKILLS || !M.skills[slot].used) return -1;
  forget_skill(slot, 1);
  return 0;
}

/* Layer whose input state the head reads, or -1 when it cannot be served (the layer lies above the split). */
static int head_layer(void) {
  const int l = M.head.layer < 0 ? M.split : M.head.layer;
  return l <= M.split ? l : -1;
}

/* Returns 0 or a negative code: -1 not a head file, -2 truncated, -3 does not fit the trunk. */
KM_API(km_head_load) int km_head_load(const uint8_t *p, uint32_t size) {
  if (!M.ready) return -1;
  km_head_header hd;
  if (size < sizeof(hd)) return -1;
  __builtin_memcpy(&hd, p, sizeof(hd));
  if (hd.magic != KM_HEAD_MAGIC || hd.version != 1) return -1;
  if (hd.in_dim != M.h.dim || !hd.hidden || hd.hidden > 4096 || !hd.n_out || hd.n_out > 256 ||
      (hd.layer != KM_NO_LAYER && hd.layer > M.h.n_layers) || hd.in_dim + hd.hidden > M.h.vocab)
    return -3;
  const size_t n = (size_t)hd.hidden * hd.in_dim + hd.hidden + (size_t)hd.n_out * hd.hidden + hd.n_out;
  if (!check_range(sizeof(hd), n * 4, size)) return -2;
  float *block = (float *)arena_alloc(n * 4 + 64);
  if (!block) return -3;
  __builtin_memcpy(block, p + sizeof(hd), n * 4);
  if (M.head.used) arena_free(M.head.block);
  M.head.used = 1;
  M.head.layer = hd.layer == KM_NO_LAYER ? -1 : (int)hd.layer;
  M.head.in_dim = (int)hd.in_dim;
  M.head.hidden = (int)hd.hidden;
  M.head.n_out = (int)hd.n_out;
  M.head.eps = hd.eps;
  M.head.block = block;
  M.head.w1 = block;
  M.head.b1 = M.head.w1 + (size_t)hd.hidden * hd.in_dim;
  M.head.w2 = M.head.b1 + hd.hidden;
  M.head.b2 = M.head.w2 + (size_t)hd.n_out * hd.hidden;
  for (int id = 0; id < KM_MAX_SLOTS; id++) M.slots[id].head_pos = -1;
  return 0;
}

KM_API(km_head_outputs) int km_head_outputs(void) { return M.head.used ? M.head.n_out : 0; }

/* Writes the head's n_out raw outputs for the last evaluated position and returns n_out. Negative: -1 no head,
   -2 the head's layer lies above the split, -3 no valid state (nothing evaluated since the last truncation). */
KM_API(km_control) int km_control(float *out) {
  if (!M.head.used) return -1;
  if (head_layer() < 0) return -2;
  if (S->n_lower == 0 || S->head_pos != S->n_lower - 1) return -3;
  const km_head *H = &M.head;
  const float *state = S->head_state;
  float ss = 0.0f;
  for (int i = 0; i < H->in_dim; i++) ss += state[i] * state[i];
  const float r = km_rsqrt(ss / (float)H->in_dim + H->eps);
  float *x = M.work, *hid = M.work + H->in_dim;
  for (int i = 0; i < H->in_dim; i++) x[i] = state[i] * r;
  for (int j = 0; j < H->hidden; j++) {
    const float v = dot_f32(H->w1 + (size_t)j * H->in_dim, x, H->in_dim) + H->b1[j];
    hid[j] = v > 0.0f ? v : 0.0f;
  }
  for (int o = 0; o < H->n_out; o++) out[o] = dot_f32(H->w2 + (size_t)o * H->hidden, hid, H->hidden) + H->b2[o];
  return H->n_out;
}

/* ------------------------------------------------------------------ forward */
static void rope_apply(float *v, int pos, int hd) {
  const int half = hd / 2;
  const float *c = M.rope + (size_t)pos * half, *s = M.rope + (size_t)M.max_ctx * half + (size_t)pos * half;
  for (int i = 0; i < half; i++) {
    float a = v[i], b = v[i + half];
    v[i] = a * c[i] - b * s[i];
    v[i + half] = b * c[i] + a * s[i];
  }
}

static void store_kv(int8_t *dst, float *scale, const float *src, int n) {
  for (int g = 0; g < n / KM_KVG; g++) {
    float tmp_scale;
    int32_t unused;
    quantize_group(src + g * KM_KVG, dst + g * KM_KVG, &tmp_scale, &unused);
    scale[g] = tmp_scale;
  }
}

static void attention_task(int t, int h, int thread) {
  const int NH = (int)M.h.n_heads, KV = (int)M.h.n_kv_heads, HD = (int)M.h.head_dim, kh = h / (NH / KV);
  const int NG = HD / KM_KVG, pos = J.pos0 + t;
  const float inv = km_rsqrt((float)HD);
  const float *q = M.q + ((size_t)t * NH + h) * HD;
  float *out = M.att + ((size_t)t * NH + h) * HD;
  float *sc = M.scores + (size_t)thread * M.max_ctx;
  const float *ksink = S->sink + (size_t)J.layer * 2 * KM_SINK * KV * HD, *vsink = ksink + (size_t)KM_SINK * KV * HD;

  /* the query as int16 against the int8 keys: integer dot products, scale applied per group of 32 */
  int16_t q16[KM_MAX_HEAD_DIM];
  float amax = 0.0f;
  for (int i = 0; i < HD; i++) { float a = q[i] < 0 ? -q[i] : q[i]; amax = a > amax ? a : amax; }
  const float qscale = amax / 32767.0f, qinv = amax > 0.0f ? 32767.0f / amax : 0.0f;
  for (int i = 0; i < HD; i++) q16[i] = (int16_t)(q[i] * qinv + (q[i] < 0 ? -0.5f : 0.5f));

  float mx = -3.0e38f;
  for (int p0 = 0; p0 <= pos; p0 += KM_BLOCK) {
    const km_kvblock kb = kv_block(S, J.layer, p0 >> KM_BLOCK_SHIFT);
    const int p1 = pos + 1 < p0 + KM_BLOCK ? pos + 1 : p0 + KM_BLOCK;
    for (int p = p0; p < p1; p++) {
      const size_t o = (size_t)(p - p0) * KV + kh;
      float s;
      if (p < KM_SINK) s = dot_f32(q, ksink + ((size_t)p * KV + kh) * HD, HD);
      else {
        const int8_t *k = kb.kc + o * HD;
        const float *ks = kb.ks + o * NG;
        float total = 0.0f;
        for (int g = 0; g < NG; g++) {
          const int16_t *qg = q16 + g * KM_KVG;
          const int8_t *kg = k + g * KM_KVG;
#if defined(__wasm_simd128__)
          v128_t acc = wasm_i32x4_splat(0);
          for (int i = 0; i < KM_KVG; i += 8)
            acc = wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(wasm_v128_load(qg + i), wasm_i16x8_load8x8(kg + i)));
          const int32_t sum = wasm_i32x4_extract_lane(acc, 0) + wasm_i32x4_extract_lane(acc, 1) +
                              wasm_i32x4_extract_lane(acc, 2) + wasm_i32x4_extract_lane(acc, 3);
#else
          int32_t sum = 0;
          for (int i = 0; i < KM_KVG; i++) sum += (int32_t)qg[i] * kg[i];
#endif
          total += (float)sum * ks[g];
        }
        s = total * qscale;
      }
      s *= inv;
      sc[p] = s;
      mx = s > mx ? s : mx;
    }
  }
  float sum = 0.0f;
  for (int p = 0; p <= pos; p++) sum += (sc[p] = km_expf(sc[p] - mx));
  const float isum = 1.0f / sum;
  for (int i = 0; i < HD; i++) out[i] = 0.0f;
  for (int p0 = 0; p0 <= pos; p0 += KM_BLOCK) {
    const km_kvblock kb = kv_block(S, J.layer, p0 >> KM_BLOCK_SHIFT);
    const int p1 = pos + 1 < p0 + KM_BLOCK ? pos + 1 : p0 + KM_BLOCK;
    for (int p = p0; p < p1; p++) {
      const size_t o = (size_t)(p - p0) * KV + kh;
      const float a = sc[p] * isum;
      if (p < KM_SINK) {
        const float *v = vsink + ((size_t)p * KV + kh) * HD;
        for (int i = 0; i < HD; i++) out[i] += a * v[i];
        continue;
      }
      const float *vs = kb.vs + o * NG;
      const int8_t *v = kb.vc + o * HD;
      for (int g = 0; g < NG; g++) {
#if defined(__wasm_simd128__)
        const v128_t av = wasm_f32x4_splat(a * vs[g]);
        for (int i = 0; i < KM_KVG; i += 8) {
          const v128_t w = wasm_i16x8_load8x8(v + g * KM_KVG + i);
          float *dst = out + g * KM_KVG + i;
          wasm_v128_store(dst, wasm_f32x4_add(wasm_v128_load(dst), wasm_f32x4_mul(av, wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(w)))));
          wasm_v128_store(dst + 4, wasm_f32x4_add(wasm_v128_load(dst + 4), wasm_f32x4_mul(av, wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(w)))));
        }
#else
        const float c = a * vs[g];
        for (int i = 0; i < KM_KVG; i++) out[g * KM_KVG + i] += c * (float)v[g * KM_KVG + i];
#endif
      }
    }
  }
}

static const km_delta *delta_of(const km_skill *sk, int l, int m) {
  return sk && sk->d[l][m].rank ? &sk->d[l][m] : NULL;
}

/* One transformer layer over the T tokens in M.x, at positions pos0 .. pos0 + T - 1. */
static void layer_forward(int l, int T, int pos0, const km_skill *sk) {
  const int D = (int)M.h.dim, NH = (int)M.h.n_heads, KV = (int)M.h.n_kv_heads, HD = (int)M.h.head_dim;
  const float eps = M.h.norm_eps;
  km_layer *L = &M.layers[l];
  for (int t = 0; t < T; t++) rmsnorm(M.xb + (size_t)t * D, M.x + (size_t)t * D, L->attn_norm, D, eps);
  {
    const km_mat *Ws[3] = {&L->w[MAT_Q], &L->w[MAT_K], &L->w[MAT_V]};
    const km_delta *ds[3] = {delta_of(sk, l, MAT_Q), delta_of(sk, l, MAT_K), delta_of(sk, l, MAT_V)};
    float *outs[3] = {M.q, M.k, M.v};
    matmul_multi(3, Ws, sk ? ds : NULL, outs, M.xb, T, MODE_STORE);
  }
  for (int t = 0; t < T; t++) {
    const int pos = pos0 + t;
    for (int h = 0; h < NH; h++) {
      float *qh = M.q + ((size_t)t * NH + h) * HD;
      rmsnorm(qh, qh, L->q_norm, HD, eps);
      rope_apply(qh, pos, HD);
    }
    for (int h = 0; h < KV; h++) {
      float *kh = M.k + ((size_t)t * KV + h) * HD, *vh = M.v + ((size_t)t * KV + h) * HD;
      rmsnorm(kh, kh, L->k_norm, HD, eps);
      rope_apply(kh, pos, HD);
      const km_kvblock kb = kv_block(S, l, pos >> KM_BLOCK_SHIFT);
      const size_t o = pos_in_block(pos) * KV + h;
      store_kv(kb.kc + o * HD, kb.ks + o * (HD / KM_KVG), kh, HD);
      store_kv(kb.vc + o * HD, kb.vs + o * (HD / KM_KVG), vh, HD);
      if (pos < KM_SINK) {
        float *sink = S->sink + (size_t)l * 2 * KM_SINK * KV * HD;
        __builtin_memcpy(sink + ((size_t)pos * KV + h) * HD, kh, HD * 4);
        __builtin_memcpy(sink + ((size_t)(KM_SINK + pos) * KV + h) * HD, vh, HD * 4);
      }
    }
  }
  J.kind = JOB_ATTENTION;
  J.layer = l;
  J.T = T;
  J.pos0 = pos0;
  dispatch();
  matmul(M.x, &L->w[MAT_O], delta_of(sk, l, MAT_O), M.att, T, MODE_ADD);
  for (int t = 0; t < T; t++) rmsnorm(M.xb + (size_t)t * D, M.x + (size_t)t * D, L->ffn_norm, D, eps);
  {
    const km_mat *Ws[2] = {&L->w[MAT_GATE], &L->w[MAT_UP]};
    const km_delta *ds[2] = {delta_of(sk, l, MAT_GATE), delta_of(sk, l, MAT_UP)};
    float *outs[2] = {M.ffa, M.ffa};
    matmul_multi(2, Ws, sk ? ds : NULL, outs, M.xb, T, MODE_SILU_MUL);
  }
  matmul(M.x, &L->w[MAT_DOWN], delta_of(sk, l, MAT_DOWN), M.ffa, T, MODE_ADD);
}

static void embed_row(float *o, int token) {
  const km_mat *E = &M.emb;
  const int cols = E->cols, ng = cols / KM_GROUP;
  if (E->type == KM_F32) {
    __builtin_memcpy(o, (const float *)E->w + (size_t)token * cols, (size_t)cols * sizeof(float));
    return;
  }
  const size_t quad = (size_t)token >> 2, lane = (size_t)token & 3;
  for (int g = 0; g < ng; g++) {
    const float s = E->s[(quad * ng + g) * 4 + lane];
    if (E->type == KM_Q4) {
      const uint8_t *blk = E->w + (quad * ng + g) * 64;
      for (int j = 0; j < 4; j++)
        for (int b = 0; b < 4; b++) {
          const uint8_t byte = blk[16 * j + 4 * lane + b];
          o[g * KM_GROUP + 4 * j + b] = (float)((int)(byte & 15) - 8) * s;
          o[g * KM_GROUP + 16 + 4 * j + b] = (float)((int)(byte >> 4) - 8) * s;
        }
    } else {
      const int8_t *blk = (const int8_t *)E->w + (quad * ng + g) * 128;
      for (int j = 0; j < 8; j++)
        for (int b = 0; b < 4; b++) o[g * KM_GROUP + 4 * j + b] = (float)blk[16 * j + 4 * lane + b] * s;
    }
  }
}

/* Layers [from, to) over the chunk in M.x. */
static void layers_forward(int from, int to, int T, int pos0, const km_skill *sk) {
  for (int l = from; l < to; l++) layer_forward(l, T, pos0, sk);
}

/* ------------------------------------------------------------------ evaluation */
/* Append n tokens through the layers below the split. Returns 0, -1 not ready (or no current slot), -2 bad token,
   -3 context full, -4 out of memory. */
KM_API(km_eval_lower) int km_eval_lower(const int32_t *tokens, int n) {
  if (!M.ready || !S->used) return -1;
  if (n <= 0) return 0;
  if (S->n_lower + n > M.max_ctx) return -3;
  for (int i = 0; i < n; i++)
    if (tokens[i] < 0 || tokens[i] >= (int)M.h.vocab) return -2;
  if (!slot_reserve(S, S->n_lower + n)) return -4;
  const int D = (int)M.h.dim, hl = M.head.used ? head_layer() : -1;
  for (int i = 0; i < n; i += KM_CHUNK) {
    const int T = n - i < KM_CHUNK ? n - i : KM_CHUNK, pos0 = S->n_lower;
    for (int t = 0; t < T; t++) {
      embed_row(M.x + (size_t)t * D, tokens[i + t]);
      *hist_ptr(S, pos0 + t) = tokens[i + t];
    }
    if (hl >= 0 && hl < M.split) {
      layers_forward(0, hl, T, pos0, NULL);
      __builtin_memcpy(S->head_state, M.x + (size_t)(T - 1) * D, (size_t)D * 4);
      layers_forward(hl, M.split, T, pos0, NULL);
    } else {
      layers_forward(0, M.split, T, pos0, NULL);
    }
    for (int t = 0; t < T; t++) __builtin_memcpy(state_ptr(S, pos0 + t), M.x + (size_t)t * D, (size_t)D * 4);
    if (hl == M.split) __builtin_memcpy(S->head_state, M.x + (size_t)(T - 1) * D, (size_t)D * 4);
    S->n_lower += T;
  }
  S->head_pos = hl >= 0 ? S->n_lower - 1 : -1;
  S->logits_ok = 0;
  return 0;
}

/* Run the layers from the split up over the positions whose upper part is missing, under the active skill, at most
   `max` positions in this call (in pieces of at most KM_CHUNK). Returns how many
   positions still miss their upper part (0 = done: the logits of the last position are in km_logits()), or a
   negative code. If the skill differs from the one the upper cache was built under, the cache is dropped first and
   rebuilt from the stored states. */
KM_API(km_eval_upper_n) int km_eval_upper_n(int max) {
  if (!M.ready || !S->used) return -1;
  if (S->n_upper > 0 && S->upper_skill != S->active) {
    S->n_upper = 0;
    S->logits_ok = 0;
  }
  S->upper_skill = S->active;
  const int D = (int)M.h.dim;
  const km_skill *sk = S->active >= 0 ? &M.skills[S->active] : NULL;
  for (int p = S->n_upper; p < S->n_lower && max > 0;) {
    int T = S->n_lower - p < KM_CHUNK ? S->n_lower - p : KM_CHUNK;
    if (T > max) T = max;
    for (int t = 0; t < T; t++) __builtin_memcpy(M.x + (size_t)t * D, state_ptr(S, p + t), (size_t)D * 4);
    layers_forward(M.split, (int)M.h.n_layers, T, p, sk);
    p += T;
    max -= T;
    S->n_upper = p;
    if (p == S->n_lower) {
      rmsnorm(M.xb, M.x + (size_t)(T - 1) * D, M.final_norm, D, M.h.norm_eps);
      matmul(S->logits, &M.emb, NULL, M.xb, 1, MODE_STORE);
      S->logits_ok = 1;
    }
  }
  return S->n_lower - S->n_upper;
}

KM_API(km_eval_upper) int km_eval_upper(void) {
  const int rc = km_eval_upper_n(0x7fffffff);
  return rc < 0 ? rc : 0;
}

KM_API(km_eval) int km_eval(const int32_t *tokens, int n) {
  const int rc = km_eval_lower(tokens, n);
  return rc ? rc : km_eval_upper();
}

/* Select the skill for the upper layers of the current slot (-1: none) without computing anything; the next
   km_eval_upper(_n) recomputes whatever was built under another skill. Returns 0, or -1 for an empty skill slot. */
KM_API(km_skill_select) int km_skill_select(int skill) {
  if (!M.ready || !S->used || skill < -1 || skill >= KM_SKILLS || (skill >= 0 && !M.skills[skill].used)) return -1;
  S->active = skill;
  return 0;
}

/* km_skill_select followed by the whole recomputation. */
KM_API(km_skill_use) int km_skill_use(int skill) {
  return km_skill_select(skill) ? -1 : km_eval_upper();
}

KM_API(km_active_skill) int km_active_skill(void) { return S->active; }

/* ------------------------------------------------------------------ sampling */
static inline uint32_t rng_next(void) {
  S->rng ^= S->rng << 13;
  S->rng ^= S->rng >> 7;
  S->rng ^= S->rng << 17;
  return (uint32_t)(S->rng >> 32);
}

KM_API(km_seed) void km_seed(uint32_t seed) { S->rng = 0x9e3779b97f4a7c15ull ^ ((uint64_t)seed << 1 | 1); }

KM_API(km_argmax) int km_argmax(void) {
  int best = 0;
  for (int i = 1; i < (int)M.h.vocab; i++)
    if (S->logits[i] + S->bias[i] > S->logits[best] + S->bias[best]) best = i;
  return best;
}

/* Repetition penalty over the last `window` tokens (Hugging Face convention), then temperature, top-k, top-p and
   min-p. temperature <= 0 selects the argmax. */
KM_API(km_sample) int km_sample(float temperature, int top_k, float top_p, float min_p, float rep_penalty, int window) {
  const int V = (int)M.h.vocab;
  float *w = M.work;
  for (int i = 0; i < V; i++) w[i] = S->logits[i] + S->bias[i];
  if (rep_penalty > 1.0f && window > 0) {
    const int from = S->n_lower - window > 0 ? S->n_lower - window : 0;
    for (int p = from; p < S->n_lower; p++) {
      const int tok = *hist_ptr(S, p);
      if (M.flag[tok]) continue;
      M.flag[tok] = 1;
      w[tok] = w[tok] > 0 ? w[tok] / rep_penalty : w[tok] * rep_penalty;
    }
    for (int p = from; p < S->n_lower; p++) M.flag[*hist_ptr(S, p)] = 0;
  }
  if (temperature <= 0.0f) {
    int best = 0;
    for (int i = 1; i < V; i++) if (w[i] > w[best]) best = i;
    return best;
  }
  if (top_k <= 0 || top_k > 256) top_k = 256;
  int top[256], n = 0;
  for (int i = 0; i < V; i++) {
    const float v = w[i];
    if (n == top_k && v <= w[top[n - 1]]) continue;
    int j = n < top_k ? n++ : n - 1;
    while (j > 0 && w[top[j - 1]] < v) { top[j] = top[j - 1]; j--; }
    top[j] = i;
  }
  float probs[256], total = 0.0f;
  const float mx = w[top[0]];
  for (int j = 0; j < n; j++) total += (probs[j] = km_expf((w[top[j]] - mx) / temperature));
  int keep = n;
  float cum = 0.0f;
  for (int j = 0; j < n; j++) {
    if (probs[j] / total < min_p * (probs[0] / total)) { keep = j; break; }
    cum += probs[j] / total;
    if (cum >= top_p) { keep = j + 1; break; }
  }
  if (keep < 1) keep = 1;
  float mass = 0.0f;
  for (int j = 0; j < keep; j++) mass += probs[j];
  float r = (float)(rng_next() >> 8) * (1.0f / 16777216.0f) * mass;
  int chosen = top[keep - 1];
  for (int j = 0; j < keep; j++) {
    r -= probs[j];
    if (r <= 0.0f) { chosen = top[j]; break; }
  }
  return chosen;
}

/* ------------------------------------------------------------------ test hook */
/* out[t * rows + r] = W[r] . x[t] for T activation vectors through the production matrix product. w and s16 hold
   one tensor as the file stores it (Q4/Q8 codes row by row, then fp16 scales; or f32 rows, s16 unused). The
   buffers are consumed (repacked in place). Returns 0, or a negative code if the matrix is not acceptable. */
KM_API(km_test_matmul) int km_test_matmul(int type, int rows, int cols, int T, uint8_t *w, const float *x, float *out) {
  if (T < 1 || T > KM_CHUNK || !alloc_activations((size_t)cols)) return -1;
  km_tensor t = {{0}, (uint32_t)type, (uint32_t)rows, (uint32_t)cols, 0, 0, 0};
  const size_t ng = (size_t)cols / KM_GROUP;
  t.bytes = type == KM_F32 ? (uint64_t)rows * cols * 4 : (uint64_t)rows * cols / (type == KM_Q4 ? 2 : 1) + (uint64_t)rows * ng * 2;
  km_mat W;
  if (!bind_mat(&W, w, &t, (uint32_t)rows, (uint32_t)cols)) return -2;
  matmul(out, &W, NULL, x, T, MODE_STORE);
  return 0;
}
