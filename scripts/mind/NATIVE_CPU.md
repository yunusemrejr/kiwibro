# Profiling Gemma 4 CPU decode

The diagnostic patch targets llama.cpp [b12443](https://github.com/ggml-org/llama.cpp/tree/301db633702f1d2c107d4feaa4268f07c0b508c5). It records elapsed CPU graph time through each operation's thread barrier. Turn it on with `GGML_KIWI_CPU_PROFILE=1` in a separate benchmark runtime. It emits one line per graph with single-token MoE work, including dense and expert totals by weight type. The times include scheduling and synchronization. Fused work is attributed to the first operation. A large projection is a matrix multiply with at least 32,768 output rows.

## Reproduce the trace

With an unpacked Linux x86-64 b12443 runtime, an existing Gemma 4 26B A4B GGUF and a separate llama.cpp checkout at the revision above:

```sh
git -C "$llama_source" apply "$kiwibro_source/scripts/mind/ggml_cpu_decode_profile.patch"
cmake -S "$llama_source" -B "$llama_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF
cmake --build "$llama_build" --target ggml-cpu-haswell --parallel 2
trace_runtime=$(mktemp -d)
cp -a "$llama_release/." "$trace_runtime/"
rm "$trace_runtime"/libggml-cpu-*.so
cp "$llama_build/bin/libggml-cpu-haswell.so" "$trace_runtime/"
GGML_KIWI_CPU_PROFILE=1 LD_LIBRARY_PATH="$trace_runtime" \
  "$trace_runtime/llama-bench" --device none -m "$gemma_gguf" \
  -p 0 -n 24 -d 256 -ngl 0 -t 6 -fa on -r 1 -o json \
  > gemma-decode.json 2> gemma-decode.log
rg '^KIWI_CPU_DECODE ' gemma-decode.log | tail -24
```

The Haswell backend requires AVX2, FMA, F16C and BMI2. The final 24 trace lines above are the timed decode tokens. Earlier lines can include graph warmup and the initial context fill, whose last output may also have single-token MoE work. All trace durations are in microseconds. Use the uninstrumented runtime for speed comparisons.

## Measurements

Gemma 4 26B A4B UD-IQ3_XXS on a Ryzen 5 7530U with six threads and 256 tokens of context. The table reports medians across 24 warm CPU decode graphs. The vocabulary projection is part of dense Q6_K.

| CPU graph operation | Median elapsed | Share of median graph time |
| --- | ---: | ---: |
| Whole graph | 200.07 ms | 100% |
| Dense matrix multiply | 130.91 ms | 65.4% |
| ↳ Dense Q6_K | 112.35 ms | 56.2% |
| ↳ Tied 262,144-row vocabulary projection | 35.01 ms | 17.5% |
| Expert matrix multiply | 50.17 ms | 25.1% |
| ↳ IQ2_S expert weights | 29.65 ms | 14.8% |
| ↳ IQ4_NL expert weights | 18.36 ms | 9.2% |
| Flash attention | 8.88 ms | 4.4% |

Two AVX2 experiments had bit-for-bit parity on sampled weights from this GGUF. The IQ2_S gather rewrite took 10.56 ms against 10.10 ms per row sweep. The Q6_K scale shuffle rewrite took 11.52 ms against 10.80 ms. These single-thread comparisons use medians of 12 paired trials with alternating execution order. The samples contained 33,792 expert rows and 49,152 vocabulary rows respectively, with 13 seeded activations including warmup.

## Wide matrix-vector scheduling experiment

In `ggml_compute_forward_mul_mat()`, the experiment raised the dynamic chunk size from 64 to 256 rows when `nr1 == 1`, `nr0 >= 32768` and the weight type is quantized. With six threads, 49,152 real Q6_K vocabulary rows agreed bit for bit on all 25 seeded input vectors. The median paired time gain was 12.3% in the unpinned row benchmark and 1.8% when pinned to one logical CPU per physical core.

For the whole model, both backends used the same b12443 build configuration, with the profiling patch removed. The benchmark used the command above with `-r 2` and two runs per backend. Each row reports both warm decode samples in order. CPU pressure is Linux's `some avg10` from `/proc/pressure/cpu` before the arm started.

| Arm | Warm decode samples | CPU pressure at start |
| --- | ---: | ---: |
| Stock chunks | 3.665, 3.451 tok/s | 5.84% |
| 256-row chunks | 2.435, 1.418 tok/s | 20.87% |
| 256-row chunks | 3.185, 3.391 tok/s | 41.91% |
| Stock chunks | 4.982, 4.868 tok/s | 28.17% |

The experiment did not establish a full-model speedup under that desktop load. KiwiBro retains stock CPU kernels. These measurements cover the stock IQ3 GGUF and CPU-only inference. Compact Q4 and Vulkan runs require their own measurements.

## Expert channel compaction experiment

The 80-expert pilot keeps six active experts and all 30 layers. The first Q4_K_M pilot header has 30 fused gate/up matrices in Q4_K, 16 down matrices in Q5_0 and 14 down matrices in Q8_0. The original 704-wide down rows cannot use 256-element K-quant blocks. llama.cpp b12443 falls back from Q4_K to Q5_0 and from Q6_K to Q8_0 for those rows. [Pinned quantization fallback](https://github.com/ggml-org/llama.cpp/blob/301db633702f1d2c107d4feaa4268f07c0b508c5/src/llama-quant.cpp#L458)

| Expert width | Gate/up weights | Down weights | Estimated full GGUF | Expert MACs per token |
| --- | ---: | ---: | ---: | ---: |
| 704, pilot | 5,104.69 MiB | 3,913.59 MiB | 10.34 GiB | 1.071 billion |
| 512, selected | 3,712.50 MiB | 2,253.28 MiB | 7.36 GiB | 0.779 billion |
| 768, zero padded | 5,568.75 MiB | 3,379.92 MiB | 10.28 GiB | 1.168 billion |

Estimates keep every non-expert tensor unchanged and assume the 16 down tensors use Q4_K and the other 14 use Q6_K at the aligned widths. MACs count gate, up and down projections only. Whole-model speed depends on the measured fraction of time in these projections, memory pressure and kernel behavior. The Radeon Renoir device reports no integer-dot or matrix-core support. b12443 supplies dequantizing expert kernels for both the original fallback types and the aligned types. [Pinned Vulkan dispatch](https://github.com/ggml-org/llama.cpp/blob/301db633702f1d2c107d4feaa4268f07c0b508c5/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6069)

The 512 experiment selects a fixed number of channels independently per expert. It ranks them by the product of squared L2 norms from the gate row, up row and down column. It copies the corresponding BF16 entries in ascending original channel order and preserves router scales, expert IDs, top-k, vision, attention and tokenizer assets. The export processes one expert at a time and writes a new safetensors checkpoint with the selected masks in its architecture report.

Static salience uses weights only. Its retained score share is a diagnostic, with task accuracy measured separately. Activation calibration can collect routed, family-stratified channel energies from `GELU(gate) * up` and weight them by each down column's norm. Evaluate a fresh Colab adapter on the resulting checkpoint if the width change needs recovery. The existing fit wrapper accepts the new architecture report and packs experts from their resulting shapes. A 768-wide control can pad both halves of the fused gate/up matrix and the corresponding down columns with zeros. That preserves the original expert function before quantization while the new quantization blocks can change numerical results.

### Paired export and evaluation

Start from the same merged BF16 checkpoint for both arms. The checkpoint must carry the pinned base revision in `architecture.json`. Run from the KiwiBro checkout with NumPy and the pinned llama.cpp conversion dependencies:

```sh
python3 -m scripts.mind.compact_gemma4_channels "$bf16_704" "$bf16_512"
for width in 704 512; do
  if [ "$width" = 704 ]; then checkpoint="$bf16_704"; else checkpoint="$bf16_512"; fi
  python3 "$llama_source/convert_hf_to_gguf.py" "$checkpoint" \
    --outtype bf16 --outfile "$results/gemma-${width}-bf16.gguf"
  "$llama_release/llama-quantize" "$results/gemma-${width}-bf16.gguf" \
    "$results/gemma-${width}-q4_k_m.gguf" Q4_K_M 4
done
```

Compare the BF16 sources or identical frozen-expert storage in Colab to isolate channel removal. Then compare default Q4_K_M on the target laptop to measure the combined width and format effect. An additional 512 arm can pass `--tensor-type-file` to keep the original per-layer down formats. Build the type file with one exact regex per down weight, such as `^blk\.0\.ffn_down_exps\.weight$=q5_0`, using the verified 704 GGUF header to choose each type. Use fresh width-specific statistics if applying an importance matrix.

Run only one server at a time. Use the same runtime, backend, context, one slot, projector, sampler, token budget, prompts and held-out probes for both GGUFs. Bind the model and runtime identities to the result. Run `scripts.mind.replay_successor` on the same frozen probes at seed 7530 and `scripts.mind.bench_successor` for uncached and cached TTFT, decode rate and peak RAM. Check exact tool calls, visual fixtures, retrieval, and full workspace episodes before choosing a candidate. The table above is a footprint estimate. The static 512 export scored 0 of 25 available exact next-action probes in Colab with thinking off, so it stays experimental. Its local full-model latency remains unmeasured.

### Tied projection Vulkan experiment

The stock IQ3 and compact Q4 GGUFs both have a 577.5 MiB tied Q6_K embedding/projection tensor. This test places it on Vulkan with zero GPU layers and CPU KV. Begin with at least 14 GiB host memory available and 2 GiB unused GTT, then monitor both during the run. These are conservative admission values based on observed host pressure and the tensor size.

```bash
# Both variables point to verified local artifacts.
args=(--device Vulkan0 -m "$gguf" -ngl 0 -nkvo 1 -nopo 1
      -b 128 -ub 128 -t 6 -fa on -p 0 -n 24 -d 256 -r 2 -o json)
overrides=(-ot '^token_embd\.weight$=Vulkan0')
LD_LIBRARY_PATH="$llama_release" "$llama_release/llama-bench" "${args[@]}" > cpu-head.json
LD_LIBRARY_PATH="$llama_release" "$llama_release/llama-bench" "${args[@]}" "${overrides[@]}" > vulkan-head.json
```

Repeat in reverse order. A short separate probe with `GGML_SCHED_DEBUG=2` and `llama-bench -v` verifies that the logits multiply follows the tied weight onto Vulkan. Run the timed trials with scheduler debugging disabled.

On the stock IQ3 GGUF, the placement trace put the tied weight and output multiply on Renoir. The timed sequence used six threads and decoded 24 tokens at a 256-token depth per sample.

| Arm | Warm decode samples | CPU pressure at start |
| --- | ---: | ---: |
| CPU projection | 5.741, 6.050 tok/s | 0.66% |
| Vulkan projection | 5.037, 5.021 tok/s | 14.45% |
| Vulkan projection | 2.969, 5.362 tok/s | 7.40% |
| CPU projection | 5.168, 5.282 tok/s | 19.05% |

The pooled medians were 5.51 tok/s on CPU and 5.03 tok/s with the Vulkan tensor placement. Desktop pressure varied between runs. The stock-GGUF test supplies no speed evidence for enabling this placement on the compact model.

## Compact pilot on the desktop

The 28-step Q4_K_M pilot has 80 stored experts, six active per token, and an
8,192-token serving window. Its text GGUF is 11,105,979,328 bytes. These
application requests asked for the same exact marker in a local text file,
used six threads, and repeated the request twice on the live engine.
The native Gemma profiles used one slot. Cold wall time includes model
startup. Desktop load varied across the separately run controls.

| Pack and profile | Cold | First warm | Second warm |
| --- | ---: | ---: | ---: |
| Installed Spark pack, default GPU placement | 30.1 s | 16.2 s | 8.6 s |
| Stock Gemma, 10 Vulkan layers | 27.8 s | 7.3 s | 2.2 s |
| Compact Gemma, 10 Vulkan layers | 28.8 s | 5.8 s | 2.1 s |
| Compact Gemma, CPU | 40.8 s | 14.4 s | 3.4 s |

Both Gemma profiles retrieved the file before generation and sent its result
in one model request. In the compact Vulkan run, peak system GTT was 13,762
MiB and available host memory bottomed out at 10,233 MiB. The CPU run left at
least 13,115 MiB available. These are measurements for the small read
workload. The same 10-layer compact profile reached the live acceptance
test's 300-second limit on an interactive 3D HTML task before it completed
its first tool call.

A separate raw test of the compact pilot observed the following across one
tool call, one screenshot and two retrieval requests with identical prompts.
Retrieval used 36 filler lines. Full Vulkan passed all four probes in
68.5 seconds after startup. The 15-layer profile failed its first retrieval
after Vulkan reported `ErrorDeviceLost`. In a follow-up test using eight
filler lines, placing dense weight operations on Vulkan and experts in host
memory passed all four probes in 136.9 seconds. Adding a 1,024 MiB expert
cache brought that shorter test to 61.0 seconds. Profile timings include
different desktop load. They record candidate behavior on this laptop.

## Repeated text on CPU

The managed b12443 runtime can draft tokens from repeated sequences in the
current context. It verifies the draft against the same model and needs no
extra model file. The experimental CPU profile uses `--spec-type
ngram-mod`, a 24-token match, and a 48–64 token draft. Tool calls and images
still use the native template and the same vision projector.

The compact Q4_K_M pilot ran with six threads, one 8,192-token slot and the
matching projector. The two passes reversed runtime order. Times exclude
server startup. All twelve probe requests passed.

| Probe | No speculative decoding | Context drafts |
| --- | ---: | ---: |
| Exact code search tool call | 9.14–9.55 s | 10.34–10.39 s |
| Read code from a screenshot | 19.92–22.14 s | 21.72–23.81 s |
| Copy 14 C declarations | 50.44–61.56 s | 21.50–22.88 s |

Copy throughput was 4.41–5.52 tokens/s without drafts and 16.10–16.83
tokens/s with drafts. The copy workload measures repeated code. A fresh
solution, image encoding, or short tool request sees different costs.
Through the app's chat stream with reasoning Off, both CPU arms returned all 14
declarations. Cold completion fell from 78.25 to 51.78 seconds. Time to first text
was 31.13 and 32.75 seconds respectively. An earlier Auto run reached its
180-second deadline under desktop load. The app profile remains experimental.

For each arm, run a single pinned server with `--device none --gpu-layers 0
--parallel 1 --threads 6 --threads-batch 6 --ctx-size 8192 --cache-ram 0
--flash-attn on --jinja --no-webui`, the same model and the same projector.
Use `--spec-type none` for the control and `--spec-type ngram-mod
--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`
for the draft arm. Repeat in reverse order. With the server listening on the
loopback endpoint and the synthetic model ID supplied as `auto`:

```bash
python3 -m scripts.mind.bench_successor \
  --endpoint cpu http://127.0.0.1:8081 auto --thinking off \
  --case tool-symbol --case screenshot-release --case preserve-code-block \
  --output cpu-arm.json
```

### Request-scoped n-gram drafting experiment

The pinned b12443 server disables its original request-level speculative settings
in the schema. The optional [request patch](llama_b12443_ngram_request_gate.patch)
adds the flat JSON boolean `"speculative.ngram_mod.enabled": false`. Its scope
is a single completion on a server launched with `--spec-type ngram-mod`.
Omitted, null and true values keep existing behavior. Requests on a server
configured with another or mixed draft methods keep their existing behavior as
well. Slot state and server-wide resources still follow the runtime's normal
lifecycle. This skips n-gram prompt indexing and draft generation for that
request.

Apply the patch to a clean [b12443 source tree](https://github.com/ggml-org/llama.cpp/tree/301db633702f1d2c107d4feaa4268f07c0b508c5)
and configure a Ninja build with the server tools enabled. The changed
translation units compile on their own:

```sh
git -C "$llama_source" apply "$kiwibro_source/scripts/mind/llama_b12443_ngram_request_gate.patch"
cmake -S "$llama_source" -B "$llama_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-O0 -g0" \
  -DGGML_NATIVE=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF
ninja -C "$llama_build" -j1 \
  tools/server/CMakeFiles/server-context.dir/server-schema.cpp.o \
  tools/server/CMakeFiles/server-context.dir/server-context.cpp.o
```

The patched request parser was tested directly for default, null, true,
false, malformed, plain, MTP and mixed-method cases. A live check still needs
a separately built release server. Launch it on loopback with the same draft
flags as above and `--metrics`. Send the same exact-copy prompt with the
boolean false, true and omitted on one server. Use temperature zero and
`cache_prompt: false` for each request. Sample
`llamacpp:spec_decode_num_draft_tokens_total` and
`llamacpp:spec_decode_num_accepted_tokens_total` from `/metrics` before and
after each completion. Both deltas should be zero for false and positive for
the enabled copy probe, with identical exact output. Measure short tool and
vision controls separately against the stock no-draft server. KiwiBro's
runtime selection stays on its existing path.

## One worker per routed expert

The opt-in patch `ggml_cpu_moe_routes.patch` assigns each of six CPU workers one expert projection for one decode token. It uses the original activation quantizer and dot kernel. This avoids the shared route buckets, internal barrier and dynamic row queue. It applies to plain Q4_K fused gate/up and Q5_0 or Q8_0 down weights at the 704-width pilot's shapes, with the existing CPU repacked kernels selected first when available. Other shapes or thread counts use the stock path.

The standalone benchmark maps one actual expert tensor from the 704 pilot at a time. It alternates the stock and opt-in paths in one build and checks every output bit. Each type had four runs covering 140 single-token inputs apiece, including masked slots and repeated IDs. Two-token and four-worker control graphs also agreed bit for bit. The three types together covered 23.65 million single-token output values.

Each timed run used 120 seeded route/input pairs and one hardware thread per physical core on the Ryzen 5 7530U. Repeating the same sequence with the execution order inverted lets each input compare stock and candidate at the same position within its pair. The first arm has a smaller chance of loading the active weights from the paired arm's preceding run. The table uses the median of the two runs with the relevant order for each input, followed by the median across inputs. Negative paired change indicates a slowdown.

| Weight | First position stock → prototype | Median paired gain | Second position stock → prototype | Median paired gain |
| --- | ---: | ---: | ---: | ---: |
| Q4_K gate/up | 0.729 → 0.739 ms | −0.24% | 0.357 → 0.363 ms | −0.69% |
| Q5_0 down | 0.503 → 0.522 ms | −3.83% | 0.232 → 0.228 ms | +1.43% |
| Q8_0 down | 0.693 → 0.701 ms | +0.89% | 0.340 → 0.327 ms | +3.29% |

The paired timings show no consistent operator-level gain large enough to justify a full-model trial. KiwiBro uses its stock CPU backend.

### Reproduce the operator benchmark

Use a clean b12443 source tree and the CMake options above. Apply the route patch, build `ggml-cpu-haswell`, and compile the standalone benchmark:

```sh
git -C "$llama_source" apply "$kiwibro_source/scripts/mind/ggml_cpu_moe_routes.patch"
cmake --build "$llama_build" --target ggml-cpu-haswell --parallel 2
cc -O3 -std=gnu11 -Wall -Wextra -Werror -I "$llama_source/ggml/include" \
  "$kiwibro_source/scripts/mind/bench_moe_routes.c" -L "$llama_build/bin" \
  -l:libggml-cpu-haswell.so -lggml-base -o "$bench"
```

Read absolute tensor offsets from the pilot GGUF with the pinned `GGUFReader`. The measured 11,105,979,328-byte pilot used `blk.0.ffn_gate_up_exps.weight` at 821439424, `blk.3.ffn_down_exps.weight` at 1790612448, and `blk.0.ffn_down_exps.weight` at 648672192. Confirm the format, shape, and offsets again if using another export. Run the selected tensor once per command, repeating the order with `--reverse` twice and then the original order:

```sh
OMP_PROC_BIND=true LD_LIBRARY_PATH="$llama_build/bin" taskset -c "$physical_cpus" \
  "$bench" "$gguf" "$offset" "$kind" 120
OMP_PROC_BIND=true LD_LIBRARY_PATH="$llama_build/bin" taskset -c "$physical_cpus" \
  "$bench" "$gguf" "$offset" "$kind" 120 --reverse
```

Each output row gives stock and prototype milliseconds. Use `q4_k`, `q5_0`, or `q8_0` as the kind. On the test host the six physical CPUs were represented by `0,2,4,6,8,10`.

### GPU cache and quantization controls

The upstream GPU expert cache accepts GPU-assigned layers whose complete expert bank uses host buffers. Upstream `CPU_REPACK` buffers lack that host flag. The cache requires room for six data slots and a guard slot per layout group. Positive CPU miss sharing adds six zero slots per group and has additional limits for batching, live LoRA and rotated expert weights. Slots are shared among compatible layers, and large batches bypass the cache. Inspect the reported cache hits, misses and uploaded MiB at shutdown or record route IDs through `LLAMA_MOE_CACHE_TRACE`. [Pinned MoE cache implementation](https://github.com/ggml-org/llama.cpp/blob/301db633702f1d2c107d4feaa4268f07c0b508c5/src/llama-moe-cache.cpp#L399)

A width-preserving quantization control can try Q4_0 or IQ4_NL for the 16 Q5_0 down matrices from the original BF16 weights. It leaves the other 14 down matrices in Q8_0 and reduces the GGUF by about 302.5 MiB. Both formats align to 32 values, so this comparison needs no width change. The impact on output quality, native tool accuracy and throughput remains to be measured.

## CPU cache and thread count at a 24,576-token allocation

The 28-step pilot ran with one slot and the matching projector. Each profile
passed the same tool call, screenshot and two retrieval checks. The uncached
retrieval prompt had 1,372 tokens. Peak RSS is the model server's resident
memory across the four requests. Times exclude server startup.

| Decode / batch threads | KV cache | Total after startup | Peak server RSS |
| --- | --- | ---: | ---: |
| 4 / 6 | F16 | 137.12 s | 17,382 MiB |
| 6 / 6 | Q8_0 | 104.85 s | 16,603 MiB |
| 6 / 6 | F16 | 81.17 s | 17,370 MiB |

The Q8_0 arm used 767 MiB less peak resident memory than the six-thread F16 arm.
The six-thread F16 arm finished fastest. The CPU pressure average at each arm's
start was 12.77%, 30.36%, and 9.32% in table order.

```bash
python3 -m scripts.mind.tune_local --server "$llama_server" --model "$gemma_gguf" \
  --mmproj "$projector" --context 24576 --filler-lines 36 --repeats 1 --threads 6 \
  --max-tokens 64 --profile cpu-f16 --profile cpu-fewer-threads --profile cpu-q8 \
  --output cpu-24k.json
```

## Long chat on the desktop

An isolated app test seeded 18,490 estimated history tokens, then requested an
exact directory and release checkpoint from the first turn. With the 28-step
pilot, the automatic GPU arm lost its Vulkan device during the recovery turn.
The runtime started a CPU server, and the turn failed to recover the details.

In opt-in extractive mode, SQLite recall supplied exact earlier passages and
the app skipped model-generated summaries. Both forced CPU runs returned the
directory and checkpoint and left the workspace intact. Both omitted the
explicit read-only wording required by acceptance.

| CPU history profile | Recovery prompt tokens | Recovery turn |
| --- | ---: | ---: |
| Original extractive window | 2,874 | 178.9 s |
| Shorter recent window | 2,237 | 78.3 s |

Desktop load differed between runs. The prompt token counts show the shorter
window's direct effect. Its tradeoff is greater reliance on exact recall for
information outside the recent messages. The setting stays optional.

The 300-step standard Q4_K_M export passed this scenario with the same shorter
extractive window and six CPU threads. The recovery turn used 2,244 prompt
tokens and took 86.5 seconds. It returned the exact directory, checkpoint and
read-only constraint, and the workspace was unchanged.
[Acceptance commands](../../docs/MIND-FINALIST-ACCEPTANCE.md)

## Smaller model control on the Ryzen laptop

The Qwen3.5-4B control used the pinned Q4_K_M build and matching Q8_0 projector,
3.40 GB together. The same managed b12443 runtime used Vulkan and a single
8,192-token slot. The warmed app ran three rounds with reasoning Off. Linux
CPU pressure `some avg10` was 6.73% at start and 7.75% at finish.
[Exact Qwen GGUF files](https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF/tree/8d43d974efc48399cb58957ad49c8b26bf5a109c)

| App workload | Verified | First text p50 / p90 | Completion p50 / p90 |
| --- | ---: | ---: | ---: |
| Short reply | 3/3 | 6.34 / 9.42 s | 11.29 / 14.39 s |
| Exact file read | 3/3 | 9.54 / 23.06 s | 15.74 / 28.91 s |
| Follow-up HTML edit and phone render | 2/3 | 118.75 / 119.91 s | 118.75 / 119.91 s |

The one unsuccessful render turn timed out at 120 seconds. The exact-read
tool result arrived at 36 ms median before generation. The small sample
describes this run. It does not isolate the model's speed from the harness
and the selected backend.

## Exact fields from text files

For a request that asks for only a named field in a single-line `.txt` record,
the native app can return the matching plain scalar from its authorized read.
The call and result stay in the chat. Other reads continue through model
generation.

Twelve distinct records passed through the standard candidate's isolated app
on the Ryzen laptop. Time to the verified read result was 50.6 ms p50 and
74.4 ms p90. Completion was 51.3 ms p50 and 74.6 ms p90. The server launch
sentinel verified that these requests needed no model process.

## Liquid AI placement control

LFM2.5-VL-3B Q4_K_M and its matching Q8_0 projector occupied 2.26 GB. The
Ryzen laptop ran the same three uncached probes on b12443, with one 8,192-token
slot. CPU used six threads and Vulkan offloaded the model to Renoir. Every
tool, screenshot and exact-copy check passed in both placements.
[Pinned Liquid AI GGUF build](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF/tree/6f730e9a2c454e8af9adc29db58e638e01e5957f)

| Probe | CPU completion | Vulkan completion |
| --- | ---: | ---: |
| Exact search tool arguments | 7.46 s | 17.22 s |
| Read text from a screenshot | 13.61 s | 20.78 s |
| Copy 14 C declarations | 31.54 s | 129.66 s |

CPU pressure `some avg10` at startup was 7.96% and 19.70% respectively. These
single requests establish a placement for the next app trials. The three
probes cover isolated capabilities.

The managed CPU app then passed three warm repetitions. Completion p50 / p90
was 6.13 / 6.26 seconds for short replies, 32 / 36 ms for direct field reads,
and 21.61 / 23.00 seconds for edits with a verified phone render. Each of the
five deeper cases failed. The nested-repository diagnosis ended after 117.7
seconds without reading the shared implementation. The visual dashboard
case ended after 98.0 seconds without a file. The 3D hatch produced a Chrome
screenshot after 131.5 seconds and timed out before the interaction check.
Exact history recall and table recovery also missed their acceptance checks.

## Compact 300-step candidate on deeper app turns

With the repository repeat guard and default Auto tools, the read-only
diagnosis finished in 187.38 seconds. The first successful directory
result arrived at 101 ms. The model opened the shared implementation and
both consumers, but its reply missed the exact workspace-relative path.
The acceptance run stopped at the diagnosis check, before the edit and
test turn. This was one forced-CPU run with six threads, one 8,192-token
slot, and 17.74 GiB peak native RSS. The sampled process used no swap.

In the visual adapter's first isolated app run, the native dashboard
caption was accurate. The pinned server spent 35.59 seconds processing
248 caption prompt tokens and 96.26 seconds generating 85 caption tokens.
The follow-up HTML request spent another 87.63 seconds processing 1,595
prompt tokens and reached 183 generated tokens before the 300-second cap.
That request used adapter scale zero. A later run verified that the image
caption stayed at scale zero while the user's HTML request used scale 0.8.
That attempt also timed out after trying to read the absent destination.
The 3D hatch took 285.94 seconds to produce a Chrome render at scale 0.8
and timed out before independent interaction verification.

## Gemma 4 E4B control

The upstream E4B Q4_K_M trunk and F16 projector together occupy 6.33 GB.
The unmodified control used the managed b12443 engine with one 8,192-token
slot, six CPU threads and reasoning Off. It passed three warm repetitions
per operation. Peak native RSS was 7.72 GiB and sampled native swap was zero.
[Pinned upstream E4B GGUF and projector](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/tree/fbc327a2e009dc5fbb127ae5461459d2d7d2a7a9)

| Operation | First text p50 / p90 | Completion p50 / p90 |
| --- | ---: | ---: |
| Short reply | 11.32 / 15.34 s | 12.68 / 18.75 s |
| Exact field from a local file | 35 / 36 ms | 35 / 36 ms |
| HTML edit and verified phone render | 73.95 / 113.37 s | 73.95 / 113.37 s |

The file read used the checked application path. The rendered cases ran
native generation, performed the edit and returned a real Chrome result.
CPU pressure `some avg10` was 49.26% at start and 23.22% at finish. These
figures describe that app run under desktop load.

With Auto tools on CPU, E4B took 146.79 seconds to open the shared policy
and name its exact workspace-relative path. The repair turn made the
expected single edit. The independent test runner passed all four tests,
with Git HEAD, index and other files preserved. The app scenario still
failed because the model omitted the explicitly requested working
directory and its command was denied. The repair turn took 240.94 seconds
including repeated denied attempts. CPU pressure `some avg10` was 7.70%
at start and 8.33% at finish.

### E4B placement and compact export

Both E4B trunks passed three raw exact CPU probes through b12443 with six
threads, one 8,192-token slot, reasoning Off, and the matching projector.

| Probe | Stock E4B CPU | Compact E4B CPU | Stock E4B Vulkan |
| --- | ---: | ---: | ---: |
| Exact code search tool call | 8.93 s | 8.41 s | 33.63 s |
| Read code from a screenshot | 12.28 s | 16.08 s | 25.33 s |
| Copy 14 C declarations | 70.70 s | 253.32 s | stopped during generation |

CPU pressure `some avg10` started at 13.94% for stock and 15.00% for compact.
It reached 58.12% while compact code copy was running. The Vulkan copy was
stopped after generating over 100 tokens at about 0.65 tokens per second.
These isolated requests establish that both exports can run the three
probes on CPU. The whole-model speed comparison needs matched load and
repeated app turns.

The compact pack then passed all nine app checks in three warm repetitions.
Peak native RSS was 7.43 GiB and sampled native swap was zero. CPU pressure
`some avg10` was 18.31% at start and 16.34% at finish. Desktop load also
varied within and between the stock and compact runs.

| Operation | Compact first text p50 / p90 | Compact completion p50 / p90 |
| --- | ---: | ---: |
| Short reply | 6.76 / 7.00 s | 8.90 / 9.60 s |
| Exact field from a local file | 41 / 45 ms | 42 / 46 ms |
| HTML edit and verified phone render | 42.91 / 104.96 s | 42.91 / 104.96 s |

The 112-step merged pack failed its five deeper app scenarios. Repository
diagnosis read both consumers and stopped before opening their shared
implementation. The history scenario timed out on its opening turn, before
the long history was seeded. The model found and analyzed the recovered
table, and its final answer missed the required result. The dashboard
produced a real Chrome render at 174.65 seconds and timed out. Its saved
HTML contained the queue name and build code. An independent 360-pixel
test also found that the page missed the exact initial status. The 3D
hatch timed out without a saved file.

### Compaction-only E4B

With the same CPU placement, context, six threads and reasoning Off, the
compaction-only trunk passed two of five deeper app scenarios. Extractive
history was enabled and repository tools used Auto. Peak native RSS was
7.47 GiB. Sampled native swap was zero.

| Scenario | App evidence | Turn time |
| --- | --- | ---: |
| Nested repository | Read both consumers and stopped before opening the shared implementation | 215.18 s |
| Long history | Recovered exact directory and checkpoint with the workspace unchanged | 183.59 s after the opening turn |
| Stale table path | Corrected the failed tool call, read the current TSV and returned the expected result | 271.14 s |
| Dashboard from screenshot | Timed out before saving the page or producing a render | 303.03 s |
| Interactive 3D hatch | Timed out before saving the page | 300.04 s |

The long-history turn used 2,233 prompt tokens after about 18,490 estimated
tokens of earlier history. The native server spent 69.77 seconds on prompt
evaluation and generated 94 tokens in 113.64 seconds. In the dashboard run
the separate image description took 212.62 seconds before the first round
with tools. The repository run started before automatic reads of verified
shared Python imports were added. These results keep the pack experimental.
[Ryzen acceptance report](https://huggingface.co/yunusemrejr/kiwimind/blob/7f744fc9a57a60cf133a82ababa783ae6927535a/variants/e4b-activation-compacted/evals/ryzen-deep-app-cpu.json)

The repository rerun with shared-source reads opened the expected file and
reported its exact path in 72.24 seconds. The follow-up changed only that
file and all four independent fixture tests passed. Git HEAD and the index
were preserved. The app acceptance still failed because the model appended
the folder to its shell command. All three attempts were stopped before
execution. The `run_command` schema now shows separate command and working
directory fields.
[Repository rerun](https://huggingface.co/yunusemrejr/kiwimind/blob/b9c5febd84c5c8af7b4493aebbfe34ef08a7bd31/variants/e4b-activation-compacted/evals/ryzen-repo-shared-source.json)

An isolated direct-image probe sent the screenshot with the model's tool
catalog and skipped the initial caption. It produced a 360-pixel render
with the exact queue, build code and initial status in 275.85 seconds.
The independent interaction check found that the generated page omitted
the requested `state` element ID. This probe used a temporary attachment
path through the app and did not enable a new default.
[Direct-image probe](https://huggingface.co/yunusemrejr/kiwimind/blob/f4e91b8c33fc2c6e1917dcc6c8b5d428dae0e664/variants/e4b-activation-compacted/evals/ryzen-direct-image-probe.json)

The pinned `llama-bench` used one warmed model, 256 tokens of context
before decoding and 12 measured output tokens per arm. In run order, six
workers reached 3.05 tokens/s, four reached 4.35 and 4.45 tokens/s, and six
reached 4.04 tokens/s. CPU pressure `some avg10` varied from 7.2% to 19.2%.
This short synthetic run exercises decode. Application latency and quality
need their own comparison before changing the managed thread count.
[Decode benchmark](https://huggingface.co/yunusemrejr/kiwimind/blob/f4e91b8c33fc2c6e1917dcc6c8b5d428dae0e664/variants/e4b-activation-compacted/evals/ryzen-decode-threads.json)
