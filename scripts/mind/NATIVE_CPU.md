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

Run only one server at a time. Use the same runtime, backend, context, one slot, projector, sampler, token budget, prompts and held-out probes for both GGUFs. Bind the model and runtime identities to the result. Run `scripts.mind.replay_successor` on the same frozen probes at seed 7530 and `scripts.mind.bench_successor` for uncached and cached TTFT, decode rate and peak RAM. Check exact tool calls, visual fixtures, retrieval, and full workspace episodes before choosing a candidate. The table above is a footprint estimate. The 512 candidate has no full-model quality or latency measurements yet.

### Tied projection Vulkan experiment

This matched test changes the placement of the 577.5 MiB tied Q6_K embedding/projection tensor on the existing pilot. It uses no GPU layer or KV offload. Begin with at least 14 GiB host memory available and 2 GiB unused GTT, then monitor both during the run. These are conservative admission values based on observed host pressure and the tensor size. The split runtime's total peak memory needs live verification.

```bash
# Both variables point to verified local artifacts.
args=(--device Vulkan0 -m "$gguf" -ngl 0 -nkvo 1 -nopo 1
      -b 128 -ub 128 -t 6 -fa on -p 0 -n 24 -d 256 -r 2 -o json)
overrides=(-ot '^token_embd\.weight$=Vulkan0')
LD_LIBRARY_PATH="$llama_release" "$llama_release/llama-bench" "${args[@]}" > cpu-head.json
LD_LIBRARY_PATH="$llama_release" "$llama_release/llama-bench" "${args[@]}" "${overrides[@]}" > vulkan-head.json
```

Repeat in reverse order. A short separate probe with `GGML_SCHED_DEBUG=2` and `llama-bench -v` can verify that the logits multiply follows the tied weight onto Vulkan. Run the timed trials with scheduler debugging disabled. Compare decoded outputs and native tool behavior as well as timing, since the CPU and Vulkan implementations can differ in floating-point accumulation. This specific placement has yet to be benchmarked.
