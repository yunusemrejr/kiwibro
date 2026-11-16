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
