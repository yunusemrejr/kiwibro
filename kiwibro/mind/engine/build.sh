#!/usr/bin/env bash
# Compile the Kiwi Mind engine to WebAssembly. Needs clang with the wasm32 target and wasm-ld.
# The three results in ../wasm are committed, so nobody needs a compiler to run Kiwibro.
#
#   build.sh           the three WebAssembly builds
#   build.sh native    a native shared library next to this script's output dir, for sanitizer runs
#                      (BUILD_DIR=... to choose where; default: ../wasm/native, not committed)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/kiwimind.c"

if [[ "${1:-}" == "native" ]]; then
  OUT="${BUILD_DIR:-$HERE/../wasm/native}"
  mkdir -p "$OUT"
  gcc -O1 -g -fPIC -shared -Wall -Wextra -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer \
    -o "$OUT/libkiwimind.so" "$SRC" -lm
  echo "built $OUT/libkiwimind.so"
  exit 0
fi

OUT="$HERE/../wasm"
mkdir -p "$OUT"
BASE=(clang --target=wasm32 -O3 -Wall -Wextra -Werror -msimd128 -mbulk-memory -mnontrapping-fptoint -nostdlib
      -fvisibility=hidden -Wl,--no-entry -Wl,--strip-all -Wl,-z,stack-size=1048576
      -Wl,--initial-memory=4194304 -Wl,--max-memory=4294967296)
THREADS=(-matomics -DKM_THREADS -Wl,--shared-memory -Wl,--import-memory -Wl,--export=__stack_pointer)

"${BASE[@]}" "$SRC" -o "$OUT/kiwimind-simd.wasm"
"${BASE[@]}" "${THREADS[@]}" "$SRC" -o "$OUT/kiwimind-simd-mt.wasm"
"${BASE[@]}" -mrelaxed-simd "${THREADS[@]}" "$SRC" -o "$OUT/kiwimind-relaxed-mt.wasm"

for f in kiwimind-simd kiwimind-simd-mt kiwimind-relaxed-mt; do
  echo "built $OUT/$f.wasm ($(stat -c %s "$OUT/$f.wasm") bytes)"
done
