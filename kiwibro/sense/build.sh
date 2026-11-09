#!/usr/bin/env bash
# Compile the Kiwi Sense core to WebAssembly. Needs clang with the wasm32 target and wasm-ld.
# The result (static/sense/sense.wasm) is committed, so nobody needs a compiler to run Kiwibro.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../../static/sense"
mkdir -p "$OUT"

clang --target=wasm32 -O2 -nostdlib -ffreestanding -fno-builtin \
  -fvisibility=hidden -Wall -Wextra -Werror \
  -Wl,--no-entry -Wl,--strip-all -Wl,-z,stack-size=65536 \
  -Wl,--initial-memory=131072 -Wl,--export=__heap_base \
  -o "$OUT/sense.wasm" "$HERE/core.c"

echo "built $OUT/sense.wasm ($(stat -c %s "$OUT/sense.wasm") bytes)"
