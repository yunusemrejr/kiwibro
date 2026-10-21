#!/usr/bin/env bash
# Start Kiwibro in the foreground. Use --open to launch the native app window.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "${KIWIBRO_DATA_DIR:-$HOME/.local/share/kiwibro}"

if [[ "${1:-}" == "--open" ]]; then
  exec "$ROOT/scripts/kiwibro.sh"
fi

exec "$ROOT/scripts/python.sh" app.py
