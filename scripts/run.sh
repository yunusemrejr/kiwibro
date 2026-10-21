#!/usr/bin/env bash
# Start Kiwibro in the foreground. Use --open to launch the native app window.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! python3 -c 'import flask, requests, bs4' >/dev/null 2>&1; then
  echo "Kiwibro: installing Python dependencies…"
  python3 -m pip install --user -r requirements.txt >/dev/null 2>&1 \
    || python3 -m pip install -r requirements.txt
fi

mkdir -p "${KIWIBRO_DATA_DIR:-$HOME/.local/share/kiwibro}"

if [[ "${1:-}" == "--open" ]]; then
  exec "$ROOT/scripts/kiwibro.sh"
fi

exec python3 app.py
