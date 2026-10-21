#!/usr/bin/env bash
# Find a Python interpreter that has Kiwibro's dependencies, creating a local
# virtualenv on demand (works on PEP 668 "externally managed" Ubuntu systems).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if python3 -c 'import flask, requests, bs4' >/dev/null 2>&1; then
  exec python3 "$@"
fi

VENV="$ROOT/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Kiwibro: creating local virtualenv and installing dependencies…" >&2
  python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c 'import flask, requests, bs4' >/dev/null 2>&1; then
  "$VENV/bin/pip" install --disable-pip-version-check -q -r "$ROOT/requirements.txt"
fi
exec "$VENV/bin/python" "$@"
