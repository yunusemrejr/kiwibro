#!/usr/bin/env bash
# Find a Python interpreter that has Kiwibro's dependencies, creating a local
# virtualenv on demand (works on PEP 668 "externally managed" Ubuntu systems).
# `python.sh --which` prints the interpreter instead of running it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pick() {
  if python3 -c 'import flask, requests, bs4' >/dev/null 2>&1; then
    command -v python3
    return
  fi
  local venv="$ROOT/.venv"
  if [[ ! -x "$venv/bin/python" ]]; then
    echo "Kiwibro: creating local virtualenv and installing dependencies…" >&2
    python3 -m venv "$venv" >&2
  fi
  if ! "$venv/bin/python" -c 'import flask, requests, bs4' >/dev/null 2>&1; then
    "$venv/bin/pip" install --disable-pip-version-check -q -r "$ROOT/requirements.txt" >&2
  fi
  echo "$venv/bin/python"
}

PY="$(pick)"
if [[ "${1:-}" == "--which" ]]; then
  echo "$PY"
  exit 0
fi
exec "$PY" "$@"
