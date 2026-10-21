#!/usr/bin/env bash
# Desktop entry point: make sure Kiwibro's server is running, then open a
# native-feeling Chrome app window (browser GUI, no Electron) or the default browser.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${KIWIBRO_PORT:-8777}"
HOST="${KIWIBRO_HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}"
DATA_DIR="${KIWIBRO_DATA_DIR:-$HOME/.local/share/kiwibro}"
mkdir -p "$DATA_DIR"

healthy() { curl -sf -m 1 "$URL/health" >/dev/null 2>&1; }

if ! healthy; then
  cd "$ROOT"
  nohup "$ROOT/scripts/python.sh" app.py >> "$DATA_DIR/server.log" 2>&1 &
  for _ in $(seq 1 120); do
    healthy && break
    sleep 0.25
  done
fi

if ! healthy; then
  if command -v notify-send >/dev/null 2>&1; then
    notify-send -i "$ROOT/static/img/kiwibro-128.png" "Kiwibro" "Server did not start. See $DATA_DIR/server.log"
  fi
  exit 1
fi

PROFILE="${KIWIBRO_CHROME_PROFILE:-$DATA_DIR/chrome-profile}"
CHROME_FLAGS=(--app="$URL" --class=Kiwibro --name=Kiwibro --user-data-dir="$PROFILE"
  --window-size=1280,860 --no-first-run --no-default-browser-check)

for bin in google-chrome google-chrome-stable chromium chromium-browser; do
  if command -v "$bin" >/dev/null 2>&1; then
    exec "$bin" "${CHROME_FLAGS[@]}" >/dev/null 2>&1
  fi
done

exec xdg-open "$URL"
