#!/usr/bin/env bash
# Desktop entry point: make sure Kiwibro's server is running the current code,
# then open a native-feeling Chrome app window (browser GUI, no Electron) or the
# default browser.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${KIWIBRO_PORT:-8777}"
HOST="${KIWIBRO_HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}"
DATA_DIR="${KIWIBRO_DATA_DIR:-$HOME/.local/share/kiwibro}"
mkdir -p "$DATA_DIR"

healthy() { curl -sf -m 1 "$URL/health" >/dev/null 2>&1; }

server_started_at() {
  curl -sf -m 1 "$URL/health" 2>/dev/null | python3 -c \
    'import json,sys; print(int(json.load(sys.stdin).get("started_at") or 0))' 2>/dev/null || echo 0
}

latest_source_mtime() {
  find "$ROOT/app.py" "$ROOT/requirements.txt" "$ROOT/kiwibro" "$ROOT/templates" "$ROOT/static" \
    -type f \( -name '*.py' -o -name '*.js' -o -name '*.css' -o -name '*.html' -o -name '*.svg' \
    -o -name '*.png' -o -name '*.ico' -o -name 'requirements.txt' \) \
    -not -path '*/__pycache__/*' -printf '%T@\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d. -f1
}

start_server() {
  cd "$ROOT"
  nohup "$ROOT/scripts/python.sh" app.py >> "$DATA_DIR/server.log" 2>&1 &
  for _ in $(seq 1 160); do
    healthy && return 0
    sleep 0.25
  done
  return 1
}

# If an older server is still running (e.g. the repo was updated after launch),
# restart it so the app always serves the current code.
if healthy; then
  started="$(server_started_at)"
  newest="$(latest_source_mtime || true)"
  stale=0
  if [ -z "$started" ] || [ "$started" -eq 0 ]; then
    stale=1
  elif [ -n "$newest" ] && [ "$newest" -gt "$started" ]; then
    stale=1
  fi
  if [ "$stale" -eq 1 ] && command -v fuser >/dev/null 2>&1; then
    echo "Kiwibro: restarting the local server to pick up changes…" >&2
    fuser -k -n tcp "$PORT" >/dev/null 2>&1 || true
    for _ in $(seq 1 40); do
      healthy || break
      sleep 0.1
    done
  fi
fi

if ! healthy; then
  start_server || true
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
