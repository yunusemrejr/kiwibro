#!/usr/bin/env bash
# Desktop entry point: make sure exactly one Kiwibro server runs the current code,
# then open a native-feeling Chrome app window (browser GUI, no Electron) or the
# default browser.
#
#   kiwibro.sh            start the server (or move a stale one onto new code), open the window
#   kiwibro.sh --restart  restart the server, open the window
#   kiwibro.sh --quit     stop the server and close the app window
#   kiwibro.sh --server   start / refresh the server only (no window)
#
# The server runs in its own systemd user scope, app-kiwibro.scope, as a process named
# "kiwibro", so system monitors (Resources, System Monitor, top) list it as Kiwibro with
# its icon. Ending it there, or `systemctl --user stop app-kiwibro.scope`, stops
# everything it started. A running server also updates itself: once the code on disk
# changes and no reply is running, it restarts onto the new code and open windows
# reload (kiwibro/selfupdate.py).
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PORT="${KIWIBRO_PORT:-8777}"
HOST="${KIWIBRO_HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}"
DATA_DIR="${KIWIBRO_DATA_DIR:-$HOME/.local/share/kiwibro}"
PROFILE="${KIWIBRO_CHROME_PROFILE:-$DATA_DIR/chrome-profile}"
LOG="$DATA_DIR/server.log"
# Monitors read the app id from the scope name: app-<id>.scope -> "kiwibro" (kiwibro.desktop).
# A second instance on another port gets its own scope.
if [ "$PORT" = "8777" ]; then UNIT="app-kiwibro"; else UNIT="app-kiwibro${PORT}"; fi
ICON="$ROOT/static/img/kiwibro-128.png"
mkdir -p "$DATA_DIR"

healthy() { curl -sf -m 1 "$URL/health" >/dev/null 2>&1; }

notify() {
  command -v notify-send >/dev/null 2>&1 && notify-send -i "$ICON" "Kiwibro" "$1" || true
  echo "Kiwibro: $1" >&2
}

# Print one field of a JSON object on stdin ("" when missing; booleans as true/false).
json_field() {
  python3 -I -c 'import json, sys
try:
    v = json.load(sys.stdin).get(sys.argv[1])
except Exception:
    v = None
print("" if v is None else str(v).lower() if isinstance(v, bool) else v)' "$1"
}

running_info() { curl -sf -m 2 "$URL/api/version" 2>/dev/null || echo '{}'; }

source_version() { python3 -I "$ROOT/kiwibro/selfupdate.py" fingerprint "$ROOT" 2>/dev/null || true; }

have_systemd() {
  command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1
}

wait_until() { # wait_until <tenths> <command...>
  local n="$1"; shift
  for _ in $(seq 1 "$n"); do
    "$@" && return 0
    sleep 0.1
  done
  return 1
}

not_healthy() { ! healthy; }
gone() { [ ! -e "/proc/$1" ]; }

# The interpreter, reached through a link named "kiwibro" so the process is called that
# (comm and argv[0]). A venv interpreter has to stay next to its pyvenv.cfg.
named_python() {
  local py dir link
  py="$("$ROOT/scripts/python.sh" --which)" || return 1
  dir="$(dirname "$py")"
  if [ -f "$dir/../pyvenv.cfg" ]; then link="$dir/kiwibro"; else link="$DATA_DIR/bin/kiwibro"; fi
  mkdir -p "$(dirname "$link")"
  if ln -sfn "$py" "$link" 2>/dev/null && "$link" -c 'import flask' >/dev/null 2>&1; then
    echo "$link"
  else
    echo "$py"
  fi
}

stop_server() {
  if have_systemd && systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null; then
    systemctl --user stop "$UNIT.scope" >/dev/null 2>&1 || true
  fi
  # A server started some other way (by hand, or by an older launcher): the pid it reports.
  local pid
  pid="$(running_info | json_field pid)"
  if [ -n "$pid" ] && tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline" | grep -q 'app\.py'; then
    kill -TERM "$pid" 2>/dev/null || true
    wait_until 50 gone "$pid" || kill -KILL "$pid" 2>/dev/null || true
  fi
  # Oldest servers report no pid: whatever listens on the port (fuser matches the local port).
  if healthy && command -v fuser >/dev/null 2>&1; then
    fuser -k -n tcp "$PORT" >/dev/null 2>&1 || true
  fi
  wait_until 50 not_healthy || true
}

start_server() {
  local py
  py="$(named_python)" || return 1
  if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 2097152 ]; then mv -f "$LOG" "$LOG.1"; fi
  cd "$ROOT"
  if have_systemd; then
    systemctl --user reset-failed "$UNIT.scope" >/dev/null 2>&1 || true
    nohup setsid systemd-run --user --scope --unit="$UNIT" --description="Kiwibro" --collect --quiet \
      -p TimeoutStopSec=10s -- "$py" app.py >> "$LOG" 2>&1 < /dev/null &
  else
    nohup setsid "$py" app.py >> "$LOG" 2>&1 < /dev/null &
  fi
  wait_until 400 healthy
}

# Start a server, or replace one that runs older code than what is on disk.
ensure_server() {
  if healthy; then
    local info have want busy
    info="$(running_info)"
    have="$(json_field code_version <<<"$info")"
    want="$(source_version)"
    if [ -z "$have" ]; then
      # A server from before self-updating: restart it unless a reply is running.
      busy="$(curl -sf -m 2 "$URL/api/turns" 2>/dev/null | json_field active_count)"
      if [ -n "$busy" ] && [ "$busy" != "0" ]; then
        notify "A reply is still running; reopen Kiwibro after it finishes to update."
      else
        echo "Kiwibro: restarting the local server to pick up changes…" >&2
        stop_server
      fi
    elif [ -n "$want" ] && [ "$have" != "$want" ]; then
      if [ "$(json_field update_blocked <<<"$info")" = "true" ]; then
        notify "The updated code has an error; still running the previous version. See $DATA_DIR/kiwibro.log"
      elif [ "$(json_field busy <<<"$info")" = "true" ]; then
        notify "Kiwibro will update as soon as the current reply finishes."
      else
        echo "Kiwibro: restarting the local server onto the new code…" >&2
        stop_server
      fi
    fi
  fi
  if ! healthy; then
    stop_server  # clears a hung server still holding the scope or the port
    start_server || true
  fi
  if ! healthy; then
    notify "Server did not start. See $LOG"
    return 1
  fi
}

# The Chrome app window: the browser process of our private profile (not its helpers).
close_window() {
  local pid args
  for pid in $(pgrep -f -- "--user-data-dir=$PROFILE" 2>/dev/null || true); do
    args="$(tr '\0' '\n' 2>/dev/null < "/proc/$pid/cmdline")" || continue
    grep -qxF -- "--user-data-dir=$PROFILE" <<<"$args" || continue
    grep -q -- '^--type=' <<<"$args" && continue
    kill -TERM "$pid" 2>/dev/null || true
  done
}

open_window() {
  local flags=(--app="$URL" --class=Kiwibro --name=Kiwibro --user-data-dir="$PROFILE"
    --window-size=1280,860 --no-first-run --no-default-browser-check)
  for bin in google-chrome google-chrome-stable chromium chromium-browser; do
    if command -v "$bin" >/dev/null 2>&1; then
      exec "$bin" "${flags[@]}" >/dev/null 2>&1
    fi
  done
  exec xdg-open "$URL"
}

case "${1:-}" in
  --quit)
    stop_server
    close_window
    ;;
  --restart)
    stop_server
    ensure_server
    open_window
    ;;
  --server)
    ensure_server
    ;;
  "")
    ensure_server
    open_window
    ;;
  *)
    echo "usage: kiwibro.sh [--restart | --quit | --server]" >&2
    exit 2
    ;;
esac
