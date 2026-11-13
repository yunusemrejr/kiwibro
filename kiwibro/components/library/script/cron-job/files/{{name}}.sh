#!/usr/bin/env bash
# Run from cron. `{{name}}.sh --crontab` prints the line to paste into `crontab -e`.
set -euo pipefail

readonly PROG="{{name}}"
readonly SCHEDULE="{{schedule}}"
# KIWIBRO: the log path and the 1 MiB cap suit most jobs; change them if this one prints more
readonly LOG="${LOG:-$HOME/.local/state/$PROG.log}"
readonly LOCK="${LOCK:-$HOME/.local/state/$PROG.lock}"
readonly MAX_BYTES="${MAX_BYTES:-1048576}"

log() { printf '%s %s\n' "$(date -Is)" "$*" >>"$LOG"; }

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
  --crontab)
    printf '%s %s/%s\n' "$SCHEDULE" "$here" "$(basename -- "${BASH_SOURCE[0]}")"
    exit 0
    ;;
  -h | --help)
    printf 'Usage: %s [--crontab]\n' "$PROG"
    printf '  no argument  run the job once and log to %s\n' "$LOG"
    printf '  --crontab    print the crontab line for this file\n'
    exit 0
    ;;
  "") ;;
  *) printf 'unknown argument: %s (try --help)\n' "$1" >&2; exit 2 ;;
esac

mkdir -p "$(dirname -- "$LOG")"

# One run at a time. The lock file is opened read-write: it is created once, never truncated.
exec {lock_fd}<>"$LOCK"
if ! flock -n "$lock_fd"; then
  log "another run is still going, stopping"
  exit 0
fi

# Keep the log small by renaming it; the previous log stays next to it as <log>.1
if [[ -f $LOG && $(stat -c %s "$LOG") -gt $MAX_BYTES ]]; then
  mv -f "$LOG" "$LOG.1"
fi

log "start"

# KIWIBRO: the job itself goes in this function; cron throws output away, so log what matters
main() {
  log "nothing to do yet"
}

main
log "done"
