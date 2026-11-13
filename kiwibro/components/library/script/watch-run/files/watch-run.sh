#!/usr/bin/env bash
# Run a command again whenever files in a folder change.
set -euo pipefail

readonly PROG="watch-run"
# KIWIBRO: the command that runs on every change; it is split on spaces
readonly COMMAND='{{command}}'
interval=2
watch="."

usage() {
  cat <<EOF
$PROG — run a command whenever a folder changes

Usage: $PROG [options] [FOLDER]

Command: $COMMAND
Options:
  -i, --interval N  seconds between checks while polling (default 2)
  -h, --help        show this text and exit

Uses inotifywait when it is installed and polls the file times otherwise.
EOF
}

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

while (($# > 0)); do
  case "$1" in
    -i | --interval)
      (($# >= 2)) || die "$1 needs a number"
      [[ $2 =~ ^[0-9]+$ ]] || die "$1 needs a number, not $2"
      interval="$2"
      shift 2
      ;;
    -h | --help) usage; exit 0 ;;
    --) shift; break ;;
    -*) die "unknown option: $1 (try --help)" ;;
    *) break ;;
  esac
done
watch="${1:-.}"
[[ -d $watch ]] || die "not a folder: $watch"

read -r -a argv <<<"$COMMAND"
((${#argv[@]} > 0)) || die "no command set at the top of this file"

run() {
  log "running: $COMMAND"
  local status=0
  "${argv[@]}" || status=$?
  if ((status > 0)); then
    log "command exited with $status; waiting for the next change"
  fi
}

if [[ -n $(command -v inotifywait) ]]; then
  log "watching $watch with inotifywait, Ctrl+C stops it"
  run
  # KIWIBRO: the events watched here are the usual ones; extend the list for your build
  inotifywait -m -r -e close_write,create,moved_to --format '%w%f' "$watch" |
    while IFS= read -r changed; do
      log "changed: $changed"
      run
    done
else
  log "inotifywait is not installed, polling $watch every ${interval}s, Ctrl+C stops it"
  snapshot() { find "$watch" -type f -printf '%T@ %p\n' | sort | md5sum; }
  last="$(snapshot)"
  run
  while sleep "$interval"; do
    current="$(snapshot)"
    if [[ $current != "$last" ]]; then
      last="$current"
      run
    fi
  done
fi
