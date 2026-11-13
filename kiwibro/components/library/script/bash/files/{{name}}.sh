#!/usr/bin/env bash
# {{purpose}}
set -euo pipefail

readonly PROG="{{name}}"
readonly VERSION="0.1.0"

verbose=0
output=""
wrote=""
files=()

usage() {
  cat <<EOF
$PROG $VERSION

{{purpose}}

Usage: $PROG [options] FILE...

Options:
  -o, --output DIR  write the summary into DIR
  -v, --verbose     log every file as it is read
  -h, --help        show this text and exit
EOF
}

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

cleanup() {
  local status=$?
  # Nothing is deleted here: the person running the script decides that.
  if [[ -n $wrote ]]; then
    log "left behind: $wrote"
    log "remove it with: rm -- $(printf '%q' "$wrote")"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'log "failed at line $LINENO"' ERR

# KIWIBRO: these options are the example's; rename them to what the script takes
while (($# > 0)); do
  case "$1" in
    -o | --output)
      (($# >= 2)) || die "$1 needs a folder"
      output="$2"
      shift 2
      ;;
    --output=*) output="${1#*=}"; shift ;;
    -v | --verbose) verbose=1; shift ;;
    -h | --help) usage; exit 0 ;;
    --) shift; break ;;
    -*) die "unknown option: $1 (try --help)" ;;
    *) break ;;
  esac
done
files=("$@")
((${#files[@]} > 0)) || die "no files given (try --help)"
[[ -d ${output:-.} ]] || die "not a folder: ${output:-.}"

total=0
# KIWIBRO: the loop below is the work; read the files, call your tools or API from here
for file in "${files[@]}"; do
  [[ -f $file ]] || die "not a file: $file"
  lines=$(wc -l <"$file")
  total=$((total + lines))
  if ((verbose)); then
    log "$file: $lines lines"
  fi
done

printf '%d files, %d lines\n' "${#files[@]}" "$total"

if [[ -n $output ]]; then
  wrote="$output/$PROG-summary.txt"
  printf '%s: %d files, %d lines\n' "$(date -Iseconds)" "${#files[@]}" "$total" >>"$wrote"
  log "summary appended to $wrote"
fi
