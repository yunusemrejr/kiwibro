#!/usr/bin/env bash
# Rename files from a pattern. Dry run unless --apply is given; never overwrites a file.
set -euo pipefail

readonly PROG="batch-rename"
readonly PATTERN="{{pattern}}"
apply=0
start=1
width=2

usage() {
  cat <<EOF
$PROG — rename files from a pattern

Usage: $PROG [options] FILE...

Pattern: $PATTERN
  {stem}  original name without the extension
  {ext}   extension with its dot
  {n}     running number, zero padded

Options:
  --apply      rename for real; without it the moves are only printed
  --start N    first number (default 1)
  --width N    digits in the number (default 2)
  -h, --help   show this text and exit
EOF
}

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

# KIWIBRO: --start, --width and the pattern cover most renames; add your own options here
while (($# > 0)); do
  case "$1" in
    --apply) apply=1; shift ;;
    --start | --width)
      (($# >= 2)) || die "$1 needs a number"
      [[ $2 =~ ^[0-9]+$ ]] || die "$1 needs a number, not $2"
      if [[ $1 == --start ]]; then start="$2"; else width="$2"; fi
      shift 2
      ;;
    -h | --help) usage; exit 0 ;;
    --) shift; break ;;
    -*) die "unknown option: $1 (try --help)" ;;
    *) break ;;
  esac
done
(($# > 0)) || die "no files given (try --help)"

n="$start"
for file in "$@"; do
  [[ -e $file ]] || die "no such file: $file"
  dir="$(dirname -- "$file")"
  base="$(basename -- "$file")"
  stem="$base"
  ext=""
  if [[ $base == *.* && $base != .* ]]; then
    ext=".${base##*.}"
    stem="${base%.*}"
  fi

  # KIWIBRO: the pattern decides the new name; add placeholders here if the user needs them
  printf -v number '%0*d' "$width" "$n"
  new="$PATTERN"
  new="${new//\{stem\}/$stem}"
  new="${new//\{ext\}/$ext}"
  new="${new//\{n\}/$number}"
  dest="$dir/$new"

  if [[ $new == "$base" ]]; then
    log "unchanged: $file"
    n=$((n + 1))
    continue
  fi
  if [[ -e $dest ]]; then
    die "would overwrite $dest; nothing more was renamed"
  fi

  if ((apply)); then
    mv -- "$file" "$dest"
    log "renamed $file -> $dest"
  else
    printf 'mv -- %q %q\n' "$file" "$dest"
  fi
  n=$((n + 1))
done

if ((apply == 0)); then
  printf 'Dry run: nothing was renamed. Add --apply to do it.\n' >&2
fi
