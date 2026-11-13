#!/usr/bin/env bash
# Dated tar backup of one folder, checked after writing. Nothing is deleted here.
set -euo pipefail

readonly PROG="backup"
readonly KEEP_DAYS="${KEEP_DAYS:-30}"
# KIWIBRO: the folder to back up and the folder that keeps the archives
SOURCE="${1:-{{source}}}"
TARGET="${2:-{{target}}}"

usage() {
  cat <<EOF
$PROG — archive a folder and verify the archive

Usage: $PROG [SOURCE] [TARGET]

  SOURCE  folder to back up (default: {{source}})
  TARGET  folder that holds the archives, must exist (default: {{target}})

Writes TARGET/<folder>-<date>-<time>.tar.gz and checks that the archive lists
the files. Archives older than $KEEP_DAYS days are listed, never removed.
EOF
}

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

case "${1:-}" in
  -h | --help) usage; exit 0 ;;
esac

SOURCE="${SOURCE%/}"
[[ -d $SOURCE ]] || die "no such folder: $SOURCE"
[[ -d $TARGET ]] || die "no such folder: $TARGET (create it with: mkdir -p -- $TARGET)"

stamp="$(date +%Y-%m-%d-%H%M%S)"
base="$(basename -- "$SOURCE")"
archive="$TARGET/$base-$stamp.tar.gz"

log "writing $archive"
tar -czf "$archive" -C "$(dirname -- "$SOURCE")" "$base"

entries="$(tar -tzf "$archive" | wc -l)"
files="$(find "$SOURCE" -type f | wc -l)"
if ((entries <= files)); then
  die "verification failed: the archive lists $entries entries, $base has $files files"
fi
log "verified: $entries entries in the archive, $files files in $base"
log "size: $(du -h "$archive" | cut -f1)"

# KIWIBRO: KEEP_DAYS decides what counts as an old backup; those are listed, never removed
old="$(find "$TARGET" -maxdepth 1 -type f -name "$base-*.tar.gz" -mtime +"$KEEP_DAYS" \
  ! -name "$(basename -- "$archive")" | sort)"
if [[ -z $old ]]; then
  log "no $base archives older than $KEEP_DAYS days"
else
  log "older than $KEEP_DAYS days, left in place:"
  while IFS= read -r file; do
    printf '  %s\n  remove with: rm -- %q\n' "$file" "$file" >&2
  done <<<"$old"
fi

printf '%s\n' "$archive"
