#!/usr/bin/env bash
# Put {{name}} in the app grid for the current user. No sudo, writes only under ~/.local/share.
set -euo pipefail

name="{{name}}"
program="{{command}}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
apps="$HOME/.local/share/applications"
icons="$HOME/.local/share/icons/hicolor/scalable/apps"

first="${program%% *}"
if [[ -z $(command -v "$first") && ! -x $first ]]; then
  printf 'Warning: %s was not found. The icon will appear but do nothing until that program exists.\n' "$first" >&2
fi

install -Dm644 "$here/$name.desktop" "$apps/$name.desktop"
install -Dm644 "$here/$name.svg" "$icons/$name.svg"

printf 'Wrote:\n  %s\n  %s\n' "$apps/$name.desktop" "$icons/$name.svg"
printf '\n{{name}} should be in the app grid now. If it is not, log out and back in.\n'
printf 'To undo, run: rm -f -- %q %q\n' "$apps/$name.desktop" "$icons/$name.svg"
