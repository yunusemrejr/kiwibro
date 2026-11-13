#!/usr/bin/env bash
# Install {{app_name}} for the current user. No sudo, nothing outside ~/.local/share.
set -euo pipefail

app_id="{{app_id}}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data="$HOME/.local/share/$app_id"
apps="$HOME/.local/share/applications"
icons="$HOME/.local/share/icons/hicolor/scalable/apps"

install -Dm755 "$here/main.py" "$data/main.py"
install -Dm644 "$here/$app_id.svg" "$icons/$app_id.svg"
install -Dm644 "$here/$app_id.desktop" "$apps/$app_id.desktop"

# KIWIBRO: the launcher must start the installed copy, so Exec gets the full path
sed -i "s|^Exec=.*|Exec=python3 $data/main.py|" "$apps/$app_id.desktop"

printf 'Wrote:\n  %s\n  %s\n  %s\n' \
  "$data/main.py" "$icons/$app_id.svg" "$apps/$app_id.desktop"
printf '\n{{app_name}} should be in the app grid now.\n'
printf 'To undo, run: rm -rf -- %q %q %q\n' \
  "$data" "$icons/$app_id.svg" "$apps/$app_id.desktop"
