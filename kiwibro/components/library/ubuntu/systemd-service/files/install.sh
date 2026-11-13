#!/usr/bin/env bash
# Install the {{name}} user units. No sudo, writes only under ~/.config/systemd/user.
set -euo pipefail

name="{{name}}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
units="$HOME/.config/systemd/user"

install -Dm644 "$here/$name.service" "$units/$name.service"
install -Dm644 "$here/$name.timer" "$units/$name.timer"

printf 'Wrote:\n  %s\n  %s\n' "$units/$name.service" "$units/$name.timer"
printf '\nStart it on a schedule:\n'
printf '  systemctl --user daemon-reload\n'
printf '  systemctl --user enable --now %s.timer\n' "$name"
printf '\nCheck it:\n'
printf '  systemctl --user list-timers %s.timer\n' "$name"
printf '  journalctl --user -u %s.service -n 30\n' "$name"
printf '\nTo undo:\n'
printf '  systemctl --user disable --now %s.timer\n' "$name"
printf '  rm -- %q %q\n' "$units/$name.timer" "$units/$name.service"
