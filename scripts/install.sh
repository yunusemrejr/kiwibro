#!/usr/bin/env bash
# Install Kiwibro into the Ubuntu/GNOME application grid (no root needed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="$HOME/.local/share/applications"
ICON_ROOT="$HOME/.local/share/icons/hicolor"

mkdir -p "$APPS_DIR"

sed "s|@ROOT@|$ROOT|g" "$ROOT/scripts/kiwibro.desktop.in" > "$APPS_DIR/kiwibro.desktop"
chmod +x "$APPS_DIR/kiwibro.desktop"

install -Dm644 "$ROOT/static/img/kiwi.svg" "$ICON_ROOT/scalable/apps/kiwibro.svg" 2>/dev/null || true
for size in 64 128 256 512; do
  src="$ROOT/static/img/kiwibro-${size}.png"
  if [[ -f "$src" ]]; then
    install -Dm644 "$src" "$ICON_ROOT/${size}x${size}/apps/kiwibro.png"
  fi
done

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -f -t "$ICON_ROOT" >/dev/null 2>&1 || true

echo "Kiwibro installed. Open it from your app grid (search “Kiwibro”)."
