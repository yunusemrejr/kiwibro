#!/usr/bin/env bash
# Build {{name}}_{{version}}_all.deb from payload/ with dpkg-deb. No sudo here.
set -euo pipefail

readonly name="{{name}}"
readonly version="{{version}}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly stage="$here/build/$name-$version"
readonly dist="$here/dist"

if [[ ! -d $here/payload ]]; then
  printf 'error: %s/payload is missing\n' "$here" >&2
  exit 1
fi

# KIWIBRO: debian/control holds the Maintainer, Section and Description users see; Installed-Size is filled in here
install -d "$stage/DEBIAN"
install -Dm644 "$here/debian/control" "$stage/DEBIAN/control"
cp -a -- "$here/payload/." "$stage/"

size="$(du -sk --exclude=DEBIAN "$stage" | cut -f1)"
sed -i "s|^Installed-Size:.*|Installed-Size: $size|" "$stage/DEBIAN/control"

install -d "$dist"
dpkg-deb --root-owner-group --build "$stage" "$dist/${name}_${version}_all.deb"

printf 'Built %s\n' "$dist/${name}_${version}_all.deb"
printf 'Install it with: sudo apt install %q\n' "$dist/${name}_${version}_all.deb"
