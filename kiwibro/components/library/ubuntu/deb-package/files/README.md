# {{name}} {{version}} package

Turns the tree in `payload/` into `dist/{{name}}_{{version}}_all.deb` with `dpkg-deb`. Building needs no sudo; installing needs sudo once. A rebuild reuses `build/`, so remove that folder if a file must disappear from the package.

## Build and install

    ./build.sh
    sudo apt install ./dist/{{name}}_{{version}}_all.deb

## Check it works

    dpkg-deb --info dist/{{name}}_{{version}}_all.deb
    dpkg -L {{name}}
    {{name}}
