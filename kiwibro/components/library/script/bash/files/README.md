# {{name}}.sh

{{purpose}}

## Use it

    ./{{name}}.sh --help
    ./{{name}}.sh -v notes.txt

`-h` prints the usage text, `-v` logs every file, `-o DIR` appends a timestamped summary line to `DIR/{{name}}-summary.txt`. An unknown option or a missing file stops the script with a non-zero exit code.

## Check it works

`shellcheck {{name}}.sh` prints nothing. `./{{name}}.sh` with no files exits 1; with two files it prints the line total. The cleanup trap only names what it wrote and the command that removes it.
