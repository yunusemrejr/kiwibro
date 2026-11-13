# {{name}}.py

{{purpose}}

## Use it

    python3 {{name}}.py --help
    python3 {{name}}.py -v -o counts.json notes.txt todo.md

Needs Python 3.12 and nothing outside the standard library.

## Check it works

`python3 -m py_compile {{name}}.py` prints nothing. `python3 {{name}}.py notes.txt` prints one line per file plus a total; with no file it exits 2 and shows the usage text. `-o counts.json` also writes the counts as JSON.
