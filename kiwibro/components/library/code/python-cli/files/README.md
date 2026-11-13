# {{name}}

A command-line tool with two subcommands:

    {{name}} stats FILE...     lines, words and characters per file, plus a total
    {{name}} longest FILE...   the longest line of each file

Install with `python3 -m pip install -e .`, then:

    {{name}} stats README.md

`-v` turns on debug logging (each file as it is read). Exit codes: 0 success,
1 a file could not be read, 2 bad arguments. Test with:

    python3 -m pytest -q
