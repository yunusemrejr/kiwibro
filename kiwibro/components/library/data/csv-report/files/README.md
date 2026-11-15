# {{name}}.py

Reads a CSV, checks the columns you name, totals one numeric column per
group and writes a Markdown report.

## Use it

    python3 {{name}}.py sample.csv --group category --sum quantity --out report.md

`--group` names the column to group by and `--sum` the numeric column to
total. Money and decimals are summed with `decimal.Decimal`, never float.
With no `--out` the report goes to the terminal. If a named column is
missing the script names every column it did find and exits non-zero.

Needs Python 3.12 and nothing outside the standard library.

## Check it works

`python3 -m pytest -q` runs the tests: grouping, the money maths, the
missing-column message and writing the report file.
