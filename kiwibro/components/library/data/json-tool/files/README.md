# {{name}}.py

Reads a JSON list of objects, keeps only the fields you name, renames
fields, keeps only the records that match a field value, and writes the
result as JSON or CSV.

## Use it

    python3 {{name}}.py sample.json --pick name,kind --where kind=tool --format csv --out out.csv

`--pick a,b,c` keeps only those fields. `--rename old=new` renames a field
and may be repeated. `--where field=value` keeps matching records only.
`--format` is `json` (the default) or `csv`; with no `--out` the result
is printed. If a picked field is missing the script names every field it
did find and exits non-zero.

Needs Python 3.12 and nothing outside the standard library.

## Check it works

`python3 -m pytest -q` runs the tests: picking, renaming, filtering, the
JSON and CSV output, and bad input.
