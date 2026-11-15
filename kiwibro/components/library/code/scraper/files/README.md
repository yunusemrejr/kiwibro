# {{name}}

Read one page and save its links or its first table as CSV, standard library only.

    python3 {{name}}.py https://example.com --what links --out links.csv
    python3 {{name}}.py https://example.com --what tables

Links are resolved against the page URL, so they come out absolute. A table always
comes out at one width: short rows are padded with empty cells. `--out -` prints the
CSV instead of writing a file, and `--timeout` sets how many seconds the page has to
answer.

Every request has a timeout, and a failure names both sides: "expected status 200,
got 404" or "expected HTML, got application/json".

Only the first table on the page is read, and the parser keeps nothing outside
`<a>`, `<td>` and `<th>`.

Test with:

    python3 -m pytest -q
