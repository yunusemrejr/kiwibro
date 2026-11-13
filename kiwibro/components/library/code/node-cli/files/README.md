# {{name}}

Print the largest files under a directory, biggest first. No dependencies:
argument parsing uses `node:util`, the tests use `node:test`.

## Run

    node bin/{{name}}.js --limit 5 ~/Downloads

`npm link` once puts `{{name}}` on your PATH. `--all` includes hidden files.
The tool only reads: files are never written, moved or removed.

## Test

    node --test
