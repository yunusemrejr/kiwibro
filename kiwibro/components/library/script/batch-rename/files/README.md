# batch-rename.sh

Renames files from a pattern. It prints the moves first and only renames with `--apply`, and it stops instead of overwriting an existing file.

## Use it

    ./batch-rename.sh photos/*.jpg
    ./batch-rename.sh --apply --start 10 --width 3 photos/*.jpg

Pattern: `{{pattern}}` — `{stem}` is the old name, `{ext}` its extension, `{n}` the running number.

## Check it works

Run it on three sample files: it prints three `mv` lines and changes nothing. Run the same line with `--apply` and `ls` shows the new names; a second run prints `unchanged` for each and still renames nothing.
