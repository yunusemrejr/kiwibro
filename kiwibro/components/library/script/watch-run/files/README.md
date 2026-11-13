# watch-run.sh

Runs one command again every time a file in a folder changes. It uses `inotifywait` when that is installed and polls the file times otherwise; neither needs any setup.

## Use it

    ./watch-run.sh .
    ./watch-run.sh -i 5 src

The command is the `COMMAND` line at the top of the script, `{{command}}` here. It is split on spaces, so use a small wrapper script instead of shell quoting.

## Check it works

Start it, then save a file in the folder: the command runs again and the log names the file that changed. A failing command is reported and the loop keeps going.
