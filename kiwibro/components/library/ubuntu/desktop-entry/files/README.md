# {{name}} launcher

A `.desktop` file, an icon and an install script, for a program that already runs from a terminal.

## Install it

Set `Exec` to the real program (`{{command}}` here) and put the program's mark in `{{name}}.svg`, then:

    ./install.sh

It copies both files into `~/.local/share/applications` and `~/.local/share/icons` and prints the command that undoes it. No sudo.

## Check it works

Open the app grid and type `{{name}}`: the icon shows and starts the program. `desktop-file-validate {{name}}.desktop` prints nothing when the file is well formed.
