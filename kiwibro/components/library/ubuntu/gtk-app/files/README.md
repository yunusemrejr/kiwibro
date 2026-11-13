# {{app_name}}

GTK 4 / libadwaita window: header bar, Add (Ctrl+N), quit (Ctrl+Q), an empty state, and notes kept in `~/.local/share/{{app_id}}/notes.json`.

## Run it

    sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1
    python3 main.py
    ./install.sh

## Check it works

Add two notes, quit and start again: both are listed. With no notes you get the empty state. `install.sh` writes the program, the icon and the `.desktop` file under `~/.local/share` and prints the command that undoes it.
