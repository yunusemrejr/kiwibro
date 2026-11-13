# {{name}} as a user service and timer

`{{name}}.service` runs `{{command}}` once as your user; `{{name}}.timer` starts it on `{{schedule}}`. Both are user units, never root.
## Install it

    ./install.sh
    systemctl --user daemon-reload
    systemctl --user enable --now {{name}}.timer

`Type=oneshot` suits a job that finishes; use `Type=simple` if the program keeps running. `Persistent=true` catches up a run missed while the machine was off.

## Check it works

    systemctl --user list-timers {{name}}.timer
    journalctl --user -u {{name}}.service -n 30
