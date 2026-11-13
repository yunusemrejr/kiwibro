# {{name}}.sh

A cron job with one run at a time, a log capped at 1 MiB, and the crontab line it needs.
## Install it

    ./{{name}}.sh --crontab
    crontab -e

That prints the line for `{{schedule}}` with an absolute path; paste it into crontab. Cron has no `~`, so keep the folder free of spaces.

## Check it works

    ./{{name}}.sh && tail ~/.local/state/{{name}}.log

Run it twice at once: the second says another run is going. Past 1 MiB the log becomes `{{name}}.log.1` and a fresh one starts.
