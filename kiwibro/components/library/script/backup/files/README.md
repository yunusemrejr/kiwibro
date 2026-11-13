# backup.sh

Dated tar backup of one folder with a verification step. It never removes anything: archives older than 30 days are listed, with the command that removes each one.

## Use it

    ./backup.sh
    ./backup.sh ~/work /media/usb/backups

The first line backs up `{{source}}` into `{{target}}`; both can be given as arguments. `KEEP_DAYS=90 ./backup.sh` changes what counts as old. The archive is `TARGET/<folder>-<date>-<time>.tar.gz`, and the target folder has to exist already.

## Check it works

Run it on a folder with a few files: it prints the archive path, the entry count and the size, and `tar -tzf <archive>` lists the same files.
