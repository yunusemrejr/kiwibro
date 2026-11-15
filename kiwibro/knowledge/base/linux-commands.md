# Linux command reference

## ls, list files and folders
Lists the files and folders in a directory, or in the current folder when no path is given.
`ls -la` shows all files, including hidden ones whose names start with a dot, with permissions, owner and size. `ls -lh /var/log` shows sizes in readable units such as K and M.

## cd, change directory
Changes the folder the shell is working in. With no argument it goes to your home folder.
`cd ~/Documents` goes to Documents in your home folder. `cd ..` moves up one level, and `cd -` returns to the previous folder.

## pwd, print current directory
Prints the full path of the folder you are in now.
`pwd` might print `/home/alice/projects`.

## cp, copy files and folders
Copies a file or folder to a new name or place. An existing destination file is overwritten without a warning, and its old content is lost, unless you use -i.
`cp notes.txt backup.txt` copies a file. `cp -r photos/ photos-copy/` copies a folder with everything in it. `cp -i a.txt b.txt` asks before overwriting.

## mv, move or rename
Moves a file or folder to another place, or renames it when the destination is a new name. An existing destination file is overwritten silently, and its old content is lost, unless you use -i.
`mv draft.txt final.txt` renames a file. `mv *.jpg pictures/` moves all JPG files into the pictures folder.

## rm, remove files
Deletes files permanently. It does not use the trash, so the deleted data cannot be restored from the shell. Without -r it refuses to delete folders.
`rm old.log` deletes one file. `rm -i *.tmp` asks before each deletion. `rm -rf folder/` deletes the folder and everything inside it with no question, so check the path first.

## mkdir, make directory
Creates a new folder. It fails if the folder already exists.
`mkdir reports` creates one folder. `mkdir -p projects/2026/notes` creates every missing folder along the path.

## rmdir, remove empty folder
Deletes a folder only when it is empty. It never deletes files, so it is the safe choice when a folder should contain nothing.
`rmdir empty-folder` deletes it. If it says the directory is not empty, the folder still holds files.

## touch, create empty file or update time
Creates an empty file when the name does not exist. When the file exists, it only changes the file's modification time and leaves the content alone.
`touch notes.txt` creates an empty file. `touch -d "2 days ago" old.txt` sets an earlier modification time.

## find, search for files
Searches a folder tree for files that match a name, type, size or age. It walks through all subfolders.
`find . -name "*.log"` lists log files under the current folder. `find ~ -type f -size +100M` lists files bigger than 100 MB. `find /tmp -type f -mtime +7` lists files not changed in more than seven days.

## grep, search text
Prints the lines in files that match a pattern. Use it to find words or messages inside files.
`grep "error" app.log` shows matching lines. `grep -in "timeout" *.txt` ignores case and shows line numbers. `grep -r "TODO" src/` searches every file in a folder tree.

## sed, edit text streams
Reads text line by line, changes it with a pattern and prints the result. With -i it changes the file itself, and the original text is lost.
`sed 's/cat/dog/g' story.txt` prints the file with every "cat" replaced by "dog". `sed -i 's/http:/https:/g' links.txt` rewrites the file in place. `sed -n '10,20p' file.txt` prints lines 10 to 20.

## awk, process columns of text
Splits each line into fields and runs actions on them. It suits reports, CSV-like data and log columns.
`awk '{print $1}' access.log` prints the first column of each line. `awk -F: '{print $1}' /etc/passwd` prints usernames, using the colon as the separator. `awk '$3 > 100' data.txt` prints lines whose third field is larger than 100.

## cat, print or join files
Prints the contents of one or more files to the screen, or joins them into one output. It does not pause on long files.
`cat notes.txt` shows the file. `cat -n script.sh` numbers the lines. `cat a.txt b.txt > both.txt` joins two files into a new file, overwriting it if it exists.

## less, read long text
Shows a file one screen at a time so you can scroll through it. It does not change the file.
`less /var/log/syslog` opens a log. Press space to go down, b to go up, `/word` to search, and q to quit.

## more, view text page by page
Shows a file one screen at a time and only moves forward. Less does everything more does and more.
`more README.txt` shows the file. Press space for the next page and q to quit.

## head, show first lines
Prints the first lines of a file, ten by default. Useful for peeking into a large file.
`head data.csv` shows the first ten lines. `head -n 3 data.csv` shows three lines.

## tail, show last lines or follow a file
Prints the last lines of a file. With -f it keeps showing new lines as they are added, which suits log files.
`tail -n 50 app.log` shows the last 50 lines. `tail -f /var/log/syslog` follows the log until you press Ctrl+C.

## chmod, change file permissions
Changes who may read, write or run a file. The values are r (read, 4), w (write, 2) and x (run, 1), set separately for owner, group and others.
`chmod +x script.sh` makes a script runnable. `chmod 644 notes.txt` lets the owner read and write and everyone else only read. `chmod -R 755 site/` overwrites the permissions of every file and folder inside site.

## chown, change owner
Changes the user and group that own a file or folder. It usually needs sudo.
`sudo chown alice notes.txt` sets the owner. `sudo chown alice:staff project/` sets owner and group. `sudo chown -R alice project/` changes every item inside, replacing their previous owners.

## ps, list running processes
Lists running programs with their process IDs (PID).
`ps aux` lists all processes of all users with CPU and memory use. `ps -ef | grep nginx` finds a program by name.

## top, live process monitor
Shows running processes that refresh every few seconds, sorted by CPU use by default. Press q to quit.
`top` opens the view. Press M to sort by memory and k to send a signal to a process by its PID.

## kill, stop a process
Sends a signal to a process by its PID. The default signal asks the program to stop, so it can save data and clean up. kill -9 forces an immediate stop, and any unsaved work in that program is lost.
`kill 4821` asks process 4821 to stop. `kill -9 4821` forces it to stop, and should be a last resort.

## pkill, kill processes by name
Sends a signal to every process whose name or command line matches a pattern, without needing the PID.
`pkill firefox` asks all Firefox processes to close. `pkill -f "python3 server.py"` matches the full command line.

## df, free disk space
Shows the total, used and free space on each mounted disk or partition.
`df -h` shows sizes in readable units. `df -h /` shows only the disk that holds the root folder.

## du, disk usage of folders
Shows how much space files and folders take up. Useful for finding what fills a disk.
`du -sh ~/Downloads` prints one total for that folder. `du -h --max-depth=1 /home` lists the size of each folder one level down.

## free, memory use
Shows total, used and free RAM and swap memory. The available column is the amount new programs can use without swapping.
`free -h` prints the values in readable units.

## tar, archive files
Packs many files into one archive file and can compress it. Common letters: c creates, x extracts, t lists, f names the file, z uses gzip compression and v shows names as it works.
`tar -czf backup.tar.gz project/` creates a compressed archive. `tar -xzf backup.tar.gz` extracts it into the current folder, overwriting files with the same names. `tar -tzf backup.tar.gz` lists the contents without extracting.

## zip, create zip archive
Packs files and folders into a .zip archive that Windows and macOS can also open. -r includes folders.
`zip -r photos.zip photos/` creates the archive. `zip -e secret.zip file.txt` asks for a password and encrypts the archive.

## unzip, extract zip archive
Extracts a .zip archive into the current folder. It asks before overwriting an existing file, unless you use -o (overwrite without asking) or -n (never overwrite).
`unzip photos.zip` extracts it. `unzip -l photos.zip` lists the contents without extracting. `unzip -d out/ photos.zip` extracts into the folder out.

## gzip, compress one file
Compresses one file into a .gz file. By default it deletes the original file after compressing.
`gzip big.log` creates big.log.gz and removes big.log. `gzip -k big.log` keeps the original too.

## gunzip, decompress gz file
Decompresses a .gz file. By default it deletes the .gz file after writing the decompressed one.
`gunzip big.log.gz` produces big.log and removes big.log.gz. `gunzip -k big.log.gz` keeps both files.

## curl, send requests and transfer data
Sends a request to a web address and prints the response. Use it to test web pages and APIs or to download files.
`curl https://example.com` prints the page. `curl -I https://example.com` shows only the response headers. `curl -L -o file.zip https://example.com/file.zip` follows redirects and saves the download as file.zip.

## wget, download files
Downloads files from the web and can resume an interrupted download. It saves the file under its remote name in the current folder.
`wget https://example.com/file.iso` downloads the file. `wget -c https://example.com/file.iso` continues a partial download. `wget -O name.txt https://example.com/data` saves it under a new name.

## ssh, secure remote shell
Opens a command shell on another computer over an encrypted connection. You log in with a password or, better, with an SSH key.
`ssh alice@192.168.1.20` logs in as alice. `ssh -p 2222 alice@server.example.com` uses port 2222. `ssh-keygen -t ed25519` creates a key pair, keeping the private key secret.

## ssh-keygen, create SSH key pair
Creates a pair of SSH keys: a private key that stays on your computer and a public key that you place on servers you want to log in to.
`ssh-keygen -t ed25519` creates a modern key pair. `ssh-copy-id alice@server.example.com` copies the public key to a server.

## scp, copy files over ssh
Copies files to or from another computer over an encrypted SSH connection. An existing file with the same name at the destination is overwritten.
`scp report.pdf alice@server:/home/alice/` uploads a file. `scp alice@server:/var/log/app.log .` downloads one into the current folder. `scp -r site/ alice@server:/var/www/` copies a folder.

## rsync, sync folders
Copies only changed parts of files, so repeat copies are fast. It works locally or over ssh. With --delete it also removes files in the destination that are missing from the source, which destroys them there.
`rsync -av --progress src/ dest/` copies with progress shown. `rsync -av src/ alice@server:/backup/` copies over ssh. `rsync -av --delete src/ dest/` makes dest match src exactly. A trailing slash on src/ copies its contents.

## systemctl, manage services
Starts, stops, enables and checks services managed by systemd, which most current Linux systems use.
`systemctl status nginx` shows whether the service is running. `sudo systemctl restart nginx` restarts it. `sudo systemctl enable nginx` makes it start at every boot.

## journalctl, read system logs
Shows the logs collected by systemd's journal, for the whole system or for one service.
`journalctl -u nginx` shows the logs of one service. `journalctl -b` shows logs since the last boot. `journalctl -f` follows new entries. `sudo journalctl --since "1 hour ago"` shows the last hour.

## apt, install and update packages
The package manager of Debian and Ubuntu systems. It installs, upgrades and removes software from online repositories, and changes need sudo.
`sudo apt update` refreshes the list of available packages. `sudo apt install git` installs a program. `sudo apt upgrade` installs newer versions of installed packages.

## apt remove, uninstall package
Removes an installed program but keeps its configuration files. The purge form also removes the configuration files.
`sudo apt remove nano` removes nano. `sudo apt purge nano` removes it together with its configuration files.

## dpkg, install or inspect .deb packages
The low-level tool for Debian packages. It installs a local .deb file and lists or inspects the packages on the system. It does not download dependencies by itself.
`sudo dpkg -i tool.deb` installs a local package. `dpkg -l | grep nginx` lists installed packages whose names match. `dpkg -L git` lists the files that the git package installed.

## snap, Snap packages
Manages Snap packages, self-contained apps that bring their own libraries and are updated by the snapd service. Ubuntu uses them for many apps.
`snap list` shows installed snaps. `sudo snap install hello-world` installs a test snap. `sudo snap remove hello-world` removes it.

## ln, create links
Creates a link to a file or folder. A symbolic link (-s) is a pointer that can cross disks. A hard link is a second name for the same data on the same disk. Deleting a symbolic link leaves its target in place.
`ln -s /usr/bin/python3 ~/bin/python` creates a symbolic link. `ln original.txt alias.txt` creates a hard link.

## which, find a command's location
Shows the full path of the program that runs when you type a command name, as found through the PATH variable.
`which python3` might print `/usr/bin/python3`. `which -a python3` lists every match in PATH.

## man, read manual pages
Shows the manual page for a command, with its options. Press space to scroll, / to search and q to quit.
`man ls` opens the ls manual. `man -k network` searches manual page titles for a word. `man 5 crontab` opens the page on the crontab file format.

## sudo, run as administrator
Runs one command with administrator (root) rights, after asking for your password. A mistake under sudo can damage the whole system, so use it only when a command needs it.
`sudo apt update` runs one command as root. `sudo -u postgres psql` runs a command as the user postgres. `sudo -i` opens a root shell, which you should leave as soon as possible.

## ip, network interfaces and routes
Shows and changes IP addresses, network interfaces and routes. It replaces the older ifconfig and route commands.
`ip addr` lists the IP addresses on each interface. `ip route` shows the default gateway. `ip link` shows whether each interface is up.

## ss, socket and port status
Lists network connections and the ports programs are listening on. The -p option shows the program, which needs sudo to see programs owned by other users.
`ss -tlnp` lists TCP ports in listening state with their program. `ss -tn` lists current TCP connections. `ss -uln` lists UDP ports in listening state.

## ping, test reachability
Sends small test packets to a host and reports whether it answers and how long the reply takes. Some hosts block ping, so no reply does not always mean the host is down.
`ping -c 4 example.com` sends four packets and stops. `ping -c 4 192.168.1.1` tests the local router.

## lsblk, list disks and partitions
Shows block devices such as disks and partitions as a tree, with sizes and mount points.
`lsblk` shows the tree. `lsblk -f` adds file system types and labels.

## mount, attach a file system
Attaches a disk or partition to a folder so its files can be read. The folder must exist first, and mounting usually needs sudo.
`sudo mount /dev/sdb1 /mnt/usb` mounts the partition on /mnt/usb. `mount` with no arguments lists what is mounted.

## umount, detach a file system
Detaches a mounted disk so it can be removed safely. It fails with "target is busy" while a program has a file open on that disk.
`sudo umount /mnt/usb` detaches the disk mounted there. Close the programs using it first.

## uname, system information
Prints information about the system and its kernel.
`uname -a` prints the kernel name, host name, kernel release and machine type. `uname -r` prints only the kernel release.

## whoami, current user name
Prints the name of the user you are logged in as.
`whoami` might print `alice`. After `sudo -i` it prints `root`.

## id, user and group IDs
Shows your user ID, your primary group and the groups you belong to.
`id` prints all three. `id -un` prints only the user name.

## env, environment variables
Lists the environment variables of the current shell, or runs a program with changed variables.
`env` lists them all. `env | grep PATH` shows the PATH variable. `env LANG=C ls` runs ls with LANG set for that one command.

## export, set environment variable
Makes a shell variable available to the programs that the shell starts. Without export, a variable stays inside the current shell.
`export EDITOR=nano` sets it for this session. To keep it, add the line to the ~/.bashrc file.

## alias, shortcut command name
Gives a short name to a longer command. An alias lasts for the current shell unless it is saved in a startup file such as ~/.bashrc.
`alias ll='ls -lh'` creates it. `alias` lists all aliases. `unalias ll` removes it.

## history, past commands
Shows the commands typed earlier in this shell. Ctrl+R searches them while you type.
`history` prints the list. `history | grep docker` finds docker commands. `!!` repeats the last command.

## xargs, run a command on many items
Reads items from standard input and passes them as arguments to another command. It is the usual way to act on a list of files from find.
`find . -name "*.bak" -print0 | xargs -0 rm` deletes every .bak file it finds, so run the find part alone first. `echo a b c | xargs mkdir` creates folders a, b and c.

## sort, sort lines
Sorts lines of text alphabetically by default. -n sorts numbers and -r reverses the order.
`sort names.txt` sorts alphabetically. `sort -n sizes.txt` sorts numbers by value. `sort -k2 -n data.txt` sorts by the second column.

## uniq, remove repeated lines
Removes repeated lines that are next to each other. Sort the input first, because repeats that are apart are not removed.
`sort words.txt | uniq` lists each word once. `sort words.txt | uniq -c` counts how often each word appears.

## wc, count lines words bytes
Counts lines, words and bytes in a file or in piped input.
`wc -l access.log` prints the number of lines. `wc -w essay.txt` counts words. `ls | wc -l` counts the entries in a folder.

## cut, extract columns
Keeps selected fields or characters from each line. -d sets the delimiter that separates fields.
`cut -d: -f1 /etc/passwd` prints the first field of each line. `cut -c1-10 file.txt` keeps the first ten characters of each line.

## tr, translate or delete characters
Changes or deletes characters in its standard input. It works on characters, not on file names.
`echo hello | tr a-z A-Z` prints HELLO. `tr -d '\r' < dos.txt > unix.txt` writes a copy without Windows line-end characters, replacing unix.txt if it exists. `tr -s ' '` squeezes repeated spaces into one.

## tee, write output to screen and file
Passes its input on to the screen and also writes it to a file. It overwrites the file unless -a is given, which appends instead.
`ls -l | tee listing.txt` shows and saves the listing. `ping -c 3 example.com | tee -a ping.log` adds the output to ping.log.

## diff, compare files
Shows the differences between two files, or between two folders with -r. Lines starting with < come from the first file and lines starting with > come from the second.
`diff old.txt new.txt` compares two files. `diff -u old.txt new.txt` uses the unified format that patches use. `diff -r dir1 dir2` compares two folders.

## date, print or set date
Prints the current date and time, and can format them. Setting the clock needs sudo and is rarely needed.
`date` prints the date and time. `date +%F` prints the date only, such as 2026-11-15. `date -u` prints the time in UTC.

## crontab, schedule commands
Edits the list of commands that run automatically at set times for your user. crontab -r deletes the whole list at once, with no confirmation.
`crontab -e` opens the list in an editor. `crontab -l` prints it. `crontab -r` deletes every scheduled job of the current user.

## nohup, keep running after logout
Runs a command so that it keeps going after you close the terminal or log out. Its output goes to nohup.out unless you redirect it.
`nohup ./server.sh > server.log 2>&1 &` runs the script in the background and writes its output to server.log.

## watch, repeat a command
Runs a command again at a fixed interval and shows the updated output on screen. The default interval is two seconds.
`watch df -h` refreshes disk usage every two seconds. `watch -n 10 "ls -l incoming/"` checks every ten seconds. Press Ctrl+C to stop.

## echo, print text
Prints text followed by a new line. Scripts use it to show messages and the value of variables.
`echo "Hello"` prints Hello. `echo $HOME` prints the home folder path. `echo "data" >> log.txt` appends a line to log.txt.

## printf, formatted output
Prints text in a set format and adds no new line unless you write \n. Use it when the exact output matters.
`printf "%05d\n" 42` prints 00042. `printf "%s has %d files\n" "$dir" 12` prints a sentence with a value in it.

## file, identify file type
Guesses what kind of data a file holds by reading its start, not by its name.
`file photo.jpg` might print a JPEG image description. `file *` lists the type of each item in the folder.

## stat, file details
Shows a file's size, permissions, owner and the times it was last changed, accessed and modified.
`stat notes.txt` prints all of these details.

## basename, last part of a path
Prints the last part of a path, which is usually the file name. Scripts use it to strip folders off a path.
`basename /home/alice/report.pdf` prints report.pdf. `basename -s .pdf /home/alice/report.pdf` prints report.

## dirname, folder part of a path
Prints everything in a path except the last part, which is the folder that holds it.
`dirname /home/alice/report.pdf` prints /home/alice.

## realpath, absolute path
Prints the full absolute path of a file or folder, with symbolic links resolved.
`realpath notes.txt` prints the full path of notes.txt in the current folder.

## seq, print a number sequence
Prints a list of numbers, one per line.
`seq 5` prints 1 to 5. `seq 0 2 10` prints 0, 2, 4, 6, 8 and 10.

## sleep, pause for a time
Waits for the given time, in seconds by default. Add a suffix m, h or d for minutes, hours or days.
`sleep 10` waits ten seconds. `sleep 2m` waits two minutes. `sleep 5; echo done` prints done after five seconds.

## timeout, limit how long a command runs
Runs a command and stops it when the time limit is reached. If it was stopped, the exit status is 124.
`timeout 30 ping example.com` stops the ping after 30 seconds. `timeout 10s curl -I https://example.com` gives the request ten seconds.

## time, measure command duration
Runs a command and reports how long it took, in real, user and system time.
`time sleep 3` reports about three seconds of real time. In Bash, time is a keyword that works before any command.

## jobs, list background jobs
Lists the jobs started from the current shell, including those running in the background with &.
`sleep 100 &` starts a job in the background, and `jobs` lists it. `fg %1` brings job 1 to the front and `bg %1` resumes it in the background.

## lsof, open files and network ports
Lists files opened by processes. With -i it shows which process uses a network port.
`sudo lsof -i :8080` shows the process listening on port 8080. `lsof -p 1234` lists the files open by process 1234.

## hostname, machine name
Prints the name of the computer. Setting it needs sudo and changes the name the machine uses.
`hostname` prints the name. `hostname -I` prints the machine's IP addresses.

## uptime, time since boot
Shows how long the system has been running, how many users are logged in and the load average.
`uptime` prints a line such as "up 3 days" with the load figures.

## reboot, restart computer
Restarts the computer at once. Unsaved work in open programs is lost.
`sudo reboot` restarts now.

## shutdown, power off computer
Turns the computer off, now or after a delay. Unsaved work in open programs is lost.
`sudo shutdown -h now` powers off now. `sudo shutdown -h +10` powers off in ten minutes.

## passwd, change password
Changes your login password. It asks for the current password first. With sudo and a user name it changes another user's password.
`passwd` changes your own password. `sudo passwd alice` sets a new password for user alice.

## dig, DNS lookup
Asks DNS servers for records of a name, such as its IP address or mail servers. Use +short for plain output.
`dig example.com A +short` prints the IPv4 address. `dig example.com MX +short` lists mail servers. `dig example.com TXT +short` lists text records.

## dmesg, kernel messages
Prints messages from the Linux kernel, such as hardware detection and driver events. Recent messages are at the end.
`sudo dmesg | tail -n 20` shows the last 20 kernel messages. Plug a USB device in and run it to see what the kernel reports.

## sha256sum, file checksum
Computes a SHA-256 checksum, a fingerprint of a file's content. Compare it with the published value to check that a download is complete and unchanged.
`sha256sum ubuntu.iso` prints the checksum. `sha256sum -c SHA256SUMS` checks every listed file against its stored checksum.

## nano, simple text editor
A small terminal text editor. Its keys are shown at the bottom of the screen.
`nano notes.txt` opens the file. Ctrl+O saves, and Ctrl+X exits.

## dd, copy raw data blocks
Copies data block by block, including whole disks or partitions. Writing to the wrong target overwrites that device completely, and every byte on it is replaced.
`sudo dd if=ubuntu.iso of=/dev/sdX bs=4M status=progress` writes an image to the device /dev/sdX, which erases everything on it. Check the device name with lsblk first.
