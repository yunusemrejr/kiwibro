# Bash shell scripting reference

## Variables, assigning values
A variable is set with name=value, with no spaces around the equals sign. It is read with a dollar sign, as in $name, and unset variables are empty.
`name="Alice"` sets the variable, and `echo "Hello $name"` prints Hello Alice. `unset name` removes it.

## Double quotes
Double quotes keep the text together as one word and still expand variables and command substitutions inside them. Always quote variables that may hold spaces.
`file="my report.txt"` then `cat "$file"` reads the right file. Without quotes, the shell splits the value at each space.

## Single quotes
Single quotes keep every character literal. Nothing inside them is expanded.
`echo '$HOME'` prints $HOME, while `echo "$HOME"` prints the home folder path.

## Backslash escape
A backslash makes the next character literal, such as a space, a dollar sign or a quote.
`echo "Price: \$5"` prints Price: $5. `mv my\ file.txt backup.txt` moves a file whose name contains a space.

## $? exit status
After each command, $? holds its exit status. Zero means success and any other value means failure.
`grep -q "error" app.log; echo $?` prints 0 if the text was found and 1 if it was not.

## Exit codes
Programs report success with 0 and failure with a number from 1 to 255. The shell uses 127 when a command name is not found and 126 when a file exists but cannot be run.
`exit 1` ends a script with failure status 1. A calling script can check it with $?.

## if and test
An if statement runs commands depending on whether a condition succeeds. The [ ] brackets are the test command, so spaces inside them are required.
`if [ -f notes.txt ]; then echo "found"; fi` prints found when notes.txt is a regular file.

## test operators
Test operators check files and values. -f means a regular file exists, -d means a folder exists, -z means a string is empty, and -eq, -lt and -gt compare numbers.
`[ -d ~/backup ]` checks for a folder. `[ "$count" -gt 10 ]` checks whether count is greater than 10.

## [[ ]] double brackets
The [[ ]] form is a Bash test that handles spaces in variables safely and supports pattern matching with ==.
`if [[ "$file" == *.txt ]]; then echo "text file"; fi` matches names ending in .txt.

## for loop
Runs a block once for each item in a list.
`for f in *.txt; do echo "$f"; done` prints every text file name. `for i in {1..3}; do echo "$i"; done` prints 1, 2 and 3.

## while loop
Repeats a block as long as its condition succeeds. A common use is reading a file line by line.
`while read -r line; do echo "$line"; done < list.txt` prints each line of list.txt. The -r option stops backslashes from being treated as escapes.

## case statement
Chooses one block by matching a value against patterns. It is clearer than many nested if statements.
`case "$1" in start) echo "starting" ;; stop) echo "stopping" ;; *) echo "unknown" ;; esac` handles three choices, the last one as a default.

## Functions
A function groups commands under a name so they can be called again. Inside it, $1 and the following numbers hold its arguments, and local keeps a variable inside the function.
`greet() { local who="$1"; echo "Hello $who"; }` then `greet Alice` prints Hello Alice.

## Positional parameters
In a script, $1, $2 and so on hold the arguments given on the command line. $# is the count of arguments, $0 is the script name, and "$@" holds all the arguments as separate words.
`./run.sh input.txt 10` gives $1 as input.txt and $2 as 10.

## shift
Removes the first argument and moves the others down one place, so $2 becomes $1.
`shift` in a loop lets a script work through its arguments one at a time.

## Pipe |
A pipe sends the standard output of one command to the standard input of the next, so commands can be chained.
`cat access.log | grep "404" | wc -l` counts the lines that contain 404.

## Redirect > (overwrite)
Sends standard output to a file. The file is created or, if it exists, its previous content is erased first.
`ls -l > listing.txt` replaces listing.txt with the listing.

## Redirect >> (append)
Sends standard output to the end of a file, keeping what is already there.
`date >> log.txt` adds the current date as a new last line of log.txt.

## Redirect 2> and 2>&1
Error messages go to standard error, which is file descriptor 2. 2> sends them to a file, and 2>&1 sends them to the same place as standard output.
`make 2> errors.txt` saves only errors. `make > build.log 2>&1` saves both output and errors in build.log.

## /dev/null
A special file that discards everything written to it.
`grep -q "error" app.log > /dev/null 2>&1` runs the search and throws away all of its output.

## Here-document
Feeds several lines of text to a command's standard input, ending at a line that holds only the marker word.
`cat > notes.txt << EOF` followed by lines of text and a final line with EOF writes those lines to notes.txt, overwriting it. Writing `<< 'EOF'` in quotes turns off variable expansion inside the text.

## Here-string
Feeds one string to a command's standard input.
`wc -w <<< "one two three"` prints 3.

## Globbing
The shell expands patterns in file names before running a command. * matches any text, ? matches one character, and [abc] matches one character from the set.
`ls *.jpg` lists all JPG files in the current folder. `rm file?.txt` only matches names such as file1.txt, so check what it matches first.

## Brace expansion
The shell turns a list in braces into several words. It creates the words only and does not check whether files exist.
`mkdir -p project/{src,docs,tests}` creates three folders. `echo {1..5}` prints 1 2 3 4 5.

## Tilde expansion
A tilde at the start of a word stands for your home folder. ~user stands for another user's home folder.
`cd ~/Downloads` goes to Downloads in your home folder.

## Command substitution
Runs a command and puts its output in place of the $( ) form. Quote it when the output may contain spaces.
`today=$(date +%F)` stores the date in the variable today. `echo "Files: $(ls | wc -l)"` prints the number of entries.

## Arithmetic expansion
Evaluates a whole-number calculation inside $(( )).
`echo $((7 * 6))` prints 42. `count=$((count + 1))` adds one to count.

## Parameter defaults
${name:-default} uses the default when the variable is empty or unset. ${#name} gives the length of the value.
`port=${PORT:-8080}` sets port to 8080 when PORT is not set. `echo ${#name}` prints the number of characters in name.

## Arrays
An array holds a list of values with index numbers, starting at 0.
`files=(a.txt b.txt c.txt)` creates it. `echo "${files[0]}"` prints a.txt, and `echo "${files[@]}"` prints all the values.

## Process substitution
Treats the output of a command as a file name, so tools that need files can read command output directly.
`diff <(sort a.txt) <(sort b.txt)` compares two files after sorting them, without temporary files.

## && and ||
The && operator runs the second command only if the first succeeds, and || runs the second only if the first fails.
`mkdir build && cd build` changes folder only when mkdir worked. `cd project || exit 1` stops the script when the folder is missing.

## Subshell ( )
Runs commands in a separate child shell. Changes to the directory or variables inside the parentheses do not affect the current shell.
`(cd logs && tar -czf ../logs.tar.gz .)` makes an archive without leaving the current folder.

## set -e, stop on error
Makes the script exit as soon as any command fails, instead of carrying on with a broken state.
`set -e` at the top of a script stops it at the first failing command.

## set -u, error on unset variable
Makes Bash treat an unset variable as an error, instead of using an empty value. This catches typing mistakes in names.
`set -u` followed by `echo "$nmae"` fails with an error instead of printing a blank line.

## set -o pipefail
Makes a pipeline fail when any command in it fails, not only the last one.
`set -o pipefail` at the top of a script stops `cat missing.txt | sort` from looking like success.

## set -euo pipefail
The usual safe start for a script. -e stops on the first error, -u stops on unset variables, and pipefail makes pipelines report failures.
Put `set -euo pipefail` on the line after the shebang line of a script.

## set -x, trace commands
Prints each command before it runs, with its expanded arguments, so you can see what a script does. The output goes to standard error.
`bash -x backup.sh` traces a whole script. `set +x` turns tracing off inside a script.

## trap
Runs a command when the script receives a signal or exits. Use it to clean up temporary files.
`trap 'rm -f "$tmpfile"' EXIT` deletes the temporary file however the script ends. Careful: the trap runs even when the script fails.

## getopts
Parses single-letter options such as -v and -o value in a script.
`while getopts "vo:" opt; do case $opt in v) verbose=1 ;; o) out="$OPTARG" ;; esac; done` reads -v and -o file.

## PATH variable
A list of folders, separated by colons, where the shell looks for commands. The first match wins.
`echo $PATH` prints the list. `export PATH="$HOME/bin:$PATH"` adds a folder to the front of it.

## Environment variables
Variables that the shell passes to programs it starts. A shell variable becomes an environment variable when it is marked with export.
`export API_URL="https://example.com"` makes API_URL visible to programs run from this shell.

## Startup file ~/.bashrc
A script that Bash runs for each new interactive shell. It is the usual place for aliases, PATH changes and prompt settings.
Add `export EDITOR=nano` to ~/.bashrc and open a new terminal, or run `source ~/.bashrc`, to use it.

## source or dot command
Runs a script in the current shell instead of a child process, so variables and functions it sets remain available.
`source ~/.bashrc` reloads the startup file. `. ./settings.sh` loads settings from settings.sh.

## read command
Reads one line of input into one or more variables.
`read -r name` waits for typing and stores it in name. `read -p "Name: " name` shows a prompt first.

## Background job &
An ampersand at the end of a command starts it in the background, so the shell is free to take more commands.
`sleep 60 &` starts a one-minute wait in the background. `jobs` lists it.

## Job control: Ctrl+C, Ctrl+Z, fg and bg
Ctrl+C stops the running command. Ctrl+Z pauses it and returns to the prompt. fg brings a job back to the front, and bg lets a paused job run in the background.
`fg %1` resumes job 1 in the foreground.

## Permissions rwx
Each file has permissions for its owner, its group and everyone else. r means read, w means write and x means run. For a folder, x means it can be entered.
`ls -l` shows them as, for example, -rwxr-x---, where the first character is the file type and the next nine are three sets of three.

## Octal permission modes
Each permission is a number: read is 4, write is 2 and execute is 1. Add them for each of owner, group and others.
755 means rwxr-xr-x, 644 means rw-r--r--, 600 means rw------- and 700 means rwx------. `chmod 600 id_key` makes a file readable and writable by its owner only.

## Symbolic chmod
Changes permissions with letters: u for owner, g for group, o for others, and a for all. + adds, - removes and = sets exactly.
`chmod u+x script.sh` makes a file executable for its owner. `chmod go-w notes.txt` removes write permission for the group and others.

## Shebang line
The first line of a script, starting with #!, names the interpreter that runs it. Without it, the file runs in the shell that starts it.
`#!/usr/bin/env bash` as the first line runs the script with the Bash found in PATH.

## Making a script runnable
A script needs execute permission and a folder in the path you type to run it. Without a slash, the shell looks only in the PATH folders.
`chmod +x backup.sh` then `./backup.sh` runs it from the current folder.

## Comments
Text after # is ignored by the shell. Use it to explain why a line exists.
`# keep the last 7 backups` is a comment line in a script.

## Syntax check, bash -n
Reads a script and reports syntax errors without running any of its commands.
`bash -n backup.sh` prints errors if the script will not parse, and prints nothing when it is valid.

## Cron line format
A crontab line has five time fields and then the command. The fields are minute (0 to 59), hour (0 to 23), day of month (1 to 31), month (1 to 12) and day of week (0 to 7, where 0 and 7 are Sunday). An asterisk means every value.
`30 2 * * 1 /home/alice/backup.sh` runs the script at 02:30 every Monday.

## Cron special strings
Some crontab systems accept words in place of the five time fields. @reboot runs the command once at startup, and @daily runs it once a day at midnight.
`@reboot /home/alice/start-server.sh` runs the script after each boot.

## Cron environment and absolute paths
Cron runs commands with a very small PATH and no terminal. Use full paths to programs and files, and send output to a log, or the output is lost.
`0 3 * * * /usr/bin/rsync -a /home/alice/docs/ /backup/docs/ >> /home/alice/backup.log 2>&1` runs a backup at 03:00 every day and writes the output to a log.

## Quoting variables in scripts
An unquoted variable is split at spaces and expanded to file name patterns. Quoting it as "$file" keeps the value whole.
`cp "$src" "$dest"` works even when the paths contain spaces.

## Default values in scripts
${1:-default} uses a default when the argument is missing, which lets a script run without every argument.
`dir="${1:-.}"` uses the current folder when no argument is given.

## Error message to standard error
Errors should go to standard error, so that they still show when output is piped or saved.
`echo "missing input file" >&2` prints the message to standard error. Then `exit 1` ends the script with failure.

## Exit from a script
exit ends the script at once with the status number given.
`exit 0` ends with success and `exit 2` ends with status 2.

## Command grouping { }
Groups commands so they share one redirect or one pipeline. Unlike ( ), it runs in the current shell, and the closing semicolon is required.
`{ echo "start"; date; } > run.log` writes both lines into run.log.

## Quoted here-document marker
Writing the marker in quotes, such as << 'EOF', keeps the text literal, so $ signs and backticks are not expanded.
`cat << 'EOF' > template.txt` writes the lines exactly as typed.

## Loop over a sequence
seq prints numbers, and a for loop can run over them.
`for i in $(seq 1 5); do echo "step $i"; done` prints five lines.

## Pattern in for loop
A for loop with a glob runs once per matching file. If nothing matches, the pattern stays as written, so check that the file exists.
`for f in *.log; do gzip "$f"; done` compresses each log file in the folder and removes the originals.
