# Git version control reference

## git init, start a new repository
Creates a Git repository in the current folder by adding a hidden .git folder that stores the history. Nothing is tracked until you add files.
`git init` works in any project folder. `git init -b main` names the first branch main.

## git clone, copy a repository
Downloads a full copy of an existing repository, with its history, into a new folder. The copy keeps a link to the original under the name origin.
`git clone https://example.com/team/app.git` creates a folder named app. `git clone --depth 1 https://example.com/team/app.git` downloads only the latest snapshot, which is faster but has no older history.

## git status, show what changed
Shows the current branch, which files are modified, which are staged for the next commit, and which are untracked, meaning new files Git does not follow yet. It changes nothing.
`git status` prints the full summary. `git status -s` prints a short form.

## git add, stage changes
Moves changes from the working folder into the staging area, so they become part of the next commit. Running it again on a file stages that file's latest version.
`git add report.txt` stages one file. `git add .` stages all changes in the current folder and below. `git add -p` lets you choose which parts of a file to stage.

## git commit, save a snapshot
Records everything in the staging area as a new commit with a message describing the change. Each commit has an ID and points to its parent commit.
`git commit -m "Fix login redirect"` commits with a one-line message. `git commit` alone opens an editor for a longer message.

## git log, show history
Lists the commits of the current branch, newest first, with their ID, author, date and message.
`git log --oneline` prints one line per commit. `git log --oneline --graph --all` draws all branches. `git log -p` adds the changes of each commit.

## git diff, show unstaged changes
Shows the line-by-line changes in files that are not yet staged. Lines starting with - were removed, and lines starting with + were added.
`git diff` shows all unstaged changes. `git diff report.txt` shows the changes to one file.

## git diff --staged, show staged changes
Shows what is in the staging area compared with the last commit. This is exactly what the next commit will contain.
`git diff --staged` prints it.

## git branch, list, create and delete branches
A branch is a movable name that points at a commit. This command lists branches, creates one or deletes one. Creating a branch does not switch to it.
`git branch` lists local branches and marks the current one with a star. `git branch feature-login` creates a branch. `git branch -d feature-login` deletes it, but only when its commits are merged.

## git branch -D, force delete a branch
Deletes a branch even when its commits are not merged anywhere. Commits that existed only on that branch can be lost, unless you find their ID in git reflog.
`git branch -D old-experiment` deletes the branch at once. Use it only when you are sure its work is not needed.

## git switch, change branch
Changes the current branch and updates the files in the working folder to match it. Uncommitted changes that would be overwritten stop the switch.
`git switch main` changes to main. `git switch -c fix-typo` creates a new branch and switches to it.

## git checkout, older branch and file command
An older command that can switch branches like git switch, and can also copy a file from a commit or the staging area. `git checkout -- file` throws away your uncommitted edits to that file.
`git checkout main` switches branch. `git checkout -b new-idea` creates and switches. `git checkout -- README.md` discards uncommitted edits to README.md.

## git merge, join another branch
Brings the commits of another branch into the current branch. When both branches changed the same lines, Git stops and reports a merge conflict.
`git switch main` followed by `git merge feature-login` merges that branch into main.

## git merge --abort, cancel a conflicted merge
Cancels a merge that stopped with conflicts and returns the branch to the state it had before the merge.
`git merge --abort` is run while the merge conflict is still open.

## git rebase, replay commits on a new base
Takes your commits on a branch and replays them on top of another commit, which gives a straight history. It creates new commit IDs, so do not rebase commits that other people already use.
`git rebase main` replays the current branch on top of the latest main.

## git rebase -i, edit recent commits
Opens a list of recent commits so you can reorder, combine, reword or drop them. It rewrites history, so use it only on commits that are not shared yet.
`git rebase -i HEAD~3` edits the last three commits. Deleting a line from the list drops that commit.

## git rebase --abort, cancel a rebase
Stops a rebase in progress and returns the branch to the state it had before the rebase started.
`git rebase --abort` is run while the rebase is still stopped on a conflict.

## git stash, set aside changes
Saves your uncommitted changes and returns the working folder to the last commit. Use it when you must switch branches in the middle of a task.
`git stash` saves the changes. `git stash push -m "login draft"` saves them with a name. `git stash list` shows the saved stashes.

## git stash pop, restore stashed changes
Applies the most recent stash to the working folder and removes it from the stash list. If applying it causes a conflict, the stash stays in the list.
`git stash pop` restores the last stash. `git stash apply` restores it and keeps it in the list.

## git reset, move a branch back
Moves the current branch to an earlier commit. The three main forms differ in what they do with your changes: --soft keeps them staged, --mixed keeps them unstaged, and --hard discards them.
`git reset --soft HEAD~1` undoes the last commit and keeps its changes staged. `git reset HEAD~1` undoes it and keeps the changes unstaged.

## git reset --hard, discard uncommitted changes
Moves the branch to another commit and overwrites the working folder to match it. Uncommitted edits and staged changes to tracked files are lost for good, and Git cannot bring them back.
`git reset --hard HEAD` throws away all uncommitted changes to tracked files. Untracked new files are not touched by it.

## git revert, undo a commit safely
Creates a new commit that reverses the changes of an earlier commit. The history is kept, so this is the right way to undo a commit that was already pushed.
`git revert HEAD` undoes the latest commit. `git revert abc1234` undoes the commit with that ID.

## git restore, undo file changes
Replaces a file in the working folder with its version from the staging area or from the last commit. Edits that were not committed are lost. With --staged it only removes a file from the staging area and leaves its content alone.
`git restore notes.txt` discards your edits to notes.txt. `git restore --staged notes.txt` unstages it.

## git rm, remove a tracked file
Deletes a file from the working folder and stages the deletion, so the next commit removes it from the project. The file stays in older commits.
`git rm old-config.txt` deletes it. `git rm --cached secret.env` stops tracking the file but keeps it on disk.

## git mv, move or rename a tracked file
Moves or renames a file and stages the change in one step.
`git mv old-name.txt new-name.txt` renames the file in the repository.

## git clean, delete untracked files
Deletes files that Git does not track, such as new files not yet added or build output. The files are removed permanently and do not go to the trash.
`git clean -n` lists what would be deleted, without deleting anything. `git clean -fd` deletes the untracked files and folders.

## git commit --amend, change the last commit
Replaces the most recent commit with a new one that includes any staged changes and, optionally, a new message. The commit gets a new ID, so do not amend a commit that was already pushed.
`git commit --amend -m "Better message"` changes only the message. `git commit --amend --no-edit` adds the staged changes to the last commit and keeps its message.

## git cherry-pick, copy one commit
Applies the changes of one commit onto the current branch as a new commit with a new ID.
`git cherry-pick abc1234` copies that commit onto the current branch.

## git fetch, download remote changes
Downloads commits and branches from a remote without changing your own branches or files. Remote-tracking branches such as origin/main are updated.
`git fetch` updates all remote-tracking branches. `git fetch --prune` also removes the ones that were deleted on the remote.

## git pull, fetch and merge
Downloads changes from the remote and merges them into the current branch. With --rebase it replays your own commits on top of the downloaded ones instead.
`git pull` updates the current branch from its remote. `git pull --rebase` keeps your commits on top, without a merge commit.

## git push, upload commits
Sends your commits to a remote repository. The first push of a new branch can set its upstream, so later pushes and pulls know where it goes.
`git push` sends the current branch. `git push -u origin feature-login` sends a new branch and links it to origin.

## git push --force, overwrite remote history
Replaces the branch on the remote with your local version, deleting remote commits that are not in yours. Other people's commits on that branch can be lost. Prefer git push --force-with-lease.
`git push --force origin feature-login` overwrites the remote branch.

## git push --force-with-lease, safer force push
A force push that refuses to overwrite the remote branch when someone has pushed to it since your last fetch. This protects commits you have not seen.
`git push --force-with-lease origin feature-login` replaces the remote branch only if it is still the version you last fetched.

## git tag, name a commit
Gives a fixed name to a commit, often a release point. A lightweight tag is only a name. An annotated tag (-a) also stores a message, the author and the date.
`git tag v1.0` tags the current commit. `git tag -a v1.0 -m "First stable version"` creates an annotated tag. `git push origin v1.0` sends that tag to origin.

## git remote, manage remote addresses
Lists, adds, renames and removes the addresses of remote repositories. The remote created by clone is named origin.
`git remote -v` lists names and addresses. `git remote add upstream https://example.com/original/app.git` adds another remote. `git remote remove upstream` removes it.

## origin, default remote name
The name Git gives to the repository you cloned from. It is only a name and can be changed.
`git push origin main` sends main to the remote named origin.

## git blame, see who changed each line
Shows, for each line of a file, the commit and author that last changed it. It helps find why a line exists.
`git blame app.py` prints the annotations for every line. `git blame -L 10,20 app.py` limits the output to lines 10 to 20.

## git bisect, find the commit that broke something
Searches the history by halving it, to find the commit that introduced a bug. You mark one commit as good and one as bad, and Git picks commits in between to test.
`git bisect start` begins. `git bisect bad` marks the current commit as bad and `git bisect good abc1234` marks a known good one. `git bisect reset` ends the search.

## git reflog, history of HEAD moves
Lists the places HEAD has pointed to recently, including commits that are no longer on any branch. It is the main way to recover commits after a reset or a deleted branch. Old entries are cleaned up after a time.
`git reflog` prints the list with commit IDs. `git branch recovered-work abc1234` saves a lost commit under a new branch name.

## git worktree, second working folder
Creates an extra working folder that holds another branch of the same repository, so you can work on two branches at once without stashing. The worktrees share one history.
`git worktree add ../app-hotfix hotfix` creates the folder app-hotfix with the branch hotfix checked out. `git worktree remove ../app-hotfix` removes that folder.

## .gitignore, files Git should ignore
A text file in the repository that lists file names and patterns Git must not track, such as build output, logs and passwords. Files already tracked stay tracked after they are listed.
A line `*.log` ignores all log files. A line `build/` ignores the build folder. Put it in the top folder of the repository.

## git config, settings
Reads and sets Git options. With --global the setting applies to all repositories of your user. Without it, the setting applies only to the current repository.
`git config --global user.name "Your Name"` and `git config --global user.email you@example.com` set the author name and email on your commits.

## git show, show one commit
Shows the message, author and changes of one commit. It can also print a file as it was at that commit.
`git show abc1234` prints one commit. `git show abc1234:README.md` prints README.md as it was in that commit.

## git help, built-in manual
Opens the manual page for a Git command.
`git help rebase` opens the rebase manual. `git merge --help` does the same for merge.

## git grep, search tracked files
Searches the files the repository tracks and prints matching lines with file names and line numbers.
`git grep "TODO"` lists every tracked line that contains TODO.

## .git folder
The hidden .git folder inside a repository holds its whole history, branches and settings. Deleting it removes the version history but leaves the current files in place.
Do not delete it by accident. Copy the whole project folder if you need a backup of its history.

## working tree
The folder of files you see and edit, as opposed to the .git folder that holds the history.

## tracked file
A file that Git already knows about, because it was added and committed before. Edits to a tracked file show up in git status as modified.

## untracked file
A file in the working folder that Git does not yet track. It appears in git status and is not part of any commit until you run git add on it.

## staging area
The place where changes wait before they are committed. git add puts changes into it, and git commit saves exactly what it holds. It is also called the index.

## commit ID, commit hash
A long code that identifies one commit, shown as hexadecimal digits. Git accepts a shortened form, usually the first seven characters, as long as it is unique.
`abc1234` is a short commit ID. `git show abc1234` uses it.

## HEAD, current position
HEAD is the name Git uses for the commit you are working on now. Usually it points to the current branch, which points to the latest commit of that branch.
`HEAD~1` means the commit before HEAD, and `HEAD~3` means three commits before it.

## commit
A saved snapshot of the project, with a message, an author, a date and a link to the commit before it. Commits form the history of a project.

## branch
A movable name that points at a commit. When you commit on a branch, the name moves forward to the new commit, and the other branches stay where they were.

## remote
Another copy of the repository, usually on a server, that you can fetch from and push to. Its address is stored in the local repository.

## remote-tracking branch
A local record of a branch on a remote, shown as origin/main. It changes when you fetch or pull, not when you commit.

## upstream branch
The remote branch that a local branch is linked to. git pull and git push use it when no other branch is named. The link is set with git push -u.

## fast-forward
A merge in which the target branch has no new commits of its own, so Git only moves its name forward to the newer commit. No merge commit is created.

## merge conflict
A stop that happens when two branches changed the same lines differently, and Git cannot choose one version alone. Git marks the file and waits for you to decide.

## how to resolve a merge conflict
Open each file named as conflicted. Between the markers <<<<<<<, ======= and >>>>>>> are the two versions. Edit the file to keep the text you want and delete the marker lines. Then run `git add` on the file and `git commit` to finish the merge.

## detached HEAD
A state where HEAD points at a commit instead of a branch, which happens after checking out a commit ID or a tag. New commits made there belong to no branch and can be lost when you switch away.
To keep them, create a branch at that point with `git switch -c my-work`.

## how to undo the last commit
To undo it and keep the changes staged, run `git reset --soft HEAD~1`. To undo it and keep the changes unstaged, run `git reset HEAD~1`. To undo it and discard the changes, run `git reset --hard HEAD~1`, which permanently loses those changes. If the commit was already pushed, use `git revert HEAD` instead.

## how to recover a deleted branch
Run `git reflog` and find the commit ID of the last commit on the deleted branch, which is shown when that branch was checked out in this clone. Then run `git branch branch-name abc1234` with that ID to recreate the branch at that commit.
