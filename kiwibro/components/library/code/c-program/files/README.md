# {{name}}

Print lines, words, bytes and the longest line for each file, or for standard
input when no file is given. Two functions do the work, `count_buffer` and
`longest_line`, and a test binary checks them.

## Build, run, test

    make
    ./build/{{name}} README.md
    make test

Binaries go to `build/`. There is no clean target on purpose, because nothing
here deletes files: remove that folder by hand when you want a fresh build.
