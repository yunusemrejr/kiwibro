#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "{{name}}.h"

/* Count one stream a line at a time, so memory stays flat for big files. */
static int report(FILE *stream, const char *label)
{
    Counts total = {0, 0, 0};
    size_t longest = 0;
    char *line = NULL;
    size_t capacity = 0;
    ssize_t read;

    while ((read = getline(&line, &capacity, stream)) != -1) {
        Counts part = count_buffer(line, (size_t)read);
        size_t line_longest = longest_line(line, (size_t)read);

        total.lines += part.lines;
        total.words += part.words;
        total.bytes += part.bytes;
        if (line_longest > longest) {
            longest = line_longest;
        }
    }
    free(line);

    if (ferror(stream)) {
        fprintf(stderr, "{{name}}: cannot read %s: %s\n", label, strerror(errno));
        return 1;
    }
    printf("%7zu %7zu %7zu %7zu  %s\n", total.lines, total.words, total.bytes,
           longest, label);
    return 0;
}

int main(int argc, char **argv)
{
    printf("%7s %7s %7s %7s  %s\n", "lines", "words", "bytes", "longest", "file");

    if (argc < 2) {
        return report(stdin, "-");
    }

    int status = 0;
    for (int i = 1; i < argc; i++) {
        FILE *stream = fopen(argv[i], "r");
        if (stream == NULL) {
            fprintf(stderr, "{{name}}: cannot open %s: %s\n", argv[i], strerror(errno));
            status = 1;
            continue;
        }
        if (report(stream, argv[i]) != 0) {
            status = 1;
        }
        fclose(stream);
    }
    return status;
}
