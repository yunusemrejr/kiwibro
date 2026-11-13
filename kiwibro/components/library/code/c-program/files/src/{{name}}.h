#ifndef {{name}}_H
#define {{name}}_H

#include <stddef.h>

typedef struct {
    size_t lines;
    size_t words;
    size_t bytes;
} Counts;

/* Count lines, words and bytes in length bytes of data. */
Counts count_buffer(const char *data, size_t length);

/* Length of the longest line, not counting its newline. */
size_t longest_line(const char *data, size_t length);

/* KIWIBRO: declare the functions your program needs here */

#endif
