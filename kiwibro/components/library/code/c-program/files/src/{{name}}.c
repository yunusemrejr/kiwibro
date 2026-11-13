#include "{{name}}.h"

#include <ctype.h>

/* KIWIBRO: replace this logic with the work your program does */
Counts count_buffer(const char *data, size_t length)
{
    Counts counts = {0, 0, 0};
    int in_word = 0;

    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)data[i];
        if (c == '\n') {
            counts.lines++;
        }
        if (isspace(c)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            counts.words++;
        }
    }
    counts.bytes = length;
    return counts;
}

size_t longest_line(const char *data, size_t length)
{
    size_t longest = 0;
    size_t current = 0;

    for (size_t i = 0; i < length; i++) {
        if (data[i] == '\n') {
            if (current > longest) {
                longest = current;
            }
            current = 0;
        } else {
            current++;
        }
    }
    return current > longest ? current : longest;
}
