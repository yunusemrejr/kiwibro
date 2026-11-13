#include <stdio.h>
#include <string.h>

#include "{{name}}.h"

/* KIWIBRO: add checks for the functions you write */
static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__,      \
                    __LINE__, #cond);                                   \
            failures++;                                                 \
        }                                                               \
    } while (0)

static void test_empty_input(void)
{
    Counts counts = count_buffer("", 0);

    CHECK(counts.lines == 0);
    CHECK(counts.words == 0);
    CHECK(counts.bytes == 0);
    CHECK(longest_line("", 0) == 0);
}

static void test_two_lines(void)
{
    const char *text = "one two\nthree\n";
    Counts counts = count_buffer(text, strlen(text));

    CHECK(counts.lines == 2);
    CHECK(counts.words == 3);
    CHECK(counts.bytes == strlen(text));
    CHECK(longest_line(text, strlen(text)) == 7);
}

static void test_line_without_final_newline(void)
{
    const char *text = "a  b";
    Counts counts = count_buffer(text, strlen(text));

    CHECK(counts.lines == 0);
    CHECK(counts.words == 2);
    CHECK(longest_line(text, strlen(text)) == 4);
}

static void test_longest_line_is_not_the_last(void)
{
    const char *text = "a\nlonger\nmid\n";

    CHECK(longest_line(text, strlen(text)) == 6);
}

int main(void)
{
    test_empty_input();
    test_two_lines();
    test_line_without_final_newline();
    test_longest_line_is_not_the_last();

    if (failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    puts("all checks passed");
    return 0;
}
