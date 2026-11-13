CC ?= cc
# KIWIBRO: warnings stay on; add -g here while debugging
CFLAGS ?= -std=c17 -Wall -Wextra -Wpedantic -Werror -O2
BUILD := build

all: $(BUILD)/{{name}}

$(BUILD)/{{name}}: src/main.c src/{{name}}.c src/{{name}}.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ src/main.c src/{{name}}.c

$(BUILD)/test_{{name}}: tests/test_{{name}}.c src/{{name}}.c src/{{name}}.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ tests/test_{{name}}.c src/{{name}}.c

test: $(BUILD)/test_{{name}}
	./$(BUILD)/test_{{name}}

.PHONY: all test

# No clean target on purpose: nothing in this kit deletes files.
# Remove the build/ folder by hand when you want to rebuild from scratch.
