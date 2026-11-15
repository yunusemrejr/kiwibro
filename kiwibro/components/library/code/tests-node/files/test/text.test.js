import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { readLines, truncate, wordCount } from "../src/text.js";

// Table-driven: one row per case, one test per row, so a new case is a new line.
for (const [what, text, wanted] of [
  ["empty text", "", 0],
  ["one word", "hello", 1],
  ["spaces and newlines", " one  two\nthree ", 3],
]) {
  test(`wordCount counts ${what}`, () => {
    assert.equal(wordCount(text), wanted);
  });
}

for (const [what, text, limit, wanted] of [
  ["short text is kept", "hello", 10, "hello"],
  ["long text ends in an ellipsis", "hello world", 8, "hello w…"],
  ["a limit of one keeps only the mark", "hello", 1, "…"],
]) {
  test(`truncate: ${what}`, () => {
    assert.equal(truncate(text, limit), wanted);
  });
}

test("truncate refuses a limit below one", () => {
  assert.throws(() => truncate("hello", 0), RangeError);
});

test("readLines returns the non-empty lines of a file", async () => {
  // The folder stays in the system temp directory; the test writes and reads only.
  const folder = await mkdtemp(join(tmpdir(), "{{name}}-"));
  const file = join(folder, "sample.txt");
  await writeFile(file, "first\n\n  third  \n");

  assert.deepEqual(await readLines(file), ["first", "third"]);
});

test("readLines rejects a file that does not exist", async () => {
  const missing = join(tmpdir(), "{{name}}-no-such-file.txt");

  await assert.rejects(readLines(missing), /ENOENT/);
});
