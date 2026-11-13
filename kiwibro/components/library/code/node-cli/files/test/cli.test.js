import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { collectFiles, formatSize, run } from "../src/cli.js";

// KIWIBRO: replace these tests with tests for your own options and output
function sink() {
  const chunks = [];
  return { write: (chunk) => chunks.push(chunk), text: () => chunks.join("") };
}

test("formatSize uses binary units", () => {
  assert.equal(formatSize(512), "512 B");
  assert.equal(formatSize(2048), "2.0 KB");
  assert.equal(formatSize(3 * 1024 * 1024), "3.0 MB");
});

test("collectFiles lists files biggest first and skips hidden ones", async () => {
  // The folder stays in the system temp directory; this tool never removes files.
  const root = await mkdtemp(join(tmpdir(), "{{name}}-"));
  await mkdir(join(root, "sub"));
  await writeFile(join(root, "small.txt"), "ab");
  await writeFile(join(root, ".hidden"), "hidden");
  await writeFile(join(root, "sub", "big.txt"), "abcdef");

  const files = await collectFiles(root);

  assert.deepEqual(
    files.map((file) => file.size),
    [6, 2],
  );
  assert.ok(files[0].path.endsWith(join("sub", "big.txt")));
});

test("run prints the largest file and exits 0", async () => {
  const root = await mkdtemp(join(tmpdir(), "{{name}}-"));
  await writeFile(join(root, "only.txt"), "12345");
  const out = sink();

  const code = await run(["--limit", "1", root], out, sink());

  assert.equal(code, 0);
  assert.match(out.text(), /5 B .*only\.txt\n$/);
});

test("run rejects a bad --limit with exit code 2", async () => {
  const err = sink();

  const code = await run(["--limit", "zero", "."], sink(), err);

  assert.equal(code, 2);
  assert.match(err.text(), /positive whole number/);
});

test("run reports an unreadable directory with exit code 1", async () => {
  const err = sink();
  const missing = join(tmpdir(), "{{name}}-missing-directory");

  const code = await run([missing], sink(), err);

  assert.equal(code, 1);
  assert.match(err.text(), /cannot read/);
});
