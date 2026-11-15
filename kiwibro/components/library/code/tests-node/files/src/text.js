import { readFile } from "node:fs/promises";

export function wordCount(text) {
  return text.split(/\s+/).filter(Boolean).length;
}

export function truncate(text, limit) {
  if (limit < 1) throw new RangeError(`truncate needs a limit of at least 1, got ${limit}`);
  if (text.length <= limit) return text;
  return `${text.slice(0, limit - 1).trimEnd()}…`;
}

// Async on purpose: the test file shows how to await a promise and a rejection.
export async function readLines(path) {
  const text = await readFile(path, "utf8");
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "");
}
