// Renders Markdown strings with the browser's renderer (static/js/markdown.js) and
// prints the HTML as a JSON array. tests/test_markdown_math.py checks the output.
//
//   node tests/markdown_math.mjs <markdown.js> <cases.json>
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [modulePath, casesPath] = process.argv.slice(2);
const { renderMarkdown } = await import(pathToFileURL(modulePath).href);
const cases = JSON.parse(readFileSync(casesPath, "utf8"));
process.stdout.write(JSON.stringify(cases.map((text) => renderMarkdown(text))));
