import { readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";

const HELP = `usage: {{name}} [--limit N] [--all] DIR

Print the largest files under DIR, biggest first.
  --limit N  how many entries to print (default 10)
  --all      include files and folders whose name starts with a dot
  -h, --help show this text
`;

const UNITS = ["B", "KB", "MB", "GB"];

export function formatSize(bytes) {
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${value} B` : `${value.toFixed(1)} ${UNITS[unit]}`;
}

// KIWIBRO: replace the walk with the input your own tool reads
export async function collectFiles(dir, { hidden = false } = {}) {
  const found = [];
  const pending = [dir];
  while (pending.length > 0) {
    const current = pending.pop();
    let entries;
    try {
      entries = await readdir(current, { withFileTypes: true });
    } catch (error) {
      throw new Error(`cannot read ${current}: ${error.message}`);
    }
    for (const entry of entries) {
      if (!hidden && entry.name.startsWith(".")) continue;
      const full = join(current, entry.name);
      if (entry.isDirectory()) {
        pending.push(full);
      } else if (entry.isFile()) {
        const info = await stat(full);
        found.push({ path: full, size: info.size });
      }
    }
  }
  return found.sort((a, b) => b.size - a.size);
}

// KIWIBRO: define your tool's own options and subcommands here
export async function run(argv, out = process.stdout, err = process.stderr) {
  let parsed;
  try {
    parsed = parseArgs({
      args: argv,
      allowPositionals: true,
      options: {
        limit: { type: "string", default: "10" },
        all: { type: "boolean", default: false },
        help: { type: "boolean", short: "h", default: false },
      },
    });
  } catch (error) {
    err.write(`{{name}}: ${error.message}\n${HELP}`);
    return 2;
  }

  const { values, positionals } = parsed;
  if (values.help) {
    out.write(HELP);
    return 0;
  }
  if (positionals.length !== 1) {
    err.write(HELP);
    return 2;
  }

  const limit = Number(values.limit);
  if (!Number.isInteger(limit) || limit < 1) {
    err.write("{{name}}: --limit must be a positive whole number\n");
    return 2;
  }

  let files;
  try {
    files = await collectFiles(positionals[0], { hidden: values.all });
  } catch (error) {
    err.write(`{{name}}: ${error.message}\n`);
    return 1;
  }

  for (const file of files.slice(0, limit)) {
    out.write(`${formatSize(file.size).padStart(8)}  ${file.path}\n`);
  }
  return 0;
}
