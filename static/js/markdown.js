/**
 * A deliberately small, escape-first Markdown renderer for chat replies.
 *
 * Safety model: every character of user/model text is HTML-escaped before it is
 * placed in the output. The only markup in the result is generated here, from a
 * fixed whitelist of tags/attributes. Links are restricted to http(s) and open
 * in a new, isolated tab. Raw HTML and remote images are never rendered.
 *
 * Code blocks are highlighted from escaped slices only, diff lines get classes,
 * and csv/tsv fences become tables. Live HTML and SVG previews are not made here:
 * richblocks.js adds them after a final render, as a sandboxed iframe (scripts
 * only, opaque origin, CSP that blocks all network access) or as an <img> data
 * URI for SVG, which cannot run script.
 */

const ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

const SAFE_SCHEME = /^https?:\/\//i;

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ESCAPES[char]);
}

function safeHref(raw) {
  const url = String(raw || "").trim();
  if (!url || !SAFE_SCHEME.test(url)) return null;
  return url;
}

const INLINE = new RegExp(
  [
    "(`+)([\\s\\S]*?)\\1", // 1,2 inline code
    "\\*\\*([\\s\\S]+?)\\*\\*", // 3 strong
    "__([\\s\\S]+?)__", // 4 strong
    "~~([\\s\\S]+?)~~", // 5 strike
    "\\[([^\\]]+)\\]\\(([^()\\s]+)\\)", // 6,7 link label, href
    "\\*([^*\\n]+?)\\*", // 8 emphasis
    "_([^_\\n]+?)_", // 9 emphasis
    "(https?:\\/\\/[^\\s<>()\"']+)", // 10 bare url
  ].join("|"),
  "g",
);

function linkHtml(href, label) {
  const safe = safeHref(href);
  if (!safe) return null;
  const target = escapeHtml(safe);
  return `<a href="${target}" target="_blank" rel="noopener noreferrer">${label}</a>`;
}

function renderInline(source, depth = 0) {
  const text = String(source ?? "");
  if (depth > 4) return escapeHtml(text);

  let out = "";
  let last = 0;
  let match;
  INLINE.lastIndex = 0;

  while ((match = INLINE.exec(text)) !== null) {
    if (match.index < last) continue;
    out += escapeHtml(text.slice(last, match.index));
    last = INLINE.lastIndex;

    if (match[2] !== undefined) {
      out += `<code>${escapeHtml(match[2])}</code>`;
    } else if (match[3] !== undefined || match[4] !== undefined) {
      out += `<strong>${renderInline(match[3] ?? match[4], depth + 1)}</strong>`;
    } else if (match[5] !== undefined) {
      out += `<del>${renderInline(match[5], depth + 1)}</del>`;
    } else if (match[6] !== undefined) {
      const link = linkHtml(match[7], renderInline(match[6], depth + 1));
      out += link ?? escapeHtml(match[0]);
    } else if (match[8] !== undefined || match[9] !== undefined) {
      out += `<em>${renderInline(match[8] ?? match[9], depth + 1)}</em>`;
    } else if (match[10] !== undefined) {
      const raw = match[10];
      const trimmed = raw.replace(/[.,;:!?]+$/, "");
      const trailing = raw.slice(trimmed.length);
      const link = linkHtml(trimmed, escapeHtml(trimmed));
      out += (link ?? escapeHtml(trimmed)) + escapeHtml(trailing);
    } else {
      out += escapeHtml(match[0]);
    }
  }

  out += escapeHtml(text.slice(last));
  return out;
}

const HIGHLIGHT_LIMIT = 20000;
const CSV_ROW_LIMIT = 200;

// Shared rule pieces. Sources use only non-capturing groups, because the
// combined per-language regex relies on group N+1 being rule N.
const DQ_STRING = /"(?:\\.|[^"\\\n])*"/;
const SQ_STRING = /'(?:\\.|[^'\\\n])*'/;
const BT_STRING = /`(?:\\[\s\S]|[^`\\])*`/;
const C_LINE = /\/\/[^\n]*/;
const C_BLOCK = /\/\*[\s\S]*?(?:\*\/|$)/;
const HASH_LINE = /#[^\n]*/;
const NUMBER = /\b(?:0[xX][\da-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/;
const CALL = /\b[A-Za-z_]\w*(?=\s*\()/;
const TYPE_NAME = /\b[A-Z]\w*\b/;
const OPERATOR = /=>|===|!==|==|!=|<=|>=|&&|\|\||\?\?|\.\.\.|->|::/;
const words = (list) => new RegExp(`\\b(?:${list.trim().split(/\s+/).join("|")})\\b`);

// Token classes: k keyword, s string, c comment, n number, f function or
// attribute, t type or tag, p operator. Language names map to one entry.
const HIGHLIGHTERS = new Map();

function defineLanguage(names, rules, flags = "g") {
  const pattern = new RegExp(rules.map(([, source]) => `(${source.source})`).join("|"), flags);
  const entry = { classes: rules.map(([cls]) => cls), pattern };
  for (const name of names) HIGHLIGHTERS.set(name, entry);
}

const JS_WORDS = words(`
  async await break case catch class const continue default delete do else enum export
  extends false finally for from function if implements import in instanceof interface
  let new null of private protected public readonly return static super switch this throw
  true try type typeof undefined var void while yield`);
const GO_WORDS = words(`
  break case chan const continue default defer else fallthrough for func go goto if import
  interface map package range return select struct switch type var true false nil`);
const GO_TYPES = words(`
  any bool byte complex64 complex128 error float32 float64 int int8 int16 int32 int64 rune
  string uint uint8 uint16 uint32 uint64 uintptr`);
const RUST_WORDS = words(`
  as async await break const continue crate dyn else enum extern false fn for if impl in let
  loop match mod move mut pub ref return self Self static struct super trait true type unsafe
  use where while`);
const JAVA_WORDS = words(`
  abstract assert boolean break byte case catch char class const continue default do double
  else enum extends false final finally float for goto if implements import instanceof int
  interface long native new null package private protected public return short static strictfp
  super switch synchronized this throw throws transient true try var void volatile while yield`);
const C_WORDS = words(`
  auto bool break case char class const constexpr continue default delete do double else enum
  explicit extern false float for friend goto if inline int long mutable namespace new nullptr
  noexcept operator private protected public register return short signed sizeof static struct
  switch template this throw true try typedef typename union unsigned using virtual void
  volatile while`);
const SQL_WORDS = words(`
  select from where and or not null is in as on join left right inner outer full group by
  order having limit offset insert into values update set delete create table drop alter index
  primary key foreign references distinct union all case when then else end with returning
  exists between like true false default`);
const PY_WORDS = words(`
  and as assert async await break class continue def del elif else except False finally for
  from global if import in is lambda None nonlocal not or pass raise return True try while
  with yield`);
const SH_WORDS = words(`
  if then else elif fi for while until do done case esac in function return local export`);
const SH_BUILTINS = words("echo cd source printf read set unset alias test exit");

defineLanguage(["js", "javascript", "jsx", "ts", "typescript", "tsx"], [
  ["c", C_BLOCK],
  ["c", C_LINE],
  ["s", DQ_STRING],
  ["s", SQ_STRING],
  ["s", BT_STRING],
  ["k", JS_WORDS],
  ["n", NUMBER],
  ["f", CALL],
  ["t", TYPE_NAME],
  ["p", OPERATOR],
]);
defineLanguage(["json"], [
  ["f", /"(?:\\.|[^"\\\n])*"(?=\s*:)/],
  ["s", DQ_STRING],
  ["k", /\b(?:true|false|null)\b/],
  ["n", NUMBER],
]);
defineLanguage(["python", "py"], [
  ["c", HASH_LINE],
  ["s", /"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)/],
  ["s", DQ_STRING],
  ["s", SQ_STRING],
  ["k", PY_WORDS],
  ["n", NUMBER],
  ["f", CALL],
  ["p", OPERATOR],
]);
defineLanguage(["bash", "sh", "shell", "zsh", "console"], [
  ["c", HASH_LINE],
  ["s", DQ_STRING],
  ["s", SQ_STRING],
  ["k", SH_WORDS],
  ["f", SH_BUILTINS],
]);
defineLanguage(
  ["html", "xml", "svg"],
  [
    ["c", /<!--[\s\S]*?(?:-->|$)/],
    ["k", /<!doctype\b/],
    ["s", DQ_STRING],
    ["s", SQ_STRING],
    ["t", /<\/?[a-z][\w:-]*/],
    ["f", /(?<=\s)[\w:-]+(?=\s*=)/],
  ],
  "gi",
);
defineLanguage(["css"], [
  ["c", C_BLOCK],
  ["s", DQ_STRING],
  ["s", SQ_STRING],
  ["k", /@[\w-]+/],
  ["n", /#[\da-fA-F]{3,8}\b|\b\d+(?:\.\d+)?(?:%|[a-zA-Z]+)?/],
  ["f", /[\w-]+(?=\s*:)/],
]);
defineLanguage(
  ["sql"],
  [
    ["c", /--[^\n]*/],
    ["c", C_BLOCK],
    ["s", SQ_STRING],
    ["s", DQ_STRING],
    ["k", SQL_WORDS],
    ["n", NUMBER],
    ["f", CALL],
    ["p", OPERATOR],
  ],
  "gi",
);
defineLanguage(["go"], [
  ["c", C_LINE],
  ["c", C_BLOCK],
  ["s", DQ_STRING],
  ["s", SQ_STRING],
  ["s", BT_STRING],
  ["k", GO_WORDS],
  ["t", GO_TYPES],
  ["n", NUMBER],
  ["f", CALL],
  ["t", TYPE_NAME],
]);
defineLanguage(["rust", "rs"], [
  ["c", C_LINE],
  ["c", C_BLOCK],
  ["s", DQ_STRING],
  ["s", SQ_STRING],
  ["k", RUST_WORDS],
  ["n", NUMBER],
  ["f", /\b[a-z_]\w*!(?!=)/],
  ["f", CALL],
  ["t", TYPE_NAME],
  ["p", OPERATOR],
]);
defineLanguage(["java"], [
  ["c", C_LINE],
  ["c", C_BLOCK],
  ["s", DQ_STRING],
  ["s", SQ_STRING],
  ["k", JAVA_WORDS],
  ["n", NUMBER],
  ["f", CALL],
  ["t", TYPE_NAME],
  ["p", OPERATOR],
]);
defineLanguage(["c", "cpp", "c++", "h", "hpp", "cc", "cxx"], [
  ["c", C_LINE],
  ["c", C_BLOCK],
  ["k", /#\s*(?:include|define|ifdef|ifndef|if|elif|else|endif|pragma|undef)\b/],
  ["s", DQ_STRING],
  ["s", SQ_STRING],
  ["k", C_WORDS],
  ["n", NUMBER],
  ["f", CALL],
  ["t", TYPE_NAME],
  ["p", OPERATOR],
]);
defineLanguage(
  ["yaml", "yml"],
  [
    ["c", HASH_LINE],
    ["s", DQ_STRING],
    ["s", SQ_STRING],
    ["f", /^[ \t]*(?:-[ \t]+)?[\w.-]+(?=[ \t]*:(?:[ \t]|$))/],
    ["k", /\b(?:true|false|null|yes|no|on|off)\b/],
    ["n", NUMBER],
  ],
  "gm",
);
defineLanguage(
  ["toml", "ini"],
  [
    ["c", /[#;][^\n]*/],
    ["t", /^[ \t]*\[\[?[^\]\n]+\]\]?/],
    ["s", DQ_STRING],
    ["s", SQ_STRING],
    ["f", /^[ \t]*[\w.-]+(?=[ \t]*=)/],
    ["k", /\b(?:true|false)\b/],
    ["n", NUMBER],
  ],
  "gm",
);

/** Scans with the language's combined regex; only matched slices get spans. */
function tokenize(code, entry) {
  const { classes, pattern } = entry;
  let out = "";
  let last = 0;
  let match;
  pattern.lastIndex = 0;
  while ((match = pattern.exec(code)) !== null) {
    if (!match[0]) {
      pattern.lastIndex += 1;
      continue;
    }
    const index = classes.findIndex((_, k) => match[k + 1] !== undefined);
    const text = escapeHtml(match[0]);
    out += escapeHtml(code.slice(last, match.index));
    out += index < 0 ? text : `<span class="tok-${classes[index]}">${text}</span>`;
    last = pattern.lastIndex;
  }
  return out + escapeHtml(code.slice(last));
}

function diffClass(line) {
  if (/^(?:diff |index |--- |\+\+\+ )/.test(line)) return "line-meta";
  if (line.startsWith("@@")) return "line-hunk";
  if (line.startsWith("+")) return "line-add";
  if (line.startsWith("-")) return "line-del";
  return "";
}

function diffHtml(code) {
  return code
    .split("\n")
    .map((line) => {
      const cls = diffClass(line);
      const text = escapeHtml(line);
      return cls ? `<span class="${cls}">${text}</span>` : text;
    })
    .join("\n");
}

/** Escaped, token-wrapped code for one fence. Unknown languages stay plain. */
function highlight(code, name) {
  const key = name.toLowerCase();
  if (code.length > HIGHLIGHT_LIMIT) return escapeHtml(code);
  if (key === "diff" || key === "patch") return diffHtml(code);
  const entry = HIGHLIGHTERS.get(key);
  return entry ? tokenize(code, entry) : escapeHtml(code);
}

function codeHead(label) {
  return (
    '<div class="code-head">' +
    `<span class="code-lang">${label}</span>` +
    '<button type="button" class="code-copy" aria-label="Copy code" title="Copy code">' +
    '<svg class="icon" aria-hidden="true"><use href="#i-copy"></use></svg>' +
    "<span>Copy</span>" +
    "</button>" +
    "</div>"
  );
}

function codeBlock(code, language) {
  const name = language ? language.slice(0, 24) : "";
  const label = name ? escapeHtml(name) : "code";
  const dataLang = name ? ` data-lang="${escapeHtml(name.toLowerCase())}"` : "";
  return (
    `<div class="code-block"${dataLang}>` +
    codeHead(label) +
    `<pre><code>${highlight(code, name)}</code></pre>` +
    "</div>"
  );
}

/**
 * Splits delimited text into rows of cells. Quotes group a field, and a doubled
 * quote inside one is a literal quote. Blank rows are dropped.
 */
function parseDelimited(text, delimiter) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let k = 0; k < text.length; k += 1) {
    const char = text[k];
    if (quoted) {
      if (char !== '"') {
        cell += char;
      } else if (text[k + 1] === '"') {
        cell += '"';
        k += 1;
      } else {
        quoted = false;
      }
    } else if (char === '"' && cell === "") {
      quoted = true;
    } else if (char === delimiter) {
      row.push(cell);
      cell = "";
    } else if (char === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  row.push(cell);
  rows.push(row);
  return rows.filter((cells) => cells.some((value) => value.trim() !== ""));
}

/** A csv or tsv fence as a table, with the raw source in a collapsed <details>. */
function csvBlock(code, key) {
  const rows = parseDelimited(code, key === "tsv" ? "\t" : ",");
  if (!rows.length) return null;
  const [header, ...body] = rows;
  const width = rows.reduce((max, row) => Math.max(max, row.length), 0);
  const cells = (tag, row) =>
    Array.from({ length: width }, (_, k) => `<${tag}>${escapeHtml(row[k] ?? "")}</${tag}>`).join("");
  const shown = body.slice(0, CSV_ROW_LIMIT);
  const extra = body.length - shown.length;
  const rowsHtml = shown.map((row) => `<tr>${cells("td", row)}</tr>`).join("");
  const note =
    extra > 0 ? `<div class="csv-note">${extra} more ${extra === 1 ? "row" : "rows"} not shown</div>` : "";
  return (
    `<div class="code-block csv-block" data-lang="${key}">` +
    codeHead(key) +
    '<div class="table-wrap"><table>' +
    `<thead><tr>${cells("th", header)}</tr></thead>` +
    `<tbody>${rowsHtml}</tbody>` +
    "</table></div>" +
    note +
    '<details class="csv-source"><summary>Source</summary>' +
    `<pre><code>${escapeHtml(code)}</code></pre>` +
    "</details>" +
    "</div>"
  );
}

function isJson(text) {
  try {
    JSON.parse(text);
    return true;
  } catch {
    return false;
  }
}

/** Language for a fence with no name: only obvious HTML, SVG and JSON. */
function guessLanguage(code) {
  const head = code.trimStart().slice(0, 16).toLowerCase();
  if (head.startsWith("<!doctype html") || head.startsWith("<html")) return "html";
  if (head.startsWith("<svg")) return "svg";
  if (/^[[{]/.test(head) && code.length <= HIGHLIGHT_LIMIT && isJson(code)) return "json";
  return "";
}

function fencedBlock(code, language) {
  const name = language || guessLanguage(code);
  const key = name.toLowerCase();
  if (key === "csv" || key === "tsv") return csvBlock(code, key) ?? codeBlock(code, name);
  return codeBlock(code, name);
}

const FENCE = /^(```+|~~~+)\s*([\w+#.-]*)\s*$/;
const FENCE_CLOSE = /^(```+|~~~+)\s*$/;
const HEADING = /^(#{1,6})\s+(.+)$/;
const RULE = /^(-{3,}|\*{3,}|_{3,})$/;
const QUOTE = /^\s*>\s?/;
// A list item on a right-trimmed raw line: leading indent, marker, text.
const ITEM = /^(\s*)([-*+]|\d{1,2}[.)])\s+(.+)$/;
// Task marker at the start of an item's text: "[ ]", "[x]" or "[X]".
const TASK = /^\[([ xX])\](?:\s+(.*))?$/;
const TABLE_CELL_RULE = /^:?-{3,}:?$/;
const MAX_LIST_DEPTH = 3;
const ALIGN_STYLE = {
  center: ' style="text-align:center"',
  right: ' style="text-align:right"',
};

/** Splits a GFM table row into trimmed cells. `\|` is a literal pipe. */
function splitRow(line) {
  let row = line.trim();
  if (row.startsWith("|")) row = row.slice(1);
  if (row.endsWith("|") && !row.endsWith("\\|")) row = row.slice(0, -1);

  const cells = [];
  let cell = "";
  for (let k = 0; k < row.length; k += 1) {
    const char = row[k];
    if (char === "\\" && row[k + 1] === "|") {
      cell += "|";
      k += 1;
    } else if (char === "|") {
      cells.push(cell.trim());
      cell = "";
    } else {
      cell += char;
    }
  }
  cells.push(cell.trim());
  return cells;
}

/** Returns per-column alignments ("left" | "center" | "right") or null. */
function parseAlignment(line) {
  if (!line.includes("|")) return null;
  const cells = splitRow(line);
  if (!cells.length || !cells.every((cell) => TABLE_CELL_RULE.test(cell))) return null;
  return cells.map((cell) => {
    const start = cell.startsWith(":");
    const end = cell.endsWith(":");
    if (start && end) return "center";
    if (end) return "right";
    return "left";
  });
}

function renderTable(headerCells, alignments, bodyRows) {
  const cell = (tag, content, align) =>
    `<${tag}${ALIGN_STYLE[align] ?? ""}>${renderInline(content)}</${tag}>`;
  const head = headerCells.map((content, k) => cell("th", content, alignments[k])).join("");
  const body = bodyRows
    .map((cells) => {
      let row = "";
      // Missing cells are padded with empty ones; extra cells are dropped.
      for (let k = 0; k < alignments.length; k += 1) {
        row += cell("td", cells[k] ?? "", alignments[k]);
      }
      return `<tr>${row}</tr>`;
    })
    .join("");
  return (
    '<div class="table-wrap"><table>' +
    `<thead><tr>${head}</tr></thead>` +
    `<tbody>${body}</tbody>` +
    "</table></div>"
  );
}

/**
 * Parses a GFM table starting at `start` (header row, separator row, body rows).
 * Returns null when there is no valid separator row yet, so a lone header line
 * (for example while a reply is still streaming) stays a normal paragraph.
 */
function parseTable(lines, start) {
  if (!lines[start].includes("|")) return null;
  const alignments = parseAlignment(lines[start + 1] ?? "");
  if (!alignments) return null;
  const headerCells = splitRow(lines[start]);
  if (headerCells.length !== alignments.length) return null;

  const bodyRows = [];
  let i = start + 2;
  while (i < lines.length) {
    const raw = lines[i];
    const trimmed = raw.trim();
    if (!trimmed || FENCE.test(trimmed) || HEADING.test(trimmed) || QUOTE.test(raw)) break;
    bodyRows.push(splitRow(raw));
    i += 1;
  }
  return { html: renderTable(headerCells, alignments, bodyRows), next: i };
}

function indentWidth(whitespace) {
  let width = 0;
  for (const char of whitespace) width += char === "\t" ? 4 : 1;
  return width;
}

function parseItem(line) {
  const match = line.trimEnd().match(ITEM);
  if (!match) return null;
  return {
    indent: indentWidth(match[1]),
    ordered: /\d/.test(match[2]),
    text: match[3],
  };
}

function parseTask(text) {
  const match = text.match(TASK);
  if (!match) return { checked: null, text };
  return { checked: match[1] !== " ", text: match[2] ?? "" };
}

function openList(container, ordered, level) {
  const node = { tag: ordered ? "ol" : "ul", items: [] };
  container.push(node);
  return { level, node, container };
}

/**
 * Collects a run of list items starting at `start` and builds a tree of
 * nested lists. Two spaces (or a tab) of extra indentation nest one level;
 * nesting stops at MAX_LIST_DEPTH and deeper items join the innermost list.
 * Mixed ordered/unordered siblings start a new list, as before.
 */
function collectList(lines, start) {
  const base = Math.floor(parseItem(lines[start]).indent / 2);
  const roots = [];
  const open = [];
  let i = start;
  let item = parseItem(lines[i]);

  while (item) {
    i += 1;
    const level = Math.max(0, Math.floor(item.indent / 2) - base);
    while (open.length && open[open.length - 1].level > level) open.pop();

    let frame = open[open.length - 1];
    if (!frame) {
      frame = openList(roots, item.ordered, level);
      open.push(frame);
    } else if (frame.level < level && open.length < MAX_LIST_DEPTH) {
      const parent = frame.node.items[frame.node.items.length - 1];
      frame = openList(parent.subs, item.ordered, level);
      open.push(frame);
    } else if (frame.node.tag !== (item.ordered ? "ol" : "ul")) {
      open.pop();
      frame = openList(frame.container, item.ordered, frame.level);
      open.push(frame);
    }

    frame.node.items.push({ ...parseTask(item.text), subs: [] });
    item = i < lines.length ? parseItem(lines[i]) : null;
  }

  return { html: roots.map((node) => renderList(node)).join(""), next: i };
}

function renderList(node) {
  const items = node.items.map((item) => {
    let cls = "";
    let box = "";
    if (item.checked !== null) {
      cls = ' class="task"';
      box = item.checked
        ? '<input type="checkbox" disabled checked> '
        : '<input type="checkbox" disabled> ';
    }
    const nested = item.subs.map((sub) => renderList(sub)).join("");
    return `<li${cls}>${box}${renderInline(item.text)}${nested}</li>`;
  });
  return `<${node.tag}>${items.join("")}</${node.tag}>`;
}

export function renderMarkdown(source) {
  const text = String(source ?? "").replace(/\r\n?/g, "\n");
  const lines = text.split("\n");
  const html = [];
  let paragraph = [];
  let i = 0;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${paragraph.map((line) => renderInline(line)).join("<br>")}</p>`);
    paragraph = [];
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      i += 1;
      continue;
    }

    const fence = trimmed.match(FENCE);
    if (fence) {
      flushParagraph();
      const marker = fence[1][0];
      const language = fence[2] || "";
      const body = [];
      i += 1;
      while (i < lines.length) {
        const candidate = lines[i];
        const close = candidate.trim().match(FENCE_CLOSE);
        if (close && close[1][0] === marker) {
          i += 1;
          break;
        }
        body.push(candidate);
        i += 1;
      }
      html.push(fencedBlock(body.join("\n"), language));
      continue;
    }

    const heading = trimmed.match(HEADING);
    if (heading) {
      flushParagraph();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (RULE.test(trimmed)) {
      flushParagraph();
      html.push("<hr>");
      i += 1;
      continue;
    }

    if (QUOTE.test(line)) {
      flushParagraph();
      const quote = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        quote.push(lines[i].replace(QUOTE, ""));
        i += 1;
      }
      html.push(`<blockquote>${quote.map((entry) => renderInline(entry)).join("<br>")}</blockquote>`);
      continue;
    }

    const table = parseTable(lines, i);
    if (table) {
      flushParagraph();
      html.push(table.html);
      i = table.next;
      continue;
    }

    if (parseItem(line)) {
      flushParagraph();
      const list = collectList(lines, i);
      html.push(list.html);
      i = list.next;
      continue;
    }

    paragraph.push(trimmed);
    i += 1;
  }

  flushParagraph();
  return html.join("");
}
