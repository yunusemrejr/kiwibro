/**
 * A deliberately small, escape-first Markdown renderer for chat replies.
 *
 * Safety model: every character of user/model text is HTML-escaped before it is
 * placed in the output. The only markup in the result is generated here, from a
 * fixed whitelist of tags/attributes. Links are restricted to http(s) and open
 * in a new, isolated tab. Raw HTML and remote images are never rendered.
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

function codeBlock(code, language) {
  const lang = language ? language.slice(0, 24) : "";
  const label = lang ? escapeHtml(lang) : "code";
  return (
    '<div class="code-block">' +
    '<div class="code-head">' +
    `<span class="code-lang">${label}</span>` +
    '<button type="button" class="code-copy" aria-label="Copy code" title="Copy code">' +
    '<svg class="icon" aria-hidden="true"><use href="#i-copy"></use></svg>' +
    "<span>Copy</span>" +
    "</button>" +
    "</div>" +
    `<pre><code>${escapeHtml(code)}</code></pre>` +
    "</div>"
  );
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
      html.push(codeBlock(body.join("\n"), language));
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
