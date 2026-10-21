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
const HEADING = /^(#{1,4})\s+(.+)$/;
const RULE = /^(-{3,}|\*{3,}|_{3,})$/;
const QUOTE = /^\s*>\s?/;
const UL = /^\s*[-*+]\s+(.+)$/;
const OL = /^\s*\d{1,2}[.)]\s+(.+)$/;

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

    const unordered = trimmed.match(UL);
    const ordered = trimmed.match(OL);
    if (unordered || ordered) {
      flushParagraph();
      const isOrdered = Boolean(ordered);
      const items = [];
      while (i < lines.length) {
        const candidate = lines[i].trim();
        const next = isOrdered ? candidate.match(OL) : candidate.match(UL);
        if (!next) break;
        items.push(next[1]);
        i += 1;
      }
      const tag = isOrdered ? "ol" : "ul";
      html.push(`<${tag}>${items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${tag}>`);
      continue;
    }

    paragraph.push(trimmed);
    i += 1;
  }

  flushParagraph();
  return html.join("");
}
