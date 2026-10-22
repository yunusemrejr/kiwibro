/**
 * Live previews for model-written HTML and SVG code blocks.
 *
 * enhanceRendered() is called on final renders only (chat.js), never while a
 * reply streams, so frames are not rebuilt on every delta. Nodes are built with
 * DOM calls, never innerHTML, and the raw code stays in the block's <pre> so Copy
 * works in both views.
 *
 * HTML: a sandboxed iframe (opaque origin, no forms, popups or top navigation)
 * whose document starts with a CSP that blocks all network requests. Scripts are
 * paused until the user presses Run: a page can still navigate its own frame, and
 * a prompt-injected page could ask the model for one that carries chat text out
 * in a URL. Refresh tags are removed and links get no target to open in.
 * SVG: an <img> with a data: URI, which cannot run script.
 */

const HTML_CSP =
  "default-src 'none'; img-src data: blob:; media-src data: blob:; style-src 'unsafe-inline'; " +
  "script-src 'unsafe-inline'; font-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'";
const COMPLETE_HTML_BYTES = 8192;
const SCRIPTED = /<script\b|\son[a-z]+\s*=|javascript:/i;
const REFRESH = /<meta\b[^>]*http-equiv\s*=\s*["']?refresh[^>]*>/gi;
const VOID_TAGS = new Set(
  "area base br col embed hr img input link meta source track wbr".split(" "),
);

function make(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  node.append(...children);
  return node;
}

function insertAt(source, index, text) {
  return source.slice(0, index) + text + source.slice(index);
}

function previewKind(lang, source) {
  if (!source.trim()) return "";
  if (lang === "svg") return "svg";
  if (lang === "html" || /^\s*<(?:!doctype html|html[\s>])/i.test(source)) return "html";
  return "";
}

/** Rough check: opening and closing tags pair up (void and self-closing tags skipped). */
function looksBalanced(source) {
  let open = 0;
  let close = 0;
  for (const [whole, slash, name] of source.matchAll(/<(\/?)([a-z][a-z0-9-]*)(?:\s[^>]*)?>/gi)) {
    if (slash) close += 1;
    else if (!VOID_TAGS.has(name.toLowerCase()) && !whole.endsWith("/>")) open += 1;
  }
  return open === close;
}

/** Complete enough to show by default. Anything else opens on the code view. */
function isComplete(source, kind) {
  if (kind === "svg") return /<\/svg>/i.test(source);
  return /<\/(?:html|body)>/i.test(source) || (source.length < COMPLETE_HTML_BYTES && looksBalanced(source));
}

/** Puts the CSP meta first in <head>, creating <head> when the document has none. */
function withCsp(source) {
  // <base target> sends link clicks to a new window, which the sandbox refuses.
  const meta = `<meta http-equiv="Content-Security-Policy" content="${HTML_CSP}"><base target="_blank">`;
  source = source.replace(REFRESH, "");
  const head = /<head(?:\s[^>]*)?>/i.exec(source);
  if (head) return insertAt(source, head.index + head[0].length, meta);
  const root = /<html(?:\s[^>]*)?>/i.exec(source);
  if (root) return insertAt(source, root.index + root[0].length, `<head>${meta}</head>`);
  const doctype = /^\s*<!doctype[^>]*>/i.exec(source);
  return insertAt(source, doctype ? doctype[0].length : 0, `<head>${meta}</head>`);
}

function htmlFrame(source, scripts) {
  return make("iframe", {
    class: "rich-frame",
    title: "HTML preview",
    sandbox: scripts ? "allow-scripts" : "",
    referrerpolicy: "no-referrer",
    loading: "lazy",
    srcdoc: withCsp(source),
  });
}

function htmlPanel(source) {
  const expand = make(
    "button",
    { type: "button", class: "rich-expand", "aria-expanded": "false" },
    "Expand",
  );
  const bar = make("div", { class: "rich-bar" });
  if (SCRIPTED.test(source)) {
    bar.append(
      make("span", { class: "rich-note" }, "Scripts paused"),
      make("button", { type: "button", class: "rich-run" }, "Run scripts"),
    );
  }
  bar.append(expand);
  return make("div", { class: "rich-preview" }, htmlFrame(source, false), bar);
}

/** Reloads the preview with scripts allowed (still sandboxed and offline). */
function runScripts(button) {
  const panel = button.closest(".rich-preview");
  const block = button.closest(".code-block");
  const frame = panel?.querySelector(".rich-frame");
  const source = block?.querySelector("pre code")?.textContent;
  if (!frame || source == null) return;
  const running = htmlFrame(source, true);
  running.className = frame.className;
  frame.replaceWith(running);
  button.disabled = true;
  button.textContent = "Running";
  panel.querySelector(".rich-note")?.remove();
}

function svgPanel(source) {
  const image = make("img", {
    class: "rich-image",
    alt: "SVG preview",
    src: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(source)}`,
  });
  return make("div", { class: "rich-preview rich-svg" }, image);
}

function viewTabs(previewFirst) {
  const tab = (view, label, pressed) =>
    make(
      "button",
      { type: "button", class: "code-tab", "data-view": view, "aria-pressed": String(pressed) },
      label,
    );
  return make(
    "div",
    { class: "code-tabs", role: "group", "aria-label": "Preview or code" },
    tab("preview", "Preview", previewFirst),
    tab("code", "Code", !previewFirst),
  );
}

function enhanceBlock(block) {
  const pre = block.querySelector(":scope > pre");
  const head = block.querySelector(":scope > .code-head");
  const label = head?.querySelector(".code-lang");
  const source = block.querySelector("pre code")?.textContent || "";
  const kind = previewKind((block.dataset.lang || "").toLowerCase(), source);
  if (!kind || !pre || !label) return;

  const complete = isComplete(source, kind);
  const preview = kind === "svg" ? svgPanel(source) : htmlPanel(source);
  preview.hidden = !complete;
  pre.hidden = complete;
  pre.before(preview);
  label.after(viewTabs(complete));
}

/** Adds previews to the code blocks inside `container`. Safe to call twice. */
export function enhanceRendered(container) {
  const blocks = container?.querySelectorAll?.(".code-block:not([data-rich])") ?? [];
  for (const block of blocks) {
    block.dataset.rich = "1";
    try {
      enhanceBlock(block);
    } catch {
      // A failed preview leaves the plain, copyable code block in place.
    }
  }
}

function showView(block, view) {
  const preview = block.querySelector(":scope > .rich-preview");
  const pre = block.querySelector(":scope > pre");
  const previewing = view === "preview";
  if (preview) preview.hidden = !previewing;
  if (pre) pre.hidden = previewing;
  for (const tab of block.querySelectorAll(":scope > .code-head .code-tab")) {
    tab.setAttribute("aria-pressed", String(tab.dataset.view === view));
  }
}

function toggleExpand(button) {
  const frame = button.closest(".rich-preview")?.querySelector(".rich-frame");
  if (!frame) return;
  const expanded = !frame.classList.contains("is-expanded");
  frame.classList.toggle("is-expanded", expanded);
  button.setAttribute("aria-expanded", String(expanded));
  button.textContent = expanded ? "Collapse" : "Expand";
}

/** Handles preview tab and expand clicks. Returns true when the event was used. */
export function handleRichClick(event) {
  const target = event.target instanceof Element ? event.target : null;
  const tab = target?.closest(".code-tab");
  if (tab) {
    const block = tab.closest(".code-block");
    if (block) showView(block, tab.dataset.view);
    return true;
  }
  const expand = target?.closest(".rich-expand");
  if (expand) {
    toggleExpand(expand);
    return true;
  }
  const run = target?.closest(".rich-run");
  if (run) {
    runScripts(run);
    return true;
  }
  return false;
}
