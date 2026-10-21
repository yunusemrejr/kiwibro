/**
 * Shared UI primitives: DOM helpers, icons, popovers, sheets, toasts,
 * relative time and tool metadata. No business logic lives here.
 */

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const SVG_NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "on") {
      for (const [event, handler] of Object.entries(value)) node.addEventListener(event, handler);
    } else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key === "hidden") node.hidden = Boolean(value);
    else if (key === "value") node.value = value;
    else if (key === "disabled") node.disabled = Boolean(value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  append(node, children);
  return node;
}

export function append(node, children) {
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child == null || child === false) continue;
    if (Array.isArray(child)) append(node, child);
    else if (child instanceof Node) node.append(child);
    else node.append(document.createTextNode(String(child)));
  }
}

export function icon(name, className = "") {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", className ? `icon ${className}` : "icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function debounce(fn, wait = 200) {
  let timer = 0;
  const debounced = (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), wait);
  };
  debounced.cancel = () => window.clearTimeout(timer);
  return debounced;
}

/* ---------- Time ---------------------------------------------------------- */

const DAY = 86_400_000;

function asDate(value) {
  if (value == null) return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function startOfDay(date) {
  const copy = new Date(date);
  copy.setHours(0, 0, 0, 0);
  return copy.getTime();
}

export function relativeTime(value) {
  const date = asDate(value);
  if (!date) return "";
  const now = Date.now();
  const diff = now - date.getTime();
  const abs = Math.abs(diff);

  if (abs < 45_000) return "now";
  if (abs < 3_600_000) return `${Math.max(1, Math.round(abs / 60_000))}m`;

  const dayDiff = Math.round((startOfDay(new Date(now)) - startOfDay(date)) / DAY);
  if (dayDiff === 0) return `${Math.round(abs / 3_600_000)}h`;
  if (dayDiff === 1) return "Yesterday";
  if (dayDiff > 1 && dayDiff < 7) {
    return date.toLocaleDateString(undefined, { weekday: "short" });
  }
  const sameYear = date.getFullYear() === new Date(now).getFullYear();
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: sameYear ? undefined : "numeric",
  });
}

export function absoluteTime(value) {
  const date = asDate(value);
  if (!date) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Refresh every element carrying data-ts (used by the 30 s clock ticker). */
export function updateRelativeTimes(root = document) {
  for (const node of root.querySelectorAll("[data-ts]")) {
    const iso = node.getAttribute("data-ts");
    if (!iso) continue;
    node.textContent = relativeTime(iso);
    node.title = absoluteTime(iso);
  }
}

export function formatTokens(count) {
  const value = Number(count) || 0;
  if (value < 1000) return String(value);
  if (value < 100_000) return `${(value / 1000).toFixed(1).replace(/\.0$/, "")}K`;
  return `${Math.round(value / 1000)}K`;
}

/* ---------- Toasts -------------------------------------------------------- */

export function toast(message, { tone = "info", iconName = "check", timeout = 3400 } = {}) {
  const stack = document.getElementById("toast-stack");
  if (!stack) return;
  const node = el("div", { class: "toast", "data-tone": tone, role: "status" }, [
    icon(iconName === "auto" ? (tone === "error" ? "alert" : "check") : iconName),
    el("span", { text: message }),
  ]);
  stack.append(node);
  while (stack.children.length > 3) stack.firstElementChild.remove();

  const dismiss = () => {
    if (!node.isConnected) return;
    node.classList.add("is-leaving");
    node.addEventListener("animationend", () => node.remove(), { once: true });
    window.setTimeout(() => node.remove(), 400);
  };
  node.addEventListener("click", dismiss);
  window.setTimeout(dismiss, timeout);
}

export function announce(message) {
  const region = document.getElementById("live-status");
  if (!region) return;
  region.textContent = "";
  window.setTimeout(() => {
    region.textContent = message;
  }, 30);
}

/* ---------- Clipboard ----------------------------------------------------- */

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      const ok = document.execCommand("copy");
      area.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

/* ---------- Popovers ------------------------------------------------------ */

let activePopover = null;

export function popoverIsOpen() {
  return Boolean(activePopover);
}

export function closePopover({ restoreFocus = true } = {}) {
  if (!activePopover) return;
  const popover = activePopover;
  activePopover = null;
  popover.cleanup(restoreFocus);
}

/**
 * Anchor a single floating panel to an element. Handles outside clicks,
 * Escape, Tab trapping, focus restore and viewport clamping.
 */
export function openPopover({
  anchor,
  content,
  align = "center",
  gap = 6,
  label = "",
  onClose = null,
  autoFocus = true,
}) {
  closePopover({ restoreFocus: false });
  const layer = document.getElementById("popover-layer");
  if (!layer || !anchor) return null;

  const popover = el("div", { class: "popover", role: "dialog", "aria-label": label });
  popover.append(content);
  layer.append(popover);
  layer.style.pointerEvents = "none";

  const anchorRect = anchor.getBoundingClientRect();
  const rect = popover.getBoundingClientRect();
  const margin = 8;
  let left;
  if (align === "start") left = anchorRect.left;
  else if (align === "end") left = anchorRect.right - rect.width;
  else left = anchorRect.left + anchorRect.width / 2 - rect.width / 2;
  left = clamp(left, margin, window.innerWidth - rect.width - margin);

  let top = anchorRect.bottom + gap;
  let origin = "top center";
  if (top + rect.height > window.innerHeight - margin) {
    const above = anchorRect.top - gap - rect.height;
    if (above > margin) {
      top = above;
      origin = "bottom center";
    } else {
      top = Math.max(margin, window.innerHeight - rect.height - margin);
    }
  }

  popover.style.left = `${Math.round(left)}px`;
  popover.style.top = `${Math.round(top)}px`;
  popover.style.setProperty("--pop-origin", origin);
  popover.style.transformOrigin = origin;

  const previouslyFocused = document.activeElement;
  if (autoFocus) {
    window.requestAnimationFrame(() => {
      const target =
        content.querySelector("[data-autofocus]") ||
        popover.querySelector(
          'button:not([disabled]), [href], input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])',
        );
      if (target) target.focus({ preventScroll: true });
      else {
        popover.tabIndex = -1;
        popover.focus({ preventScroll: true });
      }
    });
  }

  const focusables = () =>
    Array.from(
      popover.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])',
      ),
    ).filter((node) => node.getClientRects().length > 0);

  const onPointerDown = (event) => {
    if (popover.contains(event.target) || anchor.contains(event.target)) return;
    closePopover();
  };

  const onKeyDown = (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closePopover();
      return;
    }
    if (event.key !== "Tab") return;
    const items = focusables();
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const onViewportChange = () => closePopover({ restoreFocus: false });

  document.addEventListener("pointerdown", onPointerDown, true);
  document.addEventListener("keydown", onKeyDown, true);
  window.addEventListener("resize", onViewportChange);
  window.addEventListener("scroll", onViewportChange, true);

  activePopover = {
    element: popover,
    cleanup(restoreFocus) {
      popover.remove();
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("resize", onViewportChange);
      window.removeEventListener("scroll", onViewportChange, true);
      if (
        restoreFocus &&
        previouslyFocused instanceof HTMLElement &&
        document.contains(previouslyFocused)
      ) {
        previouslyFocused.focus({ preventScroll: true });
      }
      onClose?.();
    },
  };

  return {
    element: popover,
    close: () => closePopover(),
  };
}

/* ---------- Dialogs ------------------------------------------------------- */

export function wireDialog(dialog) {
  if (!dialog) return;
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) closeSheet(dialog);
  });
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeSheet(dialog);
  });
  for (const button of dialog.querySelectorAll("[data-close-sheet]")) {
    button.addEventListener("click", () => closeSheet(dialog));
  }
}

export function openSheet(dialog) {
  if (!dialog || dialog.open) return;
  dialog.showModal();
  dialog.classList.remove("is-closing");
}

export function closeSheet(dialog) {
  if (!dialog || !dialog.open || dialog.classList.contains("is-closing")) return;
  dialog.classList.add("is-closing");
  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    dialog.classList.remove("is-closing");
    if (dialog.open) dialog.close();
  };
  dialog.addEventListener("animationend", (event) => {
    if (event.target === dialog) finish();
  });
  window.setTimeout(finish, 240);
}

/* ---------- Tool metadata ------------------------------------------------- */

const TOOL_META = {
  web_search: { label: "Web search", icon: "search" },
  fetch_page: { label: "Fetch page", icon: "globe" },
  multi_search: { label: "Multi search", icon: "search" },
  fetch_pages: { label: "Fetch pages", icon: "globe" },
  web_research: { label: "Web research", icon: "search" },
  browse_page: { label: "Browse page", icon: "globe" },
  browser: { label: "Browser", icon: "globe" },
  read_file: { label: "Read file", icon: "file" },
  read_files: { label: "Read files", icon: "file" },
  list_files: { label: "List files", icon: "folder" },
  glob_files: { label: "Find files", icon: "folder" },
  grep_files: { label: "Search files", icon: "search" },
  file_outline: { label: "File outline", icon: "layers" },
  git: { label: "Git", icon: "database" },
  write_file: { label: "Write file", icon: "file-plus" },
  replace_in_file: { label: "Edit file", icon: "pencil" },
  apply_patch: { label: "Patch file", icon: "pencil" },
  file_manage: { label: "Manage files", icon: "folder" },
  run_command: { label: "Run command", icon: "terminal" },
  multi_command: { label: "Run commands", icon: "terminal" },
  terminal_session: { label: "Terminal", icon: "terminal" },
  run_python: { label: "Run Python", icon: "terminal" },
  run_javascript: { label: "Run JavaScript", icon: "terminal" },
  system_info: { label: "System info", icon: "info" },
  open_path: { label: "Open", icon: "external" },
  clipboard: { label: "Clipboard", icon: "copy" },
  notify: { label: "Notify", icon: "message" },
  screenshot: { label: "Screenshot", icon: "image" },
  model_info: { label: "Local models", icon: "sparkles" },
  delegate: { label: "Helper model", icon: "sparkles" },
  describe_image: { label: "Describe image", icon: "image" },
  study_file: { label: "Study file", icon: "file" },
  task_board: { label: "Task board", icon: "check" },
  recall: { label: "Recall", icon: "sparkles" },
  remember: { label: "Remember", icon: "sparkles" },
  calculate: { label: "Calculate", icon: "tool" },
  analyze_table: { label: "Analyze table", icon: "database" },
  doc_search: { label: "Search documents", icon: "search" },
  doc_read: { label: "Read document", icon: "file" },
  doc_add: { label: "Index documents", icon: "layers" },
  doc_list: { label: "Documents", icon: "folder" },
  undo_changes: { label: "Undo changes", icon: "refresh" },
};

export function toolLabel(name) {
  return TOOL_META[name]?.label || String(name || "Tool").replace(/_/g, " ");
}

export function toolIcon(name) {
  return TOOL_META[name]?.icon || "tool";
}

export function toolSummary(name, args) {
  if (!args || typeof args !== "object") return "";
  switch (name) {
    case "web_search":
    case "web_research":
    case "recall":
    case "grep_files":
      return args.query || args.pattern || "";
    case "multi_search":
      return Array.isArray(args.queries) ? args.queries.join(" · ") : "";
    case "fetch_pages":
      return Array.isArray(args.urls) ? `${args.urls.length} pages` : "";
    case "fetch_page":
    case "browse_page":
      return args.url || "";
    case "browser":
      return args.url || args.tab_id || args.action || "";
    case "read_file":
    case "write_file":
    case "replace_in_file":
    case "file_outline":
    case "apply_patch":
    case "file_manage":
    case "describe_image":
    case "study_file":
      return args.path || "";
    case "read_files":
      return Array.isArray(args.paths) ? args.paths.join(", ") : "";
    case "glob_files":
      return args.pattern || "";
    case "git":
      return args.action || "status";
    case "list_files":
      return args.path && args.path !== "." ? args.path : "workspace";
    case "run_command":
    case "run_python":
    case "run_javascript":
      return args.command || (typeof args.code === "string" ? args.code.split("\n")[0] : "") || "";
    case "multi_command":
      return Array.isArray(args.commands) ? args.commands.join(" && ") : "";
    case "terminal_session":
      return args.command || args.session_id || args.action || "";
    case "open_path":
      return args.target || "";
    case "clipboard":
      return args.action === "write" ? args.text || "" : "read clipboard";
    case "notify":
      return args.title || "";
    case "delegate":
      return args.task || "";
    case "task_board":
      return args.title || args.action || "";
    case "remember":
      return args.content || "";
    default:
      return "";
  }
}

export function riskLevel(risk) {
  // Accepts the safety levels (safe…critical) and the legacy low/medium/high.
  const value = String(risk || "medium").toLowerCase();
  if (value === "safe" || value === "low") return "low";
  if (value === "high" || value === "critical") return "high";
  return "medium";
}

export function summarizeArgs(args, fallback = "") {
  if (args == null) return fallback;
  if (typeof args === "string") {
    const text = args.trim();
    if (!text) return fallback;
    try {
      return summarizeArgs(JSON.parse(text), fallback);
    } catch {
      return text;
    }
  }
  if (typeof args !== "object") return String(args);
  const preferred = ["command", "path", "query", "url", "content", "message"];
  for (const key of preferred) {
    if (typeof args[key] === "string" && args[key].trim()) return args[key].trim();
  }
  try {
    return JSON.stringify(args);
  } catch {
    return fallback;
  }
}

export function prettyArgs(args) {
  if (args == null) return "";
  if (typeof args === "string") {
    try {
      return JSON.stringify(JSON.parse(args), null, 2);
    } catch {
      return args;
    }
  }
  try {
    return JSON.stringify(args, null, 2);
  } catch {
    return String(args);
  }
}

/** Parse a tool_calls entry stored in the database or carried by an event. */
export function normalizeToolCall(raw) {
  if (!raw || typeof raw !== "object") return null;
  const fn = raw.function || {};
  const name = raw.name || fn.name || "tool";
  let args = raw.args ?? fn.arguments ?? raw.arguments ?? {};
  if (typeof args === "string") {
    try {
      args = JSON.parse(args);
    } catch {
      args = { raw: args };
    }
  }
  return { id: raw.id || raw.tool_call_id || "", name, args };
}
