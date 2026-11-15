/**
 * Showing what Kiwibro made: the folder a chat works in (the badge under the chat, which
 * also changes it), the files a turn wrote as previews or a short list, and the buttons that
 * belong to a file (Open, Show in folder, Copy path). Everything comes from the server:
 * /api/workspace/info, /set, /files, /open and the sandboxed /api/preview route (see
 * kiwibro/preview.py).
 */

import { openFolderPicker } from "./folderpick.js";
import { copyText, el, icon, toast } from "./ui.js";

const MAX_SHOWN = 3;
const KIND_ICON = { html: "globe", svg: "image", image: "image", video: "play", audio: "play" };
const WRITERS = new Set(["write_file", "replace_in_file", "apply_patch"]);
const RUNNERS = new Set(["run_python", "run_javascript", "run_command", "multi_command", "terminal_session"]);
const FILE_TOOLS = new Set(["read_file", "write_file", "replace_in_file", "apply_patch", "file_outline"]);
const WORKSPACE_TOOLS = new Set([
  ...WRITERS, ...RUNNERS, ...FILE_TOOLS, "read_files", "list_files", "glob_files", "grep_files", "git",
  "file_manage", "open_path", "study_file",
]);
const NAMED = /(?:^|[\s"'`(=:[])((?:\.{0,2}\/)?[\w@%+.-]+(?:\/[\w@%+.-]+)*\.(?:html?|svg|png|jpe?g|gif|webp|mp4|webm|mp3|wav|ogg))(?=$|[\s"'`),;:\]])/gi;
const FILENAME = /^[\w+.-]+\.[A-Za-z0-9]{1,8}$/;
const TEMP_NAME = "Temporary folder";

/** Does this tool call work in the chat's folder? (A component only does when it is used.) */
export function usesWorkspace(name, args) {
  return name === "component" ? args?.action === "use" : WORKSPACE_TOOLS.has(name);
}

function addNamed(text, into) {
  for (const match of String(text || "").matchAll(NAMED)) into.add(match[1]);
}

/** The files a turn's tool calls wrote, and the ones their commands and output name. */
export function collectFiles(calls) {
  const written = new Set();
  const named = new Set();
  for (const call of calls) {
    if (call.ok === false) continue;
    const args = call.args && typeof call.args === "object" ? call.args : {};
    if (WRITERS.has(call.name)) {
      if (typeof args.path === "string") written.add(args.path);
    } else if (call.name === "file_manage") {
      if (typeof args.destination === "string" && args.destination) written.add(args.destination);
    } else if (call.name === "component" && args.action === "use") {
      for (const line of String(call.text || "").split("\n")) {
        const match = line.match(/^(.+?) \(\d+ lines?\)$/);
        if (match) written.add(match[1]);
      }
    } else if (RUNNERS.has(call.name)) {
      for (const value of [args.command, args.code, ...(Array.isArray(args.commands) ? args.commands : [])]) addNamed(value, named);
      addNamed(call.text, named);
    }
  }
  return { written: [...written].slice(0, 60), named: [...named].slice(0, 60) };
}

function folderName(path) {
  const parts = String(path || "").split("/").filter(Boolean);
  return parts[parts.length - 1] || String(path || "");
}

function sizeText(bytes) {
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

function absolute(info, path) {
  if (!info || path.startsWith("/")) return path;
  return `${info.abs.replace(/\/$/, "")}/${path.replace(/^\.\//, "")}`;
}

/** A labelled block of text (arguments, a result) with a button that copies it. */
export function detailBlock(label, text, className = "") {
  const pre = el("pre", { class: className, text });
  const copy = el("button", { class: "file-btn", type: "button", "aria-label": `Copy ${label.toLowerCase()}`, title: `Copy ${label.toLowerCase()}` }, [icon("copy")]);
  copy.addEventListener("click", () => flashCopied(copy, pre.textContent));
  return el("div", { class: "detail-block" }, [
    el("div", { class: "detail-head" }, [el("div", { class: "detail-label", text: label }), copy]),
    pre,
  ]);
}

async function flashCopied(button, text, done = "Copied") {
  if (!(await copyText(text))) {
    toast("Could not copy", { tone: "error" });
    return;
  }
  button.classList.add("is-copied");
  button.querySelector("use")?.setAttribute("href", "#i-check");
  toast(done);
  window.setTimeout(() => {
    if (!button.isConnected) return;
    button.classList.remove("is-copied");
    button.querySelector("use")?.setAttribute("href", "#i-copy");
  }, 1600);
}

export function createShowFiles({ api, getConversationId, getSettings, onFolderChanged }) {
  const dock = document.getElementById("workdir");
  const badge = el("button", { class: "where where-btn", type: "button", "aria-haspopup": "dialog", on: { click: () => openPicker() } });
  dock.append(badge);
  let cache = null;
  let seq = 0;
  let shown = null; // what the badge shows: { key, data }
  let chosen = null; // a new chat's folder, chosen before the chat exists: { path } or { temp: true }

  /** The chat's folder (cached for the open chat). Null when the server cannot say. */
  function info() {
    const id = getConversationId();
    if (!id) return Promise.resolve(null);
    if (cache?.id !== id) cache = { id, promise: api.workspaceInfo(id).catch(() => null) };
    return cache.promise;
  }

  /** The parts of a folder line: icon, name, branch. A tool card also shows the folders it sits in. */
  function whereParts(data, { full }) {
    const safe = data.safe_copy;
    const text = safe ? `Safe copy of ${folderName(safe.real)}` : data.temp ? TEMP_NAME : full ? data.path : data.name || folderName(data.path);
    const cut = safe || data.temp || !full ? 0 : text.lastIndexOf("/") + 1;
    return [
      icon("folder"),
      cut ? el("span", { class: "where-dir", text: text.slice(0, cut) }) : null,
      el("span", { class: "where-name", text: text.slice(cut) }),
      data.branch ? el("span", { class: "where-branch", title: `Git branch ${data.branch}` }, [icon("branch"), el("span", { text: data.branch })]) : null,
    ];
  }

  /** `~/Desktop/` + `app` + branch; a safe copy says so, in the words of the safe copy strip. */
  function whereNode(data) {
    const safe = data.safe_copy;
    return el("span", { class: "where", title: safe ? `Safe copy of ${safe.real}` : data.abs }, whereParts(data, { full: true }));
  }

  /** The folder a chat that does not exist yet will start in: the one picked, else what the settings say. */
  async function newChatFolder() {
    if (chosen?.path) return { temp: false, abs: chosen.path, name: folderName(chosen.path), path: chosen.path };
    if (chosen?.temp || getSettings()?.workspace_mode === "temp") return { temp: true, abs: "" };
    cache = cache?.id === "" ? cache : { id: "", promise: api.workspaceInfo().catch(() => null) };
    return cache.promise;
  }

  /** What the badge shows now, or null when there is nothing to show (no answer, or a safe copy: its strip says it). */
  async function currentFolder() {
    const id = getConversationId();
    const data = id ? await info() : await newChatFolder();
    return data && !data.safe_copy ? { key: id || "new", data } : null;
  }

  async function renderDock() {
    const mine = ++seq;
    if ((getConversationId() || "new") !== shown?.key) dock.hidden = true; // another chat: never show the last one's folder
    const now = await currentFolder();
    if (mine !== seq) return;
    shown = now;
    dock.hidden = !now;
    if (!now) return;
    badge.toggleAttribute("title", Boolean(now.data.abs));
    if (now.data.abs) badge.title = now.data.abs;
    badge.setAttribute("aria-label", `Folder for this chat: ${now.data.temp ? TEMP_NAME : now.data.name || folderName(now.data.path)}`);
    badge.replaceChildren(...whereParts(now.data, { full: false }).filter(Boolean));
  }

  /** Reads the chat's folder again (it may have changed) and updates the badge above the composer. */
  function refresh() {
    cache = null;
    return renderDock();
  }

  /** Makes `body` ({ path } or { temp: true }) the folder of the open chat, or of the next one when there is no chat yet. */
  async function change(body) {
    const id = getConversationId();
    if (id) {
      const data = await api.workspaceSet({ conversation_id: id, ...body });
      cache = { id, promise: Promise.resolve(data) };
    } else if (body.temp) {
      chosen = { temp: true };
    } else {
      const listing = await api.workspaceFolders(body.path);
      if (!listing.allowed) throw new Error("Kiwibro cannot work in that folder.");
      chosen = { path: listing.path };
      cache = null;
    }
    toast("Folder changed");
    if (id) onFolderChanged?.();
    await renderDock();
  }

  function openPicker() {
    if (!shown) return;
    const { data } = shown;
    openFolderPicker({
      anchor: badge,
      api,
      path: data.temp ? "" : data.abs,
      temp: data.temp,
      onUse: (path) => change({ path }),
      onTemp: () => change({ temp: true }),
    });
  }

  async function act(action, path) {
    try {
      await api.workspaceOpen({ conversation_id: getConversationId(), action, path });
      toast(action === "open" ? "Opened on your computer" : "Showing its folder");
    } catch (error) {
      toast(error?.message || "Could not open that", { tone: "error" });
    }
  }

  /** Icon buttons for one file. A safe copy never reaches the desktop, so it only copies the path. */
  function actions(file, data) {
    const box = el("div", { class: "file-actions", role: "group", "aria-label": `Actions for ${file.path}` });
    const button = (label, name, onClick) =>
      el("button", { class: "file-btn", type: "button", "aria-label": label, title: label, on: { click: onClick } }, [icon(name)]);
    if (file.url) {
      box.append(el("a", { class: "file-btn", href: file.url, target: "_blank", rel: "noopener noreferrer", "aria-label": "Open in new tab", title: "Open in new tab" }, [icon("external")]));
    }
    if (!data?.safe_copy) {
      box.append(button("Open", "file", () => act("open", file.path)), button("Show in folder", "folder", () => act("reveal", file.path)));
    }
    const copy = button("Copy path", "copy", () => flashCopied(copy, file.abs || absolute(data, file.path), "Path copied"));
    box.append(copy);
    return box;
  }

  /** The path of a file with its buttons, filled in once the folder is known. */
  function fileBar(path) {
    const bar = el("div", { class: "file-bar" }, [icon("file"), el("span", { class: "file-bar-path", title: path, text: path })]);
    info().then((data) => bar.append(actions({ path }, data)));
    return bar;
  }

  /** What a file tool card shows first: the folder, and for a file tool the file with its buttons. */
  function toolHeader(name, args, result) {
    if (!usesWorkspace(name, args)) return null;
    const box = el("div", { class: "tool-where", hidden: true });
    info().then((data) => {
      if (!data) return;
      box.append(whereNode(data));
      const path = FILE_TOOLS.has(name) && typeof args?.path === "string" && result?.ok !== false ? args.path : "";
      if (path) box.append(fileBar(path));
      box.hidden = false;
    });
    return box;
  }

  function media(file) {
    const bar = (expand) => el("div", { class: "rich-bar" }, expand ? [el("button", { type: "button", class: "rich-expand", "aria-expanded": "false", text: "Expand" })] : []);
    if (file.kind === "html") {
      const frame = el("iframe", {
        class: "rich-frame", title: `Preview of ${file.path}`, sandbox: "allow-scripts",
        referrerpolicy: "no-referrer", loading: "lazy", src: file.url,
      });
      return el("div", { class: "rich-preview" }, [frame, bar(true)]);
    }
    if (file.kind === "video") {
      const video = el("video", { class: "rich-video", controls: true, preload: "metadata", playsinline: true, src: file.url, "aria-label": file.path });
      return el("div", { class: "rich-preview" }, [video]);
    }
    if (file.kind === "audio") {
      return el("div", { class: "rich-preview" }, [el("audio", { class: "rich-audio", controls: true, preload: "none", src: file.url, "aria-label": file.path })]);
    }
    const image = el("img", { class: file.kind === "svg" ? "rich-image" : "rich-image rich-photo", alt: file.path, loading: "lazy", decoding: "async", src: file.url });
    return el("div", { class: "rich-preview rich-media" }, [image, bar(true)]);
  }

  function card(file, data) {
    return el("div", { class: "shown-card", role: "group", "aria-label": file.path }, [
      el("div", { class: "shown-head" }, [
        icon(KIND_ICON[file.kind] || "file"),
        el("span", { class: "shown-name", title: file.path, text: file.path }),
        actions(file, data),
      ]),
      media(file),
    ]);
  }

  function row(file, data) {
    const main = file.inline
      ? el("button", { class: "shown-row-main", type: "button", "aria-expanded": "false", title: "Show it here" }, [])
      : el("span", { class: "shown-row-main", title: "Too large to show here" }, []);
    main.append(icon(KIND_ICON[file.kind] || "file"), el("span", { class: "shown-name", text: file.path }), el("span", { class: "shown-size", text: sizeText(file.size) }));
    const item = el("div", { class: "shown-row" }, [el("div", { class: "shown-row-line" }, [main, actions(file, data)])]);
    if (file.inline) {
      let body = null;
      main.addEventListener("click", () => {
        const open = main.getAttribute("aria-expanded") !== "true";
        main.setAttribute("aria-expanded", String(open));
        if (open && !body) {
          body = media(file);
          item.append(body);
        }
        if (body) body.hidden = !open;
      });
    }
    return item;
  }

  /** The page first (index.html before the others), then what the page does not already use; the rest is listed. */
  async function pick(files) {
    const rank = (file) => (/(^|\/)index\.html?$/i.test(file.path) ? 0 : 1);
    const pages = files.filter((file) => file.kind === "html" && file.inline).sort((a, b) => rank(a) - rank(b)).slice(0, 2);
    const text = (await Promise.all(pages.map((page) => fetch(page.url).then((r) => r.text()).catch(() => "")))).join("\n");
    const others = files.filter((file) => file.inline && file.kind !== "html" && !text.includes(file.name));
    const shown = [...pages, ...others].slice(0, MAX_SHOWN);
    return { shown, listed: files.filter((file) => !shown.includes(file)) };
  }

  /** After a turn: what its tool calls wrote or named, shown under the reply. */
  async function showTurn(flow, calls, since, onGrow) {
    const { written, named } = collectFiles(calls);
    const id = getConversationId();
    if (!id || !(written.length || named.length)) return;
    const holder = el("div", { class: "shown", hidden: true });
    flow.append(holder);
    try {
      const data = await api.workspaceFiles({ conversation_id: id, written, named, since: Number.isFinite(since) ? since : Date.now() / 1000 });
      const files = data?.files || [];
      if (!files.length || !holder.isConnected || id !== getConversationId()) return holder.remove();
      const [folder, picked] = await Promise.all([info(), pick(files)]);
      if (!holder.isConnected || id !== getConversationId()) return;
      holder.append(...(folder ? [whereNode(folder)] : []), ...picked.shown.map((file) => card(file, folder)));
      if (picked.listed.length) holder.append(el("div", { class: "shown-list" }, picked.listed.map((file) => row(file, folder))));
      holder.hidden = false;
      onGrow?.();
    } catch {
      holder.remove();
    }
  }

  /** A fenced block labelled with a file name that exists in the folder gets that file's buttons. */
  async function fileBlocks(container) {
    const blocks = [...(container?.querySelectorAll?.(".code-block[data-lang]:not([data-file])") ?? [])]
      .map((block) => [block, block.querySelector(".code-lang")?.textContent || ""])
      .filter(([, label]) => FILENAME.test(label));
    if (!blocks.length || !getConversationId()) return;
    blocks.forEach(([block]) => (block.dataset.file = "1"));
    try {
      const data = await api.workspaceFiles({ conversation_id: getConversationId(), check: blocks.map(([, label]) => label) });
      const folder = await info();
      for (const [block, label] of blocks) {
        const path = data?.present?.[label];
        const head = block.querySelector(".code-head");
        if (!path || !head || !block.isConnected) continue;
        head.querySelector(".code-lang").classList.add("is-file");
        head.querySelector(".code-copy").before(actions({ path }, folder));
      }
    } catch {
      /* the block keeps its plain Copy button */
    }
  }

  return {
    refresh,
    info,
    toolHeader,
    showTurn,
    fileBlocks,
    fileBar,
    /** The folder picked for a chat that does not exist yet: { path } or { temp: true }, else null. */
    pending: () => chosen,
    forgetPick() {
      chosen = null;
    },
    reset() {
      chosen = null;
      renderDock();
    },
  };
}
