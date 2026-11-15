/**
 * The folder picker under the chat: a popover that changes the folder a chat works in.
 * It lists the folders inside a folder (GET /api/workspace/folders) and hands the choice
 * back; whoever opened it decides what choosing means (showfiles.js).
 */

import { el, icon, openPopover, closePopover } from "./ui.js";

const ROW = ".folder-row";

/**
 * `path` is where the list starts ("" = the home folder); `temp` says the chat already
 * works in a temporary folder. `onUse(path)` and `onTemp()` throw to say why a choice
 * failed: the message is shown in the popover, which stays open.
 */
export function openFolderPicker({ anchor, api, path = "", temp = false, onUse, onTemp }) {
  const input = el("input", {
    type: "text",
    value: path,
    "aria-label": "Folder path",
    spellcheck: "false",
    autocomplete: "off",
    "data-autofocus": "",
  });
  const error = el("div", { class: "inline-error", role: "alert", hidden: true });
  const list = el("div", { class: "menu folder-list", "aria-label": "Folders" });
  const use = el("button", { class: "btn btn-primary", type: "button", text: "Use this folder", disabled: true });
  const tempButton = temp ? null : el("button", { class: "btn btn-ghost", type: "button", text: "Temporary folder" });
  const note = temp ? null : el("div", { class: "folder-note", text: "Cleared when the computer restarts." });
  const content = el("div", { class: "folder-pop" }, [
    el("div", { class: "folder-title", text: "Folder for this chat" }),
    el("div", { class: "path-field" }, [icon("folder"), input]),
    error,
    list,
    el("div", { class: "folder-actions" }, [tempButton, use]),
    note,
  ]);
  const handle = openPopover({ anchor, content, align: "start", label: "Folder for this chat" });
  if (!handle) return;
  const popover = handle.element;

  let shown = null; // the last listing the server sent
  let typed = false; // the field was edited since that listing
  let busy = false;
  let turn = 0;

  /** A popover opened above its anchor keeps its bottom edge where it is when its height changes. */
  function resettle(change) {
    const was = popover.offsetHeight;
    const above = parseFloat(popover.style.top) + was <= anchor.getBoundingClientRect().top + 1;
    change();
    const grew = popover.offsetHeight - was;
    if (above && grew) popover.style.top = `${Math.max(8, Math.round(parseFloat(popover.style.top) - grew))}px`;
  }

  function setError(message) {
    resettle(() => {
      error.textContent = message || "";
      error.hidden = !message;
    });
  }

  function syncButtons() {
    const allowed = typed ? Boolean(input.value.trim()) : Boolean(shown?.allowed);
    use.disabled = busy || !allowed;
    if (tempButton) tempButton.disabled = busy;
  }

  function row(name, iconName, onClick, title = "") {
    return el("button", { class: "menu-item folder-row", type: "button", tabindex: "-1", title, on: { click: onClick } }, [
      icon(iconName),
      el("span", { class: "menu-item-body" }, [el("span", { class: "menu-item-name", text: name })]),
    ]);
  }

  function paint() {
    const refocus = list.contains(document.activeElement);
    const rows = [];
    if (shown.parent) rows.push(row("Up", "chevron-up", () => browse(shown.parent), shown.parent));
    for (const folder of shown.folders) rows.push(row(folder.name, "folder", () => browse(folder.path), folder.path));
    rows[0]?.setAttribute("tabindex", "0");
    const notes = [];
    if (!shown.folders.length) notes.push(el("div", { class: "menu-empty", text: "No folders here" }));
    if (shown.truncated) notes.push(el("div", { class: "menu-empty", text: "Not all folders are shown" }));
    list.replaceChildren(...rows, ...notes);
    list.scrollTop = 0;
    if (refocus) rows[0]?.focus({ preventScroll: true });
  }

  async function browse(target) {
    const mine = ++turn;
    list.setAttribute("aria-busy", "true");
    try {
      const listing = await api.workspaceFolders(target);
      if (mine !== turn) return;
      shown = listing;
      typed = false;
      input.value = listing.path;
      setError("");
      paint();
    } catch (failure) {
      if (mine !== turn) return;
      setError(failure?.message || "That folder can't be opened.");
    } finally {
      if (mine === turn) list.removeAttribute("aria-busy");
      syncButtons();
    }
  }

  /** Runs a choice; on success the popover closes and focus goes back to the anchor. */
  async function choose(run) {
    if (busy) return;
    busy = true;
    syncButtons();
    setError("");
    try {
      await run();
      closePopover();
    } catch (failure) {
      setError(failure?.message || "That folder can't be used.");
      busy = false;
      syncButtons();
    }
  }

  use.addEventListener("click", () => choose(() => onUse(typed ? input.value.trim() : shown.path)));
  tempButton?.addEventListener("click", () => choose(onTemp));
  input.addEventListener("input", () => {
    typed = true;
    syncButtons();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      if (input.value.trim()) choose(() => onUse(input.value.trim()));
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      list.querySelector(ROW)?.focus();
    }
  });
  list.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const rows = [...list.querySelectorAll(ROW)];
    const next = rows[rows.indexOf(document.activeElement) + (event.key === "ArrowDown" ? 1 : -1)];
    if (next) next.focus();
    else if (event.key === "ArrowUp") input.focus();
  });

  browse(path || undefined).then(() => {
    if (!shown && path) return browse(undefined); // the folder is gone: start from home
  });
}
