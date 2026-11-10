/**
 * The Notes panel: the user's own notebook, plus notes Kiwibro saved in a chat.
 * A list on the left and an editor on the right; one pane at a time on phones.
 * Edits autosave. A list refresh never overwrites text the user is typing.
 */
import { renderMarkdown } from "./markdown.js";
import { $, el, icon, debounce, relativeTime, absoluteTime, openSheet, toast } from "./ui.js";

const AUTOSAVE_MS = 700;
const SEARCH_MS = 250;
const TAG_LIMIT = 12;
const PHONE_QUERY = "(max-width: 760px)";
const EMPTY_ALL = "No notes yet. Write one, or ask Kiwibro in a chat to take a note.";
const EMPTY_SEARCH = "Nothing matches.";
const PICK_NOTE = "Pick a note, or start a new one.";
const NEEDS_TEXT = "Write something to save";

export function createNotesView({ dialog, api, onCount, onOpenConversation }) {
  const root = $("#notes-body", dialog);
  const countEl = $("#notes-count", dialog);

  /* ---------- Skeleton, built once ------------------------------------------ */

  const searchInput = el("input", {
    type: "search",
    placeholder: "Search notes",
    "aria-label": "Search notes",
    autocomplete: "off",
  });
  const newButton = el("button", { class: "btn btn-primary btn-compact", type: "button" }, [icon("plus"), "New"]);
  const tagsEl = el("div", { class: "nt-tags", hidden: true });
  const listEl = el("div", { class: "nt-list" });
  const side = el("section", { class: "nt-side", "aria-label": "All notes" }, [
    el("div", { class: "nt-top" }, [el("div", { class: "search-field" }, [icon("search"), searchInput]), newButton]),
    tagsEl,
    listEl,
  ]);

  const backButton = el("button", { class: "icon-btn nt-back", type: "button", "aria-label": "All notes" }, [
    icon("arrow-left"),
  ]);
  const writeButton = el("button", { class: "nt-mode", type: "button", "aria-pressed": "true", text: "Write" });
  const readButton = el("button", { class: "nt-mode", type: "button", "aria-pressed": "false", text: "Read" });
  const savedEl = el("span", { class: "nt-saved", "aria-live": "polite" });
  const pinButton = el("button", { class: "icon-btn nt-pin-btn", type: "button", "aria-label": "Pin note" }, [
    icon("pin"),
  ]);
  const deleteButton = el("button", { class: "icon-btn nt-delete", type: "button", "aria-label": "Delete note" }, [
    icon("trash"),
  ]);
  const tools = el("div", { class: "nt-tools" }, [savedEl, pinButton, deleteButton]);
  const keepButton = el("button", { class: "btn btn-ghost btn-compact", type: "button", text: "Keep" });
  const confirmButton = el("button", { class: "btn btn-danger btn-compact", type: "button", text: "Delete" });
  const confirmBox = el("div", { class: "nt-confirm", hidden: true }, [
    el("span", { text: "Delete this note?" }),
    confirmButton,
    keepButton,
  ]);
  const bar = el("div", { class: "nt-bar" }, [
    backButton,
    el("div", { class: "nt-mode-group" }, [writeButton, readButton]),
    el("span", { class: "nt-spacer" }),
    tools,
    confirmBox,
  ]);

  const titleInput = el("input", {
    class: "nt-title",
    type: "text",
    placeholder: "Title",
    maxlength: "120",
    "aria-label": "Title",
  });
  const tagInput = el("input", {
    class: "nt-taginput",
    type: "text",
    placeholder: "Add tags, separated by commas",
    "aria-label": "Tags",
  });
  const bodyArea = el("textarea", {
    class: "nt-body",
    placeholder: "Write here. Markdown works.",
    maxlength: "20000",
    "aria-label": "Note text",
  });
  const readView = el("div", { class: "nt-read md", hidden: true });
  const openChatButton = el("button", { class: "text-btn", type: "button", text: "Open that chat" });
  const originBar = el("div", { class: "nt-origin", hidden: true }, [
    el("span", { text: "Saved by Kiwibro during a chat" }),
    openChatButton,
  ]);
  const editorEl = el("div", { class: "nt-editor", hidden: true }, [
    bar,
    titleInput,
    tagInput,
    bodyArea,
    readView,
    originBar,
  ]);
  const placeholderEl = el("div", { class: "panel-empty", text: PICK_NOTE });
  const main = el("section", { class: "nt-main" }, [editorEl, placeholderEl]);
  root.classList.add("nt-root");
  root.replaceChildren(side, main);

  /* ---------- State ---------------------------------------------------------- */

  const state = { q: "", tag: "", notes: [], tags: [], total: 0, loaded: false };
  // The open note: { id, saved, pinned, origin, sourceId, chain, inFlight }.
  // `saved` holds the values the server last confirmed (or the ones it was sent).
  let editor = null;
  let mode = "write";
  let confirming = false;
  let pinBusy = false;
  let listSeq = 0;
  let openSeq = 0;

  const isWide = () => !window.matchMedia(PHONE_QUERY).matches;
  const rowEls = () => Array.from(listEl.querySelectorAll(".nt-item"));

  const autosave = debounce(() => {
    flush();
  }, AUTOSAVE_MS);
  const searchLater = debounce(() => {
    const q = searchInput.value.trim();
    if (q === state.q) return;
    state.q = q;
    loadList();
  }, SEARCH_MS);

  /* ---------- Fields and diffs ----------------------------------------------- */

  function parseTags(text) {
    return text
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
  }

  function readFields() {
    return { title: titleInput.value.trim(), body: bodyArea.value, tags: parseTags(tagInput.value) };
  }

  function fieldsOf(note) {
    return { title: note.title || "", body: note.body || "", tags: [...(note.tags || [])] };
  }

  function changedFields(saved, now) {
    const patch = {};
    if (now.title !== saved.title) patch.title = now.title;
    if (now.body !== saved.body) patch.body = now.body;
    if (now.tags.join(",") !== saved.tags.join(",")) patch.tags = now.tags;
    return patch;
  }

  function hasChanges() {
    return Boolean(editor) && Object.keys(changedFields(editor.saved, readFields())).length > 0;
  }

  function recordFor(note) {
    return {
      id: note?.id ?? null,
      saved: note ? fieldsOf(note) : { title: "", body: "", tags: [] },
      pinned: Boolean(note?.pinned),
      origin: note?.origin || "you",
      sourceId: note?.source_conversation_id || null,
      chain: Promise.resolve(),
      inFlight: 0,
    };
  }

  // The start of the body as plain text. A leading heading that repeats the title is skipped.
  function previewOf(note) {
    let body = String(note.body || "");
    const first = body.split("\n").find((line) => line.trim()) || "";
    const heading = first.replace(/^\s*#{1,6}\s*/, "").trim();
    const repeatsTitle =
      heading !== first.trim() && heading.toLowerCase() === String(note.title || "").trim().toLowerCase();
    if (repeatsTitle) body = body.slice(body.indexOf(first) + first.length);
    return body
      .replace(/^\s*(?:[-+*]|\d+\.)\s+/gm, "")
      .replace(/[#*_>`]/g, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  /* ---------- List ----------------------------------------------------------- */

  function buildRow(note) {
    const open = Boolean(editor && editor.id === note.id);
    const stamp = note.updated_at || note.created_at || "";
    const meta = [el("span", { dataset: { ts: stamp }, title: absoluteTime(stamp), text: relativeTime(stamp) })];
    if (note.origin === "kiwibro") meta.push(el("span", { text: " · saved by Kiwibro" }));
    return el(
      "button",
      { class: "nt-item", type: "button", dataset: { id: note.id }, "aria-current": open ? "true" : false },
      [
        el("span", { class: "nt-item-head" }, [
          el("span", { class: "nt-item-title", text: note.title || "Untitled" }),
          note.pinned ? el("span", { class: "nt-pin", title: "Pinned" }, [icon("pin")]) : null,
        ]),
        el("span", { class: "nt-item-preview", text: previewOf(note) }),
        el("span", { class: "nt-item-meta" }, meta),
      ],
    );
  }

  function renderList() {
    if (!state.notes.length) {
      listEl.replaceChildren(el("div", { class: "panel-empty", text: state.q || state.tag ? EMPTY_SEARCH : EMPTY_ALL }));
      return;
    }
    listEl.replaceChildren(...state.notes.map(buildRow));
  }

  function renderLoading() {
    listEl.replaceChildren(
      el("div", { class: "nt-state" }, [el("span", { class: "spinner", role: "status", "aria-label": "Loading notes" })]),
    );
  }

  function renderError(message) {
    listEl.replaceChildren(
      el("div", { class: "nt-state" }, [
        el("div", { class: "panel-error", role: "alert" }, [
          icon("alert"),
          el("span", { text: message || "Could not load notes." }),
          el("button", { class: "btn btn-ghost btn-compact", type: "button", text: "Try again", on: { click: () => loadList() } }),
        ]),
      ]),
    );
  }

  function renderTags() {
    let shown = state.tags.slice(0, TAG_LIMIT);
    const active = state.tags.find((item) => item.tag === state.tag);
    if (active && !shown.includes(active)) shown = [...shown.slice(0, TAG_LIMIT - 1), active];
    tagsEl.hidden = shown.length === 0;
    tagsEl.replaceChildren(
      ...shown.map((item) =>
        el(
          "button",
          {
            class: "nt-tag",
            type: "button",
            dataset: { tag: item.tag },
            "aria-pressed": String(item.tag === state.tag),
          },
          [el("span", { text: item.tag }), el("span", { class: "nt-tag-count", text: String(item.count) })],
        ),
      ),
    );
  }

  function renderCount() {
    countEl.textContent = state.total > 0 ? String(state.total) : "";
  }

  function updateRowSelection() {
    for (const row of rowEls()) {
      if (editor && row.dataset.id === editor.id) row.setAttribute("aria-current", "true");
      else row.removeAttribute("aria-current");
    }
  }

  // Replace one row in place (or insert it) without rebuilding the list.
  function upsertRow(note) {
    const index = state.notes.findIndex((item) => item.id === note.id);
    let at = index;
    if (index >= 0) {
      state.notes[index] = note;
    } else {
      at = note.pinned ? 0 : state.notes.findIndex((item) => !item.pinned);
      if (at < 0) at = state.notes.length;
      state.notes.splice(at, 0, note);
    }
    const fresh = buildRow(note);
    const old = rowEls().find((row) => row.dataset.id === note.id);
    if (old) {
      const hadFocus = old === document.activeElement;
      old.replaceWith(fresh);
      if (hadFocus) fresh.focus();
      return;
    }
    listEl.querySelector(".panel-empty")?.remove();
    listEl.insertBefore(fresh, rowEls()[at] ?? null);
  }

  // Keep fresher local copies of notes that a slower list response returns stale.
  function mergeFresh(incoming) {
    const cached = new Map(state.notes.map((note) => [note.id, note]));
    return incoming.map((note) => {
      const known = cached.get(note.id);
      return known && known.updated_at > note.updated_at ? known : note;
    });
  }

  async function loadList() {
    const seq = ++listSeq;
    if (!state.loaded) renderLoading();
    try {
      const data = await api.notes({ q: state.q, tag: state.tag });
      if (seq !== listSeq) return;
      state.loaded = true;
      state.notes = mergeFresh(data.notes || []);
      state.tags = data.tags || [];
      state.total = Number.isFinite(data.total) ? data.total : state.notes.length;
      renderTags();
      renderList();
      renderCount();
      onCount?.(state.total);
      syncFromList();
    } catch (error) {
      if (seq !== listSeq) return;
      if (state.loaded) toast(error?.message || "Could not load notes", { tone: "error" });
      else renderError(error?.message);
    }
  }

  // A clean open note takes the newer text the list brings (for example, one Kiwibro edited).
  function syncFromList() {
    if (!editor || editor.id == null || editor.inFlight || hasChanges()) return;
    const note = state.notes.find((item) => item.id === editor.id);
    if (!note) return;
    const fresh = fieldsOf(note);
    editor.pinned = Boolean(note.pinned);
    editor.origin = note.origin || "you";
    editor.sourceId = note.source_conversation_id || null;
    if (Object.keys(changedFields(editor.saved, fresh)).length) {
      editor.saved = fresh;
      titleInput.value = fresh.title;
      bodyArea.value = fresh.body;
      tagInput.value = fresh.tags.join(", ");
      renderMode();
    }
    renderPin();
    renderOrigin();
    syncStatus();
  }

  /* ---------- Editor --------------------------------------------------------- */

  function setStatus(text) {
    savedEl.textContent = text;
  }

  function syncStatus() {
    if (!editor) return;
    if (editor.id == null && !bodyArea.value.trim()) setStatus(NEEDS_TEXT);
    else if (hasChanges()) setStatus("Saving…");
    else setStatus("Saved");
  }

  function renderMode() {
    writeButton.setAttribute("aria-pressed", String(mode === "write"));
    readButton.setAttribute("aria-pressed", String(mode === "read"));
    bodyArea.hidden = mode !== "write";
    readView.hidden = mode !== "read";
    if (mode !== "read") return;
    if (!bodyArea.value.trim()) {
      readView.replaceChildren(el("p", { class: "nt-read-empty", text: "Nothing written yet." }));
      return;
    }
    readView.innerHTML = renderMarkdown(bodyArea.value);
  }

  function renderPin() {
    const pinned = Boolean(editor?.pinned);
    pinButton.setAttribute("aria-pressed", String(pinned));
    pinButton.setAttribute("aria-label", pinned ? "Unpin note" : "Pin note");
    pinButton.disabled = !editor || editor.id == null || pinBusy;
  }

  function renderOrigin() {
    originBar.hidden = !(editor && editor.origin === "kiwibro" && editor.sourceId);
  }

  function showConfirm(on) {
    confirming = on;
    bar.classList.toggle("nt-confirming", on);
    tools.hidden = on;
    confirmBox.hidden = !on;
  }

  function showRecord(record) {
    editor = record;
    titleInput.value = record.saved.title;
    bodyArea.value = record.saved.body;
    tagInput.value = record.saved.tags.join(", ");
    editorEl.hidden = false;
    placeholderEl.hidden = true;
    showConfirm(false);
    renderMode();
    renderPin();
    renderOrigin();
    syncStatus();
    updateRowSelection();
  }

  function showPlaceholder() {
    editor = null;
    autosave.cancel();
    showConfirm(false);
    editorEl.hidden = true;
    placeholderEl.hidden = false;
    updateRowSelection();
  }

  const showMain = () => root.classList.add("nt-show-main");
  const showList = () => root.classList.remove("nt-show-main");

  function setMode(next) {
    mode = next;
    renderMode();
  }

  function onEdit() {
    if (!editor) return;
    if (editor.id == null && !bodyArea.value.trim()) {
      autosave.cancel();
      setStatus(NEEDS_TEXT);
      return;
    }
    setStatus("Saving…");
    autosave();
  }

  /* ---------- Saving --------------------------------------------------------- */

  // Saves run one at a time per note. Resolves true when everything is saved.
  function flush() {
    autosave.cancel();
    if (!editor) return Promise.resolve(true);
    const record = editor;
    record.chain = record.chain.then(() => saveRecord(record)).catch(() => false);
    return record.chain;
  }

  async function saveRecord(record) {
    if (record !== editor) return true;
    const fields = readFields();
    if (record.id == null) return createRecord(record, fields);
    const patch = changedFields(record.saved, fields);
    if (!Object.keys(patch).length) {
      syncStatus();
      return true;
    }
    record.inFlight += 1;
    setStatus("Saving…");
    try {
      const { note } = await api.patchNote(record.id, patch);
      record.saved = { ...record.saved, ...patch };
      if ("title" in patch) adoptServerTitle(record, patch.title, note);
      applySaved(record, note);
      return finishSave(record);
    } catch (error) {
      return failSave(record, error);
    } finally {
      record.inFlight -= 1;
    }
  }

  async function createRecord(record, fields) {
    if (!fields.body.trim()) {
      if (record === editor) setStatus(NEEDS_TEXT);
      return true;
    }
    record.inFlight += 1;
    setStatus("Saving…");
    try {
      const { note } = await api.createNote({ title: fields.title, body: fields.body, tags: fields.tags });
      record.id = note.id;
      record.saved = fields;
      adoptServerTitle(record, fields.title, note);
      state.total += 1;
      applySaved(record, note);
      renderCount();
      onCount?.(state.total);
      if (state.q || state.tag) loadList();
      return finishSave(record);
    } catch (error) {
      return failSave(record, error);
    } finally {
      record.inFlight -= 1;
    }
  }

  // The server names a note whose title was sent empty (first line of the body).
  function adoptServerTitle(record, sentTitle, note) {
    if (sentTitle !== "" || !note.title) return;
    record.saved.title = note.title;
    if (record === editor && !titleInput.value.trim()) titleInput.value = note.title;
  }

  function applySaved(record, note) {
    record.pinned = Boolean(note.pinned);
    record.origin = note.origin || record.origin;
    record.sourceId = note.source_conversation_id || null;
    upsertRow(note);
    if (record === editor) {
      updateRowSelection();
      renderPin();
      renderOrigin();
    }
  }

  function finishSave(record) {
    if (record === editor) syncStatus();
    return true;
  }

  function failSave(record, error) {
    if (record === editor) setStatus("Not saved");
    toast(error?.message || "Could not save note", { tone: "error" });
    return false;
  }

  /* ---------- Switching, creating, pinning, deleting -------------------------- */

  // Returns false when the open note must stay (a save failed, or it is an unfinished new note).
  async function leaveEditor() {
    if (!editor) return true;
    if (editor.id == null) {
      const fields = readFields();
      if (!fields.body.trim()) {
        if (!fields.title && !fields.tags.length) return true;
        toast(`${NEEDS_TEXT} first.`, { tone: "info" });
        bodyArea.focus();
        return false;
      }
    }
    for (let attempt = 0; attempt < 3; attempt += 1) {
      if (!(await flush())) return false;
      if (!hasChanges()) return true;
    }
    return false;
  }

  async function openNote(id) {
    if (editor && editor.id === id) {
      showMain();
      return true;
    }
    const seq = ++openSeq;
    if (!(await leaveEditor())) return false;
    if (seq !== openSeq) return false;
    const note = state.notes.find((item) => item.id === id);
    if (!note) return false;
    showRecord(recordFor(note));
    showMain();
    return true;
  }

  async function newNote() {
    if (editor && editor.id == null) {
      titleInput.focus();
      return;
    }
    const seq = ++openSeq;
    if (!(await leaveEditor())) return;
    if (seq !== openSeq) return;
    showRecord(recordFor(null));
    showMain();
    titleInput.focus();
  }

  async function togglePin() {
    if (!editor || editor.id == null || pinBusy) return;
    const record = editor;
    pinBusy = true;
    renderPin();
    try {
      if (!(await flush())) return;
      if (editor !== record) return;
      const { note } = await api.patchNote(record.id, { pinned: !record.pinned });
      record.pinned = Boolean(note.pinned);
      await loadList();
    } catch (error) {
      toast(error?.message || "Could not change the pin", { tone: "error" });
    } finally {
      pinBusy = false;
      renderPin();
    }
  }

  async function deleteNote() {
    const record = editor;
    if (!record || confirmButton.disabled) return;
    confirmButton.disabled = true;
    autosave.cancel();
    try {
      if (record.id != null) {
        await record.chain;
        await api.deleteNote(record.id);
      }
    } catch (error) {
      confirmButton.disabled = false;
      toast(error?.message || "Could not delete note", { tone: "error" });
      return;
    }
    confirmButton.disabled = false;
    const index = record.id == null ? -1 : state.notes.findIndex((note) => note.id === record.id);
    if (record.id != null) {
      state.notes = state.notes.filter((note) => note.id !== record.id);
      state.total = Math.max(0, state.total - 1);
      renderCount();
      onCount?.(state.total);
      toast("Note deleted", { iconName: "trash" });
    }
    renderList();
    const next = index >= 0 ? state.notes[index] || state.notes[index - 1] || null : null;
    if (next) showRecord(recordFor(next));
    else showPlaceholder();
    showList();
    loadList();
  }

  /* ---------- Wiring --------------------------------------------------------- */

  function moveFromRow(event) {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const rows = rowEls();
    const at = rows.indexOf(event.target.closest(".nt-item"));
    if (at < 0) return;
    event.preventDefault();
    const next = rows[at + (event.key === "ArrowDown" ? 1 : -1)];
    if (!next) return;
    openNote(next.dataset.id).then((ok) => {
      if (ok && isWide()) next.focus();
    });
  }

  searchInput.addEventListener("input", searchLater);
  newButton.addEventListener("click", newNote);
  tagsEl.addEventListener("click", (event) => {
    const button = event.target.closest(".nt-tag");
    if (!button) return;
    const tag = button.dataset.tag;
    state.tag = state.tag === tag ? "" : tag;
    renderTags();
    loadList();
  });
  listEl.addEventListener("click", (event) => {
    const row = event.target.closest(".nt-item");
    if (row) openNote(row.dataset.id);
  });
  listEl.addEventListener("keydown", moveFromRow);

  backButton.addEventListener("click", () => {
    showList();
    rowEls()
      .find((row) => row.getAttribute("aria-current") === "true")
      ?.focus();
  });
  writeButton.addEventListener("click", () => setMode("write"));
  readButton.addEventListener("click", () => setMode("read"));
  pinButton.addEventListener("click", togglePin);
  deleteButton.addEventListener("click", () => showConfirm(true));
  keepButton.addEventListener("click", () => {
    showConfirm(false);
    deleteButton.focus();
  });
  confirmButton.addEventListener("click", deleteNote);
  openChatButton.addEventListener("click", () => {
    if (editor?.sourceId) onOpenConversation?.(editor.sourceId);
  });

  for (const field of [titleInput, tagInput, bodyArea]) field.addEventListener("input", onEdit);
  titleInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      bodyArea.focus();
    }
  });
  editorEl.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      flush();
    }
  });

  // Escape cancels an open delete confirm instead of closing the panel.
  dialog.addEventListener(
    "keydown",
    (event) => {
      if (!confirming || event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      showConfirm(false);
      deleteButton.focus();
    },
    true,
  );
  dialog.addEventListener("close", () => {
    showConfirm(false);
    flush();
  });

  showPlaceholder();

  /* ---------- Public --------------------------------------------------------- */

  function open() {
    openSheet(dialog);
    showList();
    loadList().then(() => {
      if (!editor && isWide() && state.notes.length) openNote(state.notes[0].id);
    });
  }

  function refreshIfOpen() {
    if (dialog.open) loadList();
  }

  return { open, refreshIfOpen };
}
