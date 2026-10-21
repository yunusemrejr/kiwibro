// File attachments for the chat composer: pick, paste or drop files, upload each
// one as it arrives and hand the ready attachments to the next message. Nothing
// is sent from here; the chat request reads take() when it goes out.

import { el, icon, toast } from "./ui.js";

const UPLOAD_URL = "/api/attachments";
const MAX_ATTACHMENTS = 8;
const MAX_BYTES = 25 * 1024 * 1024;
const IMAGE_KINDS = new Set(["image", "svg"]);

function formatSize(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  const mb = value / (1024 * 1024);
  return `${mb >= 10 ? Math.round(mb) : mb.toFixed(1)} MB`;
}

// Images and SVG get the picture icon; everything else gets the document icon.
// Before the server answers, the browser's MIME type decides.
function looksLikeImage(item) {
  if (item.kind) return IMAGE_KINDS.has(item.kind);
  return String(item.file.type || "").startsWith("image/");
}

function carriesFiles(event) {
  return Array.from(event.dataTransfer?.types || []).includes("Files");
}

/* ---------- Composer attachments ------------------------------------------- */

export function createAttachments({ composer, input, anchor, onChange, disabled = false }) {
  const items = [];
  let nextId = 1;
  let tray = null;

  const notify = () => onChange?.();
  const isLive = (id) => items.some((item) => item.id === id);

  function syncTray() {
    if (tray) tray.hidden = items.length === 0;
  }

  /* ---------- Chips ---------------------------------------------------------- */

  function mount(item) {
    const iconSvg = icon("file", "attach-icon");
    const name = el("span", { class: "attach-name" });
    const size = el("span", { class: "attach-size" });
    const remove = el(
      "button",
      {
        class: "attach-remove",
        type: "button",
        on: { click: (event) => removeItem(item, event.detail === 0) },
      },
      [icon("x")],
    );
    const chip = el("div", { class: "attach-chip", role: "listitem" }, [iconSvg, name, size, remove]);
    item.refs = { chip, use: iconSvg.querySelector("use"), name, size, remove };
    tray.append(chip);
    paint(item);
  }

  // Every visible change to a chip goes through here, so the chip is updated in
  // place and keeps keyboard focus while an upload finishes.
  function paint(item) {
    const { chip, use, name, size, remove } = item.refs;
    use.setAttribute("href", `#i-${looksLikeImage(item) ? "image" : "file"}`);
    chip.dataset.state = item.state;
    chip.title =
      item.state === "uploading" ? "Uploading…" : item.state === "error" ? item.error : item.name;
    chip.toggleAttribute("aria-busy", item.state === "uploading");
    name.textContent = item.name;
    size.textContent = formatSize(item.size);
    remove.setAttribute("aria-label", `Remove ${item.name}`);
  }

  /* ---------- Upload --------------------------------------------------------- */

  function start(file) {
    const item = {
      id: nextId++,
      file,
      name: file.name,
      size: file.size,
      state: "uploading",
      kind: "",
      attachment: null,
      error: "",
      controller: new AbortController(),
      refs: null,
    };
    items.push(item);
    mount(item);
    syncTray();
    notify();
    upload(item);
  }

  async function upload(item) {
    const body = new FormData();
    body.append("file", item.file, item.name);
    let attachment;
    try {
      const response = await fetch(UPLOAD_URL, {
        method: "POST",
        body,
        signal: item.controller.signal,
      });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data?.ok || !data.attachment) {
        throw new Error(data?.error?.message || `Upload failed (HTTP ${response.status})`);
      }
      attachment = data.attachment;
    } catch (error) {
      // A chip removed mid-upload stays removed, whatever the request did.
      if (!isLive(item.id)) return;
      const message =
        error?.name === "TypeError" ? "Couldn't reach the server" : error?.message || "Upload failed";
      item.state = "error";
      item.error = message;
      paint(item);
      toast(message, { tone: "error", iconName: "alert" });
      notify();
      return;
    }
    if (!isLive(item.id)) return;
    item.state = "ready";
    item.kind = attachment.kind || "other";
    item.attachment = { ...attachment };
    item.name = attachment.name || item.name;
    item.size = Number(attachment.size) || item.size;
    paint(item);
    notify();
  }

  function addFiles(list) {
    let overflow = false;
    for (const file of list) {
      if (items.length >= MAX_ATTACHMENTS) {
        overflow = true;
        continue;
      }
      if (file.size > MAX_BYTES) {
        toast(`${file.name} is larger than 25 MB`, { tone: "error", iconName: "alert" });
        continue;
      }
      start(file);
    }
    if (overflow) {
      toast(`Up to ${MAX_ATTACHMENTS} files per message`, { tone: "error", iconName: "alert" });
    }
  }

  /* ---------- Removal -------------------------------------------------------- */

  // Aborting an upload that is still running keeps busy() honest and stops
  // the request; the isLive check covers a response that already arrived.
  function drop(item) {
    item.controller.abort();
    const index = items.indexOf(item);
    if (index !== -1) items.splice(index, 1);
    item.refs.chip.remove();
  }

  function removeItem(item, fromKeyboard) {
    drop(item);
    syncTray();
    // Keyboard removal would otherwise leave focus on nothing.
    if (fromKeyboard) input.focus();
    notify();
  }

  function take() {
    const ready = items
      .filter((item) => item.state === "ready")
      .map((item) => ({ ...item.attachment }));
    for (const item of items.filter((entry) => entry.state !== "uploading")) drop(item);
    syncTray();
    notify();
    return ready;
  }

  function clear() {
    for (const item of items.slice()) drop(item);
    syncTray();
    notify();
  }

  /* ---------- Wiring --------------------------------------------------------- */

  function wire() {
    tray = el("div", {
      class: "attach-tray",
      role: "list",
      "aria-label": "Attachments",
      hidden: true,
    });
    input.before(tray);

    const fileInput = el("input", { type: "file", multiple: true, hidden: true });
    composer.append(fileInput);
    fileInput.addEventListener("change", () => {
      const picked = Array.from(fileInput.files || []);
      // Reset so picking the same file again still fires change.
      fileInput.value = "";
      addFiles(picked);
    });

    const button = el(
      "button",
      {
        class: "icon-btn attach-btn",
        type: "button",
        "aria-label": "Attach files",
        title: "Attach images, audio, video or documents",
        on: { click: () => fileInput.click() },
      },
      [icon("file-plus")],
    );
    anchor.before(button);

    // Plain-text paste has no files, so it falls through to the textarea untouched.
    input.addEventListener("paste", (event) => {
      const files = event.clipboardData?.files;
      if (!files?.length) return;
      event.preventDefault();
      addFiles(Array.from(files));
    });

    // Only file drags are claimed; dragged text keeps its native drop behaviour.
    composer.addEventListener("dragover", (event) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      composer.classList.add("is-dragging");
    });
    composer.addEventListener("dragleave", (event) => {
      if (event.relatedTarget && composer.contains(event.relatedTarget)) return;
      composer.classList.remove("is-dragging");
    });
    composer.addEventListener("drop", (event) => {
      composer.classList.remove("is-dragging");
      if (!carriesFiles(event)) return;
      event.preventDefault();
      addFiles(Array.from(event.dataTransfer.files || []));
    });
  }

  if (!disabled) wire();

  return {
    count: () => items.filter((item) => item.state === "ready").length,
    busy: () => items.some((item) => item.state === "uploading"),
    take,
    clear,
  };
}

/* ---------- Sent messages -------------------------------------------------- */

/** Static chips for attachments shown inside a sent user message. */
export function attachmentChips(list) {
  if (!Array.isArray(list) || !list.length) return null;
  return el(
    "div",
    { class: "attach-list", role: "list", "aria-label": "Attachments" },
    list
      .filter((attachment) => attachment && typeof attachment === "object")
      .map((attachment) =>
        el("div", { class: "attach-chip", role: "listitem", title: attachment.path || "" }, [
          icon(IMAGE_KINDS.has(attachment.kind) ? "image" : "file", "attach-icon"),
          el("span", { class: "attach-name", text: attachment.name || attachment.path || "" }),
        ]),
      ),
  );
}
