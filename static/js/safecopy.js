/**
 * Safe copy: the strip above the composer, the review sheet, the End confirm and the
 * start control in Settings > Workspace. A copy belongs to a chat. Nothing reaches the
 * real folder until the user applies a proof they have read.
 */
import { el, icon, openSheet, wireDialog, toast, copyText, openPopover, closePopover } from "./ui.js";

const KIND_TEXT = { added: "new", modified: "changed", deleted: "deleted" };
const STALE_TEXT = "The copy changed while you were reading. Here is the new proof.";
const CONFLICT_TEXT = "These files changed in your folder after the copy was made, so nothing can be applied:";
const CONFLICT_NEXT = "End this copy and start a new one to work on the folder as it is now.";

function files(count) {
  return count === 1 ? "1 file" : `${count} files`;
}

/** The home folder shown as ~, the way the chat shows paths. */
export function shortPath(path) {
  return String(path || "").replace(/^\/(?:home|Users)\/[^/]+(?=\/|$)/, "~");
}

/** The folder's own name: the last path segment. */
export function folderName(path) {
  const parts = String(path || "").split("/").filter(Boolean);
  return parts[parts.length - 1] || String(path || "");
}

/** `demo: 2 changed, 1 new, 1 deleted. +5 −3 lines.`, leaving out zero parts. */
export function summaryLine(totals, name) {
  const parts = [];
  if (totals.modified) parts.push(`${totals.modified} changed`);
  if (totals.added) parts.push(`${totals.added} new`);
  if (totals.deleted) parts.push(`${totals.deleted} deleted`);
  const lines = [];
  if (totals.added_lines) lines.push(`+${totals.added_lines}`);
  if (totals.removed_lines) lines.push(`−${totals.removed_lines}`);
  const head = `${name}: ${parts.length ? `${parts.join(", ")}.` : "no changes."}`;
  return lines.length ? `${head} ${lines.join(" ")} lines.` : head;
}

function setBusy(button, busy) {
  button.disabled = busy;
  button.toggleAttribute("aria-busy", busy);
  const spinner = button.querySelector(":scope > .spinner");
  if (busy && !spinner) button.prepend(el("span", { class: "spinner", "aria-hidden": "true" }));
  if (!busy && spinner) spinner.remove();
}

function copyButton(textOf) {
  const label = el("span", { text: "Copy" });
  const button = el("button", { class: "code-copy", type: "button", "aria-label": "Copy code", title: "Copy code" }, [
    icon("copy"),
    label,
  ]);
  button.addEventListener("click", async () => {
    if (!(await copyText(textOf()))) {
      toast("Couldn't copy. Select the text and copy it.", { tone: "error", iconName: "alert" });
      return;
    }
    button.classList.add("is-copied");
    label.textContent = "Copied";
    window.setTimeout(() => {
      button.classList.remove("is-copied");
      label.textContent = "Copy";
    }, 1600);
  });
  return button;
}

function commandBlock(text) {
  return el("div", { class: "code-block safecopy-code safecopy-cmd" }, [
    el("div", { class: "code-head" }, [el("span", { class: "code-lang", text: "Terminal" }), copyButton(() => text)]),
    el("pre", {}, [el("code", { text })]),
  ]);
}

function diffChildren(diff) {
  const children = [];
  diff.replace(/\n$/, "").split("\n").forEach((line, index) => {
    if (index) children.push("\n");
    let cls = null;
    if (/^(\+\+\+|---|diff |index )/.test(line)) cls = "line-meta";
    else if (line.startsWith("@@")) cls = "line-hunk";
    else if (line.startsWith("+")) cls = "line-add";
    else if (line.startsWith("-")) cls = "line-del";
    children.push(cls ? el("span", { class: cls, text: line }) : line);
  });
  return children;
}

/** The diff under a disclosure; the label flips between Show and Hide. */
function diffDisclosure(diff) {
  const summary = el("summary", { class: "btn btn-ghost", text: "Show the changes" });
  const details = el("details", { class: "safecopy-diff" }, [
    summary,
    el("div", { class: "code-block safecopy-code" }, [el("pre", {}, [el("code", {}, diffChildren(diff))])]),
  ]);
  details.addEventListener("toggle", () => {
    summary.textContent = details.open ? "Hide the changes" : "Show the changes";
  });
  return details;
}

function errorBlock(lines) {
  return el("div", { class: "safecopy-error", role: "alert" }, lines.map((text) => el("div", { text })));
}

const heading = (text) => el("h3", { class: "safecopy-h", text });
const note = (text) => el("p", { class: "safecopy-note", text });

/** The undo section: the command, the scratch-copy line, and the git alternative when there is one. */
function undoNodes(revert) {
  if (!revert.command) return [];
  const nodes = [heading("Undo"), commandBlock(revert.command)];
  if (revert.verified) {
    nodes.push(note(`Tried on a scratch copy first: ${revert.files_checked} files came back byte for byte.`));
  }
  if (revert.git_restore) {
    nodes.push(note("Your folder is a clean git checkout, so this works too:"), commandBlock(revert.git_restore));
  }
  return nodes;
}

function deletionNodes(removals) {
  if (!removals.length) return [];
  return [
    heading("Deletions"),
    note("Kiwibro never deletes anything. These files were deleted in the copy; run these yourself if you want them gone:"),
    commandBlock(removals.map((entry) => entry.command).join("\n")),
  ];
}

function changeRow(change) {
  return el("div", { class: "safecopy-file" }, [
    el("span", { class: "safecopy-path", text: change.path }),
    el("span", { class: "safecopy-kind", text: KIND_TEXT[change.kind] || change.kind }),
    el("span", { class: "safecopy-lines", text: `+${change.added_lines} −${change.removed_lines}` }),
  ]);
}

/** The proof, top to bottom: summary and path, conflicts, files, diff, undo, deletions, left out. */
function proofNodes(proof, name, path) {
  const { totals, changes, conflicts, diff, revert, removals, left_out: leftOut } = proof;
  const nodes = [
    el("p", { class: "safecopy-summary", text: summaryLine(totals, name) }),
    el("p", { class: "safecopy-where", text: path }),
  ];
  if (conflicts.length) nodes.push(errorBlock([CONFLICT_TEXT, ...conflicts]));
  if (changes.length) nodes.push(el("div", { class: "safecopy-files" }, changes.map(changeRow)));
  if (conflicts.length) nodes.push(note(CONFLICT_NEXT));
  if (diff) nodes.push(diffDisclosure(diff));
  nodes.push(...undoNodes(revert), ...deletionNodes(removals));
  if (leftOut.length) {
    nodes.push(
      heading("Left out"),
      el(
        "div",
        { class: "safecopy-files" },
        leftOut.map((entry) =>
          el("div", { class: "safecopy-file" }, [
            el("span", { class: "safecopy-path", text: entry.path }),
            el("span", { class: "safecopy-kind", text: entry.reason }),
          ]),
        ),
      ),
    );
  }
  return nodes;
}

/**
 * Wires the strip, the review sheet, the End confirm and the start control.
 * `getConversationId()` gives the open chat (or null); `openChat(id)` opens a chat;
 * `reloadChat()` re-reads the open chat's messages.
 */
export function createSafeCopy({ api, getConversationId, openChat, reloadChat }) {
  const byId = (id) => document.getElementById(id);
  const strip = byId("safecopy-strip");
  const folderLine = byId("safecopy-folder");
  const countLine = byId("safecopy-count");
  const reviewButton = byId("safecopy-review");
  const endButton = byId("safecopy-end");
  const sheet = byId("safecopy-sheet");
  const reviewBody = byId("safecopy-body");
  const applyButton = byId("safecopy-apply");

  let seq = 0;
  let review = null; // { proof, real } of the proof on screen

  wireDialog(sheet);

  function showStrip(view) {
    const count = view.changed_files;
    folderLine.textContent = `Safe copy of ${folderName(view.real)}`;
    folderLine.title = shortPath(view.real);
    countLine.textContent = count ? `${files(count)} changed` : "No changes yet";
    countLine.dataset.changed = String(count > 0);
    reviewButton.classList.toggle("btn-primary", count > 0);
    reviewButton.classList.toggle("btn-ghost", count === 0);
    strip.hidden = false;
  }

  /** Reads the chat's session and shows or hides the strip. Late answers for another chat are ignored. */
  async function refresh() {
    const id = getConversationId();
    const mine = ++seq;
    if (!id) {
      strip.hidden = true;
      return;
    }
    try {
      const view = await api.safeCopyStatus(id);
      if (mine !== seq || id !== getConversationId()) return;
      if (view.active) showStrip(view);
      else strip.hidden = true;
    } catch {
      if (mine === seq) strip.hidden = true;
    }
  }

  function showProof(proof, real, notice = "") {
    review = { proof, real };
    const name = folderName(real);
    const path = shortPath(real);
    const writable = proof.changes.filter((change) => change.kind !== "deleted").length;
    reviewBody.replaceChildren(...(notice ? [errorBlock([notice])] : []), ...proofNodes(proof, name, path));
    applyButton.hidden = false;
    applyButton.title = `Apply to ${path}`;
    applyButton.replaceChildren(el("span", { class: "safecopy-apply-label", text: `Apply to ${name}` }));
    applyButton.disabled = proof.conflicts.length > 0 || writable === 0;
  }

  async function loadProof(notice = "") {
    const id = getConversationId();
    if (!id) return;
    applyButton.hidden = true;
    reviewBody.replaceChildren(el("div", { class: "safecopy-loading" }, [el("span", { class: "spinner", role: "status", "aria-label": "Loading" })]));
    try {
      const [view, { proof }] = await Promise.all([api.safeCopyStatus(id), api.safeCopyProof(id)]);
      if (id !== getConversationId()) return;
      showProof(proof, view.real, notice);
    } catch (error) {
      reviewBody.replaceChildren(errorBlock([error.message || "Couldn't load the proof."]));
    }
  }

  function openReview() {
    openSheet(sheet);
    loadProof();
  }

  function showApplied(applied) {
    const undo = el("button", { class: "btn btn-ghost safecopy-undo", type: "button", text: "Undo now" });
    undo.addEventListener("click", () => undoCopy(undo));
    reviewBody.replaceChildren(
      el("p", { class: "safecopy-summary", text: `Written: ${files(applied.written.length)}.` }),
      ...undoNodes(applied.revert),
      el("div", { class: "safecopy-row" }, [undo]),
      ...deletionNodes(applied.removals),
    );
    applyButton.hidden = true;
  }

  async function apply() {
    const id = getConversationId();
    if (!review || !id) return;
    setBusy(applyButton, true);
    try {
      const { applied } = await api.safeCopyApply(id, review.proof.fingerprint);
      showApplied(applied);
      refresh();
      reloadChat?.();
    } catch (error) {
      await loadProof(error.code === "stale" ? STALE_TEXT : error.message || "Couldn't apply the changes.");
    } finally {
      setBusy(applyButton, false);
    }
  }

  async function undoCopy(button) {
    const id = getConversationId();
    if (!id) return;
    setBusy(button, true);
    try {
      const { undo } = await api.safeCopyUndo(id);
      reviewBody.replaceChildren(
        el("p", { class: "safecopy-summary", text: `Put back: ${files(undo.restored.length)}.` }),
        ...(undo.kept.length
          ? [
              note("These files were added and are still there:"),
              commandBlock(undo.kept.map((entry) => entry.command).join("\n")),
            ]
          : []),
      );
      refresh();
      reloadChat?.();
    } catch (error) {
      reviewBody.prepend(errorBlock([error.message || "Couldn't undo the changes."]));
      setBusy(button, false);
    }
  }

  async function endCopy(discard, button) {
    const id = getConversationId();
    if (!id) return;
    setBusy(button, true);
    try {
      await api.safeCopyEnd(id, discard);
      closePopover();
      toast(discard ? "Safe copy discarded." : "Safe copy ended.");
      refresh();
    } catch (error) {
      toast(error.message || "Couldn't end the safe copy.", { tone: "error", iconName: "alert" });
    } finally {
      setBusy(button, false);
    }
  }

  /** The End confirm: the app's own confirm style, in a popover from the End button. */
  function askEnd() {
    const keep = el("button", { class: "btn btn-ghost", type: "button", text: "Keep the copy" });
    const discard = el("button", { class: "btn btn-danger", type: "button", text: "Discard the copy" });
    const cancel = el("button", { class: "btn btn-ghost", type: "button", text: "Cancel" });
    const content = el("div", { class: "confirm-row safecopy-end" }, [
      el("div", { class: "confirm-text", text: "End the safe copy?" }),
      el("div", { class: "confirm-text", text: "Changes that were not applied stay only in the copy." }),
      el("div", { class: "confirm-actions" }, [keep, discard, cancel]),
    ]);
    openPopover({ anchor: endButton, content, align: "end", label: "End the safe copy?" });
    keep.addEventListener("click", () => endCopy(false, keep));
    discard.addEventListener("click", () => endCopy(true, discard));
    cancel.addEventListener("click", () => closePopover());
    cancel.focus({ preventScroll: true });
  }

  reviewButton.addEventListener("click", openReview);
  endButton.addEventListener("click", askEnd);
  applyButton.addEventListener("click", apply);

  /**
   * The start control for Settings > Workspace: a button that opens a folder field, prefilled
   * with the workspace folder. Starting makes a new chat when none is open.
   */
  function starter({ folder = "", onStarted } = {}) {
    const open = el("button", { class: "btn btn-ghost", type: "button", text: "Work in a safe copy" });
    const input = el("input", {
      type: "text",
      value: folder,
      "aria-label": "Folder to copy",
      spellcheck: "false",
      autocomplete: "off",
    });
    const field = el("div", { class: "path-field" }, [icon("folder"), input]);
    const error = el("div", { class: "inline-error", hidden: true });
    const cancel = el("button", { class: "btn btn-ghost", type: "button", text: "Cancel" });
    const go = el("button", { class: "btn btn-primary", type: "submit", text: "Start the copy" });
    const form = el("form", { class: "safecopy-start-form", hidden: true }, [
      field,
      error,
      el("div", { class: "safecopy-row" }, [cancel, go]),
    ]);
    const section = el("div", { class: "safecopy-start" }, [open, form]);

    const id = getConversationId();
    if (id) {
      api
        .safeCopyStatus(id)
        .then((view) => {
          if (view.active) section.hidden = true;
        })
        .catch(() => {});
    }

    /** The open chat, or a new one created and opened first. */
    async function chatForCopy() {
      const current = getConversationId();
      if (current) return current;
      const created = await api.createConversation();
      const newId = created.conversation?.id || created.id;
      await openChat(newId);
      return newId;
    }

    open.addEventListener("click", () => {
      open.hidden = true;
      form.hidden = false;
      input.focus();
      input.select();
    });
    cancel.addEventListener("click", () => {
      error.hidden = true;
      form.hidden = true;
      open.hidden = false;
      open.focus();
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      error.hidden = true;
      setBusy(go, true);
      input.disabled = true;
      cancel.disabled = true;
      try {
        const chat = await chatForCopy();
        await api.safeCopyStart(chat, input.value.trim());
        toast("Safe copy started.");
        onStarted?.();
        refresh();
      } catch (err) {
        error.textContent = err.message || "The copy couldn't be made.";
        error.hidden = false;
      } finally {
        setBusy(go, false);
        input.disabled = false;
        cancel.disabled = false;
      }
    });
    return section;
  }

  return { refresh, starter };
}
