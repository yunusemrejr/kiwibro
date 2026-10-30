/**
 * Memory manager and Settings sheets.
 */

import {
  el,
  icon,
  clear,
  openSheet,
  debounce,
  toast,
  toolLabel,
  relativeTime,
  absoluteTime,
} from "./ui.js";

const KINDS = ["fact", "preference", "project", "person", "decision"];
const PERMISSION_ORDER = [
  "web_search",
  "multi_search",
  "fetch_page",
  "fetch_pages",
  "web_research",
  "browse_page",
  "browser",
  "read_file",
  "read_files",
  "list_files",
  "glob_files",
  "grep_files",
  "file_outline",
  "git",
  "write_file",
  "replace_in_file",
  "apply_patch",
  "file_manage",
  "run_command",
  "multi_command",
  "terminal_session",
  "run_python",
  "run_javascript",
  "system_info",
  "open_path",
  "clipboard",
  "notify",
  "screenshot",
  "model_info",
  "delegate",
  "describe_image",
  "study_file",
  "task_board",
  "recall",
  "remember",
  "calculate",
  "analyze_table",
  "doc_search",
  "doc_read",
  "doc_add",
  "doc_list",
  "convert_document",
  "remote",
  "workspace",
  "undo_changes",
];

const TOOL_PROFILES = [
  ["auto", "Auto (core set)"],
  ["full", "Full (all tools)"],
  ["minimal", "Minimal (classics)"],
  ["coding", "Coding"],
  ["research", "Research"],
  ["desktop", "Desktop"],
];

// Permissions: Automatic (smart / autonomous: the server's risk policy decides every
// step) or Manual (supervised: the per-tool choices below decide). See safety.decide.
const PERMISSION_STYLES = [
  ["automatic", "Automatic"],
  ["manual", "Manual"],
];
const RISKY_STEPS = [
  ["smart", "Ask me first"],
  ["autonomous", "Refuse them"],
];
const TOOL_PERMISSIONS = [
  ["auto", "Auto"],
  ["ask", "Ask"],
  ["deny", "Deny"],
];
// Where a memory came from (server: memories.origin).
const ORIGIN_LABELS = {
  you: "Added by you",
  chat: "Saved in a chat",
  learned: "Learned",
};
const SANDBOX_MODES = [
  ["auto", "On"],
  ["strict", "On, no network"],
  ["off", "Off"],
];

function titleCase(value) {
  return String(value || "").replace(/(^|\s)\S/g, (char) => char.toUpperCase());
}

function contextLabel(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "";
  return `${Math.round(number / 1024)}K`;
}

function fileSize(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function emptyState(iconName, title, detail) {
  const node = el("div", { class: "list-empty" }, [icon(iconName)]);
  if (title) node.append(el("div", { class: "list-empty-title", text: title }));
  if (detail) node.append(el("div", { text: detail }));
  return node;
}

/* ========================================================================== */
/* Memory                                                                     */
/* ========================================================================== */

export function createMemorySheet({ dialog, api, onChange, onOpenConversation }) {
  const countEl = dialog.querySelector("#memory-count");
  const searchEl = dialog.querySelector("#memory-search");
  const kindsEl = dialog.querySelector("#memory-kinds");
  const listEl = dialog.querySelector("#memory-list");
  const addForm = dialog.querySelector("#memory-add-form");
  const addInput = dialog.querySelector("#memory-new");
  const addKind = dialog.querySelector("#memory-new-kind");
  const addButton = addForm.querySelector('button[type="submit"]');
  const forgetButton = dialog.querySelector("#forget-all-btn");

  const state = { q: "", kind: "", memories: [], total: 0, editingId: null, loaded: false };
  let requestId = 0;
  let forgetTimer = 0;
  let forgetArmed = false;

  const notifyCount = () => {
    if (!state.q && !state.kind) onChange?.(state.total);
  };

  function resetForget() {
    window.clearTimeout(forgetTimer);
    forgetArmed = false;
    forgetButton.classList.remove("is-armed");
    forgetButton.textContent = "Forget all";
  }

  function renderKinds() {
    for (const chip of kindsEl.querySelectorAll(".kind-chip")) {
      chip.setAttribute("aria-pressed", String((chip.dataset.kind || "") === state.kind));
    }
  }

  function memoryRow(memory) {
    const item = el("div", {
      class: `memory-item${memory.pinned ? " is-pinned" : ""}${state.editingId === memory.id ? " is-editing" : ""}`,
      dataset: { id: memory.id },
    });

    if (state.editingId === memory.id) {
      const textarea = el("textarea", { rows: "3", "aria-label": "Edit memory" });
      textarea.value = memory.content;
      const kindSelect = el("select", { class: "memory-add-kind", "aria-label": "Memory kind" });
      for (const kind of KINDS) {
        const option = el("option", { value: kind, text: titleCase(kind) });
        if (kind === memory.kind) option.selected = true;
        kindSelect.append(option);
      }
      const save = el("button", { class: "btn btn-primary btn-compact", type: "button", text: "Save" });
      const cancel = el("button", { class: "btn btn-ghost btn-compact", type: "button", text: "Cancel" });
      const edit = el("div", { class: "memory-edit" }, [
        textarea,
        el("div", { class: "memory-edit-actions" }, [kindSelect, cancel, save]),
      ]);
      cancel.addEventListener("click", () => {
        state.editingId = null;
        render();
      });
      save.addEventListener("click", async () => {
        const content = textarea.value.trim();
        if (!content) {
          textarea.focus();
          return;
        }
        save.disabled = true;
        try {
          await api.patchMemory(memory.id, { content, kind: kindSelect.value });
          state.editingId = null;
          await refresh();
        } catch (error) {
          save.disabled = false;
          toast(error?.message || "Could not save memory", { tone: "error" });
        }
      });
      textarea.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          save.click();
        } else if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          cancel.click();
        }
      });
      item.append(edit);
      window.requestAnimationFrame(() => {
        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
      });
      return item;
    }

    const stamp = memory.updated_at || memory.created_at;
    const time = el("span", { class: "memory-time", "data-ts": stamp });
    time.textContent = relativeTime(stamp);
    time.title = absoluteTime(stamp);
    const originLabel = ORIGIN_LABELS[memory.origin];
    let origin = null;
    if (originLabel && memory.source_conversation_id && onOpenConversation) {
      origin = el("button", {
        class: "memory-origin is-link",
        type: "button",
        title: "Open the chat this came from",
        text: originLabel,
      });
      origin.addEventListener("click", () => onOpenConversation(memory.source_conversation_id));
    } else if (originLabel) {
      origin = el("span", { class: "memory-origin", text: originLabel });
    }
    const meta = el("div", { class: "memory-meta" }, [
      el("span", { class: "memory-kind", text: memory.kind || "fact" }),
      memory.pinned
        ? el("span", { class: "memory-pinned", title: "Pinned" }, [icon("pin"), el("span", { text: "Pinned" })])
        : null,
      el("span", { class: "memory-meta-dot" }),
      time,
      origin ? el("span", { class: "memory-meta-dot" }) : null,
      origin,
    ]);
    const main = el("div", { class: "memory-main" }, [
      el("div", { class: "memory-content", text: memory.content }),
      meta,
    ]);

    const pin = el(
      "button",
      {
        class: `memory-action${memory.pinned ? " is-active" : ""}`,
        type: "button",
        title: memory.pinned ? "Unpin memory" : "Pin memory",
        "aria-label": memory.pinned ? "Unpin memory" : "Pin memory",
        "aria-pressed": String(Boolean(memory.pinned)),
      },
      [icon("pin")],
    );
    pin.addEventListener("click", async () => {
      try {
        const result = await api.patchMemory(memory.id, { pinned: !memory.pinned });
        Object.assign(memory, result.memory || {});
        render();
      } catch (error) {
        toast(error?.message || "Could not update memory", { tone: "error" });
      }
    });

    const edit = el(
      "button",
      { class: "memory-action", type: "button", title: "Edit memory", "aria-label": "Edit memory" },
      [icon("pencil")],
    );
    edit.addEventListener("click", () => {
      state.editingId = memory.id;
      render();
    });

    const remove = el(
      "button",
      {
        class: "memory-action danger",
        type: "button",
        title: "Delete memory",
        "aria-label": "Delete memory",
      },
      [icon("trash")],
    );
    remove.addEventListener("click", async () => {
      try {
        await api.deleteMemory(memory.id);
        state.memories = state.memories.filter((entry) => entry.id !== memory.id);
        state.total = Math.max(0, state.total - 1);
        countEl.textContent = String(state.total);
        notifyCount();
        render();
      } catch (error) {
        toast(error?.message || "Could not delete memory", { tone: "error" });
      }
    });

    item.append(main, el("div", { class: "memory-actions" }, [pin, edit, remove]));
    return item;
  }

  function render() {
    renderKinds();
    countEl.textContent = String(state.total);
    clear(listEl);

    if (!state.loaded) {
      listEl.append(el("div", { class: "list-empty" }, [el("span", { class: "spinner" })]));
      return;
    }
    if (!state.memories.length) {
      listEl.append(
        emptyState(
          "sparkles",
          state.q || state.kind ? "No matches" : "Nothing remembered yet",
          state.q || state.kind
            ? "Try a different search or filter."
            : "Facts Kiwibro learns, and things you tell it to keep, appear here.",
        ),
      );
      return;
    }
    for (const memory of state.memories) listEl.append(memoryRow(memory));
    if (state.memories.length < state.total) {
      listEl.append(
        el("div", { class: "count-hint", style: { marginTop: "8px", textAlign: "center" } }, [
          `${state.memories.length} of ${state.total}`,
        ]),
      );
    }
  }

  async function refresh() {
    if (!dialog.open) return;
    const token = ++requestId;
    try {
      const data = await api.memories({ q: state.q, kind: state.kind });
      if (token !== requestId) return;
      state.memories = data.memories || [];
      state.total = Number.isFinite(data.total) ? data.total : state.memories.length;
      state.loaded = true;
      countEl.textContent = String(state.total);
      notifyCount();
      if (state.editingId && !state.memories.some((entry) => entry.id === state.editingId)) {
        state.editingId = null;
      }
      render();
    } catch (error) {
      if (token !== requestId) return;
      state.loaded = true;
      clear(listEl);
      listEl.append(emptyState("alert", "Couldn't load memory", error?.message || "Try again."));
      toast(error?.message || "Couldn't load memory", { tone: "error" });
    }
  }

  const debouncedSearch = debounce(() => refresh(), 240);
  searchEl.addEventListener("input", () => {
    state.q = searchEl.value.trim();
    debouncedSearch();
  });

  kindsEl.addEventListener("click", (event) => {
    const chip = event.target.closest(".kind-chip");
    if (!chip) return;
    state.kind = chip.dataset.kind || "";
    renderKinds();
    refresh();
  });

  addInput.addEventListener("input", () => {
    addButton.disabled = !addInput.value.trim();
  });

  addForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const content = addInput.value.trim();
    if (!content) return;
    addButton.disabled = true;
    addButton.textContent = "…";
    try {
      await api.createMemory({ content, kind: addKind.value });
      addInput.value = "";
      state.q = "";
      state.kind = "";
      searchEl.value = "";
      await refresh();
      addInput.focus();
      toast("Memory saved");
    } catch (error) {
      toast(error?.message || "Could not save memory", { tone: "error" });
    } finally {
      addButton.textContent = "Add";
      addButton.disabled = !addInput.value.trim();
    }
  });

  forgetButton.addEventListener("click", async () => {
    if (!forgetArmed) {
      forgetArmed = true;
      forgetButton.textContent = "Forget all?";
      forgetButton.classList.add("is-armed");
      forgetTimer = window.setTimeout(resetForget, 4000);
      return;
    }
    resetForget();
    forgetButton.disabled = true;
    try {
      await api.forgetMemories();
      state.memories = [];
      state.total = 0;
      state.loaded = true;
      countEl.textContent = "0";
      notifyCount();
      render();
      toast("All memories forgotten");
    } catch (error) {
      toast(error?.message || "Could not forget memories", { tone: "error" });
    } finally {
      forgetButton.disabled = false;
    }
  });

  const open = () => {
    openSheet(dialog);
    if (!state.loaded) render();
    refresh();
    window.requestAnimationFrame(() => {
      dialog.querySelector(".sheet-scroll")?.scrollTo({ top: 0 });
      searchEl.focus({ preventScroll: true });
    });
  };

  return {
    open,
    refreshIfOpen() {
      if (dialog.open) refresh();
    },
    resetForget,
  };
}

/* ========================================================================== */
/* Settings                                                                   */
/* ========================================================================== */

export function createSettingsSheet({ dialog, api, actions, isMock = false }) {
  const body = dialog.querySelector("#settings-body");
  const versionEl = dialog.querySelector("#settings-version");
  const footerUrlEl = dialog.querySelector("#settings-lm-url");

  const data = {
    settings: null,
    models: [],
    current: null,
    lmstudio: false,
    engine: "lmstudio",
    direct: null,
    health: null,
    workspace: null,
    dataDir: "",
    loading: false,
    error: null,
  };

  let forgetTimer = 0;
  let forgetArmed = false;
  let opening = false;
  let lastAutomatic = "smart"; // restored when switching back from Manual

  function resetForget() {
    window.clearTimeout(forgetTimer);
    forgetArmed = false;
  }

  async function load() {
    data.loading = true;
    data.error = null;
    render();
    const [settingsResult, modelsResult, healthResult, workspaceResult, engineResult] = await Promise.allSettled([
      api.settings(),
      api.models(),
      api.health(),
      api.workspace(),
      api.engine(),
    ]);

    if (settingsResult.status === "fulfilled") {
      data.settings = settingsResult.value.settings;
      data.dataDir = settingsResult.value.data_dir || "";
    } else {
      data.error = settingsResult.reason?.message || "Couldn't load settings.";
    }
    if (modelsResult.status === "fulfilled") {
      data.models = modelsResult.value.models || [];
      data.current = modelsResult.value.current || data.settings?.model || null;
      data.lmstudio = Boolean(modelsResult.value.lmstudio);
    }
    if (healthResult.status === "fulfilled") data.health = healthResult.value;
    if (workspaceResult.status === "fulfilled") data.workspace = workspaceResult.value;
    // If the engine call fails, the saved setting still says which engine is active.
    const engineData = engineResult.status === "fulfilled" ? engineResult.value : null;
    data.engine = engineData?.engine || data.settings?.engine || "lmstudio";
    data.direct = engineData?.direct || null;

    data.loading = false;
    render();
  }

  async function patch(patchObject, rollback) {
    try {
      const result = await actions.patchSettings(patchObject);
      if (result) data.settings = result;
      return true;
    } catch (error) {
      rollback?.();
      toast(error?.message || "Couldn't save settings", { tone: "error" });
      return false;
    }
  }

  function group(label) {
    return el("section", { class: "settings-group" }, [
      el("div", { class: "settings-label", text: label }),
    ]);
  }

  function modelRow(model, selected) {
    const dotTone = model.state === "loaded" ? "loaded" : "idle";
    const metaParts = [];
    if (model.context) metaParts.push(`${contextLabel(model.context)} context`);
    if (model.tools) metaParts.push("tools");
    if (model.vision) metaParts.push("vision");
    if (model.quantization) metaParts.push(model.quantization);
    if (model.state !== "loaded") metaParts.push("not loaded");

    const row = el(
      "button",
      {
        class: `model-row${selected ? " is-selected" : ""}`,
        type: "button",
        "aria-pressed": String(selected),
        title: model.id,
      },
      [
        el("span", { class: "status-dot", "data-tone": dotTone }),
        el("span", { class: "model-row-body" }, [
          el("span", { class: "model-row-name", text: model.name || model.id }),
          el("span", { class: "model-row-meta", text: metaParts.join(" · ") }),
        ]),
        icon("check", "model-row-check"),
      ],
    );
    row.addEventListener("click", async () => {
      if (selected) return;
      const ok = await actions.selectModel(model.id);
      if (ok) {
        data.current = model.id;
        render();
      }
    });
    return row;
  }

  function switchRow({ name, description, checked, onChange }) {
    const control = el("button", {
      class: "switch",
      type: "button",
      role: "switch",
      "aria-checked": String(Boolean(checked)),
      "aria-label": name,
    });
    control.addEventListener("click", () => {
      const next = control.getAttribute("aria-checked") !== "true";
      control.setAttribute("aria-checked", String(next));
      onChange(next, control);
    });
    return el("div", { class: "setting-row" }, [
      el("div", { class: "setting-text" }, [
        el("div", { class: "setting-name", text: name }),
        description ? el("div", { class: "setting-desc", text: description }) : null,
      ]),
      control,
    ]);
  }

  function segmented(label, options, mode, onChoose) {
    const group = el("div", {
      class: "segmented",
      role: "radiogroup",
      "aria-label": label,
    });
    const buttons = [];
    for (const [value, text] of options) {
      const button = el("button", {
        class: "segment",
        type: "button",
        role: "radio",
        "data-mode": value,
        "aria-checked": String(mode === value),
        text,
      });
      button.addEventListener("click", () => {
        if (button.getAttribute("aria-checked") === "true") return;
        const previous = mode;
        mode = value;
        for (const item of buttons) item.setAttribute("aria-checked", String(item === button));
        onChoose(value, previous);
      });
      buttons.push(button);
      group.append(button);
    }
    group.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      event.preventDefault();
      const activeIndex = buttons.findIndex((button) => button === document.activeElement);
      if (activeIndex === -1) return;
      const delta = event.key === "ArrowRight" ? 1 : -1;
      const next = buttons[(activeIndex + delta + buttons.length) % buttons.length];
      next.focus();
      next.click();
    });
    return group;
  }

  // LM Studio's address (LM Link models already come through this computer's server) and
  // an API token for servers with authentication on. The token is write-only here.
  function serverRows() {
    const urlInput = el("input", {
      class: "setting-input",
      type: "text",
      value: data.settings.lmstudio_url || "",
      "aria-label": "LM Studio server address",
      spellcheck: "false",
      autocomplete: "off",
    });
    urlInput.addEventListener("change", async () => {
      const previous = data.settings.lmstudio_url || "";
      const value = urlInput.value.trim();
      if (!value || value === previous) {
        urlInput.value = previous;
        return;
      }
      if (await patch({ lmstudio_url: value }, () => (urlInput.value = previous))) await load();
    });
    const tokenSet = Boolean(data.settings.lmstudio_token_set);
    const tokenInput = el("input", {
      class: "setting-input",
      type: "password",
      placeholder: tokenSet ? "Saved · type to replace" : "Not needed on this computer",
      "aria-label": "LM Studio API token",
      autocomplete: "off",
    });
    tokenInput.addEventListener("change", async () => {
      const value = tokenInput.value.trim();
      if (!value) return;
      tokenInput.value = "";
      if (await patch({ lmstudio_token: value })) await load();
    });
    const tokenControls = [tokenInput];
    if (tokenSet) {
      const remove = el("button", { class: "text-btn", type: "button", text: "Remove" });
      remove.addEventListener("click", async () => {
        if (await patch({ lmstudio_token: "" })) await load();
      });
      tokenControls.push(remove);
    }
    const row = (name, description, controls) =>
      el("div", { class: "setting-row" }, [
        el("div", { class: "setting-text" }, [
          el("div", { class: "setting-name", text: name }),
          el("div", { class: "setting-desc", text: description }),
        ]),
        ...controls,
      ]);
    return el("div", {}, [
      row("Server", "Models linked with LM Link appear here through this computer's LM Studio.", [urlInput]),
      row("API token", "Only for an LM Studio server that requires authentication.", tokenControls),
    ]);
  }

  // Direct mode: which llama.cpp build runs the models, where the model files are, and how
  // many CPU threads it may use. Each save reloads the sheet so it shows what was found.
  function directRows() {
    const direct = data.direct;
    const runtimeInput = el("input", {
      class: "setting-input",
      type: "text",
      value: data.settings.direct_runtime || "",
      placeholder: direct?.runtime?.path || "Automatic",
      "aria-label": "llama.cpp runtime path",
      spellcheck: "false",
      autocomplete: "off",
    });
    runtimeInput.addEventListener("change", async () => {
      const previous = data.settings.direct_runtime || "";
      const value = runtimeInput.value.trim();
      if (value === previous) {
        runtimeInput.value = previous;
        return;
      }
      if (await patch({ direct_runtime: value }, () => (runtimeInput.value = previous))) await load();
    });
    const modelsInput = el("input", {
      class: "setting-input",
      type: "text",
      value: data.settings.direct_models_dir || "",
      placeholder: direct?.model_dirs?.[0] || "Automatic",
      "aria-label": "Models folder",
      spellcheck: "false",
      autocomplete: "off",
    });
    modelsInput.addEventListener("change", async () => {
      const previous = data.settings.direct_models_dir || "";
      const value = modelsInput.value.trim();
      if (value === previous) {
        modelsInput.value = previous;
        return;
      }
      if (await patch({ direct_models_dir: value }, () => (modelsInput.value = previous))) await load();
    });
    const threadsInput = el("input", {
      class: "setting-input",
      type: "number",
      min: "0",
      max: "256",
      step: "1",
      value: data.settings.direct_threads || "",
      placeholder: `Auto (${direct?.threads ?? 0})`,
      "aria-label": "CPU threads",
    });
    threadsInput.addEventListener("change", async () => {
      const previous = data.settings.direct_threads || 0;
      const parsed = parseInt(threadsInput.value, 10);
      const threads = Number.isFinite(parsed) ? parsed : 0;
      if (threads === previous) return;
      if (await patch({ direct_threads: threads }, () => (threadsInput.value = previous || ""))) await load();
    });
    const row = (name, description, controls) =>
      el("div", { class: "setting-row" }, [
        el("div", { class: "setting-text" }, [
          el("div", { class: "setting-name", text: name }),
          el("div", { class: "setting-desc", text: description }),
        ]),
        ...controls,
      ]);
    const rows = [
      switchRow({
        name: "Graphics acceleration",
        description: direct?.gpu_available
          ? "Runs models on the GPU, an integrated one too: faster replies, and the processor stays free and cool."
          : "No GPU build of llama.cpp found a GPU here, so models run on the CPU.",
        checked: (data.settings.direct_gpu || "auto") !== "off",
        onChange: async (next, control) => {
          const ok = await patch({ direct_gpu: next ? "auto" : "off" }, () =>
            control.setAttribute("aria-checked", String(!next)),
          );
          if (ok) await load();
        },
      }),
      row("llama.cpp runtime", "Empty uses LM Studio's bundled engine. Set a path to use your own llama-server build.", [runtimeInput]),
      row("Models folder", "Empty scans LM Studio's models folders.", [modelsInput]),
      row("CPU threads", "Auto uses every physical core.", [threadsInput]),
    ];

    const servers = direct?.servers || [];
    if (servers.length) {
      const stop = el("button", { class: "text-btn", type: "button", text: "Stop" });
      stop.addEventListener("click", async () => {
        stop.disabled = true;
        try {
          await api.engineStop();
        } catch (error) {
          toast(error?.message || "Couldn't stop the engine", { tone: "error" });
        }
        await load();
      });
      const description = servers
        .map((server) => `${server.model} · ${Math.round(server.context_length / 1024)}K · ${server.gpu || `${server.threads} threads`}`)
        .join("; ");
      rows.push(row("Running", description, [stop]));
    }
    return el("div", {}, rows);
  }

  function workspaceSection() {
    const section = group("Workspace");
    const workspace = data.settings?.workspace || data.workspace?.root || "";
    const input = el("input", {
      type: "text",
      value: workspace,
      "aria-label": "Workspace folder",
      spellcheck: "false",
      autocomplete: "off",
    });
    const field = el("div", { class: "path-field" }, [icon("folder"), input]);
    const error = el("div", { class: "inline-error", hidden: true });
    section.append(field, error);

    const files = data.workspace?.files || [];
    const list = el("div", { class: "file-list" });
    for (const file of files.slice(0, 6)) {
      list.append(
        el("div", { class: "file-row", title: file.path }, [
          icon(file.type === "dir" ? "folder" : "file"),
          el("span", { class: "file-path", text: file.path }),
          el("span", { class: "file-size", text: file.type === "dir" ? "" : fileSize(file.size) }),
        ]),
      );
    }
    if (files.length > 6) {
      list.append(el("div", { class: "file-row" }, [el("span", { class: "file-path", text: `+${files.length - 6} more` })]));
    }
    if (!files.length) {
      section.append(el("div", { class: "inline-hint", text: "This folder is empty. Tools read and write only inside it." }));
    } else {
      section.append(list);
    }

    const save = async () => {
      const value = input.value.trim();
      if (!value || value === workspace) {
        input.value = workspace;
        return;
      }
      error.hidden = true;
      const result = await actions.patchSettings({ workspace: value }).catch((err) => {
        input.value = workspace;
        error.textContent = err?.message || "That folder can't be used.";
        error.hidden = false;
        return null;
      });
      if (!result) return;
      data.settings = result;
      const workspaceResult = await api.workspace().catch(() => null);
      if (workspaceResult) data.workspace = workspaceResult;
      render();
      toast("Workspace updated");
    };

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        input.blur();
      }
    });
    input.addEventListener("blur", save);
    return section;
  }

  function render() {
    // A re-render (e.g. switching Automatic/Manual) keeps the reader where they were.
    const scrollTop = body.scrollTop;
    clear(body);
    window.requestAnimationFrame(() => {
      body.scrollTop = scrollTop;
    });

    if (data.loading && !data.settings) {
      body.append(el("div", { class: "list-empty" }, [el("span", { class: "spinner" })]));
      return;
    }
    if (data.error) {
      body.append(emptyState("alert", "Couldn't load settings", data.error));
      const retry = el("button", { class: "text-btn", type: "button", text: "Try again", style: { display: "block", margin: "8px auto 0" } });
      retry.addEventListener("click", load);
      body.append(retry);
      return;
    }
    if (!data.settings) return;

    /* --- Models: LM Studio API or direct llama.cpp --- */
    const lmSection = group("Models");
    const modelsCount = data.models.length;
    const directMode = data.engine === "direct";
    const direct = data.direct;
    const directReady = Boolean(direct?.available && direct?.runtime);

    // The engine switch comes first: it decides what the status card below describes.
    lmSection.append(
      switchRow({
        name: "Optimized engine",
        description:
          "Runs LM Studio's downloaded models with Kiwibro's own llama.cpp setup: on the GPU when there is one, " +
          "thinking capped without re-reading the chat, and LM Studio doesn't need to be open.",
        checked: directMode,
        onChange: async (next, control) => {
          const ok = await patch({ engine: next ? "direct" : "lmstudio" }, () =>
            control.setAttribute("aria-checked", String(!next)),
          );
          if (ok) await load();
        },
      }),
    );

    let statusName;
    let statusSub;
    if (directMode) {
      const count = direct?.models_count || 0;
      statusName = directReady ? "Optimized engine ready" : "Optimized engine unavailable";
      statusSub = directReady
        ? [direct.runtime.label, direct.gpu, `${count} model${count === 1 ? "" : "s"}`].filter(Boolean).join(" · ")
        : direct?.error || "No llama.cpp runtime was found.";
    } else {
      statusName = data.lmstudio ? "LM Studio is running" : "LM Studio isn't running";
      statusSub = data.lmstudio
        ? `${data.settings.lmstudio_url} · ${modelsCount} model${modelsCount === 1 ? "" : "s"}`
        : "Start LM Studio, then reconnect.";
    }
    const statusCard = el("div", { class: "status-card" }, [
      el("span", { class: "status-dot", "data-tone": (directMode ? directReady : data.lmstudio) ? "loaded" : "warn" }),
      el("div", { class: "status-text" }, [
        el("div", { class: "status-name", text: statusName }),
        el("div", { class: "status-sub", text: statusSub }),
      ]),
      (() => {
        const refresh = el(
          "button",
          {
            class: "icon-btn menu-refresh",
            type: "button",
            title: directMode ? "Refresh" : "Reconnect",
            "aria-label": directMode ? "Refresh" : "Reconnect to LM Studio",
          },
          [icon("refresh")],
        );
        refresh.addEventListener("click", async () => {
          refresh.disabled = true;
          await load();
          refresh.disabled = false;
        });
        return refresh;
      })(),
    ]);
    lmSection.append(statusCard);

    if (data.lmstudio && modelsCount) {
      const modelList = el("div", { class: "model-list" });
      for (const model of data.models) modelList.append(modelRow(model, model.id === data.current));
      lmSection.append(modelList);
    } else if (!data.models.length) {
      const emptyText = directMode
        ? "No downloaded models were found in LM Studio's models folder."
        : data.lmstudio
          ? "No chat models are available in LM Studio."
          : "Models will appear here once LM Studio is reachable.";
      lmSection.append(el("div", { class: "model-empty", text: emptyText }));
    }
    lmSection.append(directMode ? directRows() : serverRows());
    body.append(lmSection);

    // One select row per enum setting; a failed save puts the old value back.
    const choiceRow = (key, name, description, choices) => {
      const current = data.settings[key] || choices[0][0];
      const select = el("select", { class: "setting-select", "aria-label": name });
      for (const [value, label] of choices) {
        const option = el("option", { value, text: label });
        if (value === current) option.selected = true;
        select.append(option);
      }
      select.addEventListener("change", async () => {
        const previous = data.settings[key] || choices[0][0];
        await patch({ [key]: select.value }, () => {
          select.value = previous;
        });
      });
      return el("div", { class: "setting-row" }, [
        el("div", { class: "setting-text" }, [
          el("div", { class: "setting-name", text: name }),
          el("div", { class: "setting-desc", text: description }),
        ]),
        select,
      ]);
    };

    /* --- Permissions --- */
    // Automatic: the risk policy decides each step and the per-tool list stays out of the
    // way. Manual: the per-tool choices decide, and only then are they shown.
    const permSection = group("Permissions");
    const autonomy = data.settings.autonomy || "smart";
    const manual = autonomy === "supervised";
    permSection.append(
      el("div", { class: "setting-row" }, [
        el("div", { class: "setting-text" }, [
          el("div", { class: "setting-name", text: "Tool use" }),
          el("div", {
            class: "setting-desc",
            text: manual
              ? "You decide, tool by tool, what runs on its own and what asks first."
              : "Safe and undoable work runs on its own. Dangerous commands are always refused.",
          }),
        ]),
        segmented("Permission style", PERMISSION_STYLES, manual ? "manual" : "automatic", async (value) => {
          if (value === "manual") lastAutomatic = autonomy;
          await patch({ autonomy: value === "manual" ? "supervised" : lastAutomatic });
          render();
        }),
      ]),
    );
    if (!manual) {
      permSection.append(
        choiceRow(
          "autonomy",
          "Risky steps",
          "Writing outside the workspace, publishing (git push), stopping other programs.",
          RISKY_STEPS,
        ),
      );
    } else {
      const permissions = data.settings.permissions || {};
      const keys = [
        ...PERMISSION_ORDER.filter((key) => key in permissions),
        ...Object.keys(permissions).filter((key) => !PERMISSION_ORDER.includes(key)),
      ];
      for (const key of keys) {
        const row = el("div", { class: "setting-row" }, [
          el("div", { class: "setting-text" }, [
            el("div", { class: "setting-name", text: toolLabel(key) }),
          ]),
          segmented(`${toolLabel(key)} permission`, TOOL_PERMISSIONS, permissions[key] || "ask", async (mode, previous) => {
            const next = { ...permissions, [key]: mode };
            permissions[key] = mode;
            const ok = await actions
              .patchSettings({ permissions: next })
              .catch((error) => {
                permissions[key] = previous;
                render();
                toast(error?.message || "Couldn't change permission", { tone: "error" });
                return null;
              });
            if (ok) data.settings = ok;
          }),
        ]);
        permSection.append(row);
      }
    }
    body.append(permSection);

    /* --- Behaviour --- */
    const behaviour = group("Behaviour");
    behaviour.append(
      switchRow({
        name: "Agent mode",
        description: "Let Kiwibro use tools and act on your machine.",
        checked: data.settings.agent_enabled,
        onChange: async (next, control) => {
          const ok = await patch({ agent_enabled: next }, () => {
            control.setAttribute("aria-checked", String(!next));
          });
          void ok;
        },
      }),
      switchRow({
        name: "Automatic memory",
        description: "Learn lasting facts from your chats; you can edit or remove any of them.",
        checked: data.settings.memory_enabled,
        onChange: async (next, control) => {
          await patch({ memory_enabled: next }, () => {
            control.setAttribute("aria-checked", String(!next));
          });
        },
      }),
    );
    behaviour.append(
      choiceRow(
        "sandbox",
        "Command sandbox",
        "Commands and code run in a sealed box: read-only system, your workspace writable, credentials hidden.",
        SANDBOX_MODES,
      ),
      choiceRow(
        "tool_profile",
        "Tool profile",
        "Which tools the model sees. Auto offers one fixed core set sized for small models; others run when called by name.",
        TOOL_PROFILES,
      ),
    );
    // Servers the `remote` tool may use without asking (plus your ~/.ssh/config hosts).
    const hostsText = () => (data.settings.remote_hosts || []).join(", ");
    const hostsInput = el("input", {
      class: "setting-input",
      type: "text",
      value: hostsText(),
      placeholder: "prod, deploy@10.0.0.5",
      "aria-label": "Remote hosts",
      spellcheck: "false",
    });
    hostsInput.addEventListener("change", async () => {
      const previous = hostsText();
      const hosts = hostsInput.value.split(/[\s,]+/).filter(Boolean);
      await patch({ remote_hosts: hosts }, () => {
        hostsInput.value = previous;
      });
    });
    behaviour.append(
      el("div", { class: "setting-row" }, [
        el("div", { class: "setting-text" }, [
          el("div", { class: "setting-name", text: "Remote hosts" }),
          el("div", {
            class: "setting-desc",
            text: "Servers Kiwibro may work on over SSH. Hosts from your ~/.ssh/config are trusted too.",
          }),
        ]),
        hostsInput,
      ]),
    );
    const roundsInput = el("input", {
      class: "setting-input",
      type: "number",
      min: "1",
      max: "32",
      step: "1",
      value: String(data.settings.max_tool_rounds || 16),
      "aria-label": "Max tool steps per reply",
      style: { width: "72px" },
    });
    roundsInput.addEventListener("change", async () => {
      const previous = String(data.settings.max_tool_rounds || 16);
      const value = Number(roundsInput.value);
      if (!Number.isInteger(value) || value < 1 || value > 32) {
        roundsInput.value = previous;
        toast("Tool steps must be between 1 and 32", { tone: "error" });
        return;
      }
      await patch({ max_tool_rounds: value }, () => {
        roundsInput.value = previous;
      });
    });
    behaviour.append(
      el("div", { class: "setting-row" }, [
        el("div", { class: "setting-text" }, [
          el("div", { class: "setting-name", text: "Max tool steps per reply" }),
          el("div", {
            class: "setting-desc",
            text: "How many tool rounds one reply may take before Kiwibro wraps up.",
          }),
        ]),
        roundsInput,
      ]),
    );
    body.append(behaviour);

    /* --- Workspace --- */
    body.append(workspaceSection());

    /* --- Data --- */
    const dataSection = group("Data");
    dataSection.append(
      el("div", { class: "setting-row" }, [
        el("div", { class: "setting-text" }, [
          el("div", { class: "setting-name", text: "Local database" }),
          el("div", {
            class: "setting-desc",
            text: data.dataDir || "~/.local/share/kiwibro",
          }),
        ]),
        el("span", { class: "status-dot", "data-tone": "loaded" }),
      ]),
      el("div", { class: "inline-hint", text: "Chats, memories and settings never leave this machine." }),
      el("div", { class: "setting-row" }, [
        el("div", { class: "setting-text" }, [
          el("div", { class: "setting-name danger", text: "Forget everything Kiwibro remembers" }),
        ]),
        (() => {
          const button = el("button", { class: "text-btn danger", type: "button", text: "Forget all" });
          button.addEventListener("click", async () => {
            if (!forgetArmed) {
              forgetArmed = true;
              button.textContent = "Forget all?";
              forgetTimer = window.setTimeout(() => {
                resetForget();
                button.textContent = "Forget all";
              }, 4000);
              return;
            }
            resetForget();
            button.textContent = "Forget all";
            try {
              const result = await actions.forgetMemories();
              toast(Number.isFinite(result) ? `Forgot ${result} memories` : "Memories forgotten");
              await load();
            } catch (error) {
              toast(error?.message || "Couldn't forget memories", { tone: "error" });
            }
          });
          return button;
        })(),
      ]),
    );
    body.append(dataSection);

    versionEl.textContent = data.health?.version ? `Kiwibro ${data.health.version}` : "Kiwibro";
    footerUrlEl.textContent = isMock ? "Mock mode" : "Local only";
  }

  async function open() {
    openSheet(dialog);
    body.scrollTop = 0;
    if (opening) return;
    opening = true;
    try {
      if (data.settings) render();
      await load();
    } finally {
      opening = false;
    }
  }

  return { open, refreshIfOpen: () => dialog.open && load() };
}

/* Exported for the header/settings model lists. */
export function modelMetaLine(model) {
  const parts = [];
  if (model.context) parts.push(`${contextLabel(model.context)} context`);
  if (model.tools) parts.push("tools");
  if (model.vision) parts.push("vision");
  if (model.quantization) parts.push(model.quantization);
  if (model.state !== "loaded") parts.push("not loaded");
  return parts.join(" · ");
}
