/**
 * Kiwibro frontend controller: bootstrap, conversation lifecycle, chat stream,
 * composer, sidebar, model switching, sheets and keyboard shortcuts.
 */

import { api, IS_MOCK, consumeSSE, chatError } from "./api.js";
import { createChatView } from "./chat.js";
import { createInferenceUI } from "./inference.js";
import { createMemorySheet, createSettingsSheet, modelMetaLine } from "./sheets.js";
import {
  $,
  $$,
  el,
  icon,
  clear,
  openPopover,
  closePopover,
  popoverIsOpen,
  wireDialog,
  toast,
  announce,
  relativeTime,
  absoluteTime,
  updateRelativeTimes,
  formatTokens,
} from "./ui.js";

const els = {
  app: $("#app"),
  sidebar: $("#sidebar"),
  sidebarToggle: $("#sidebar-toggle"),
  brandHome: $("#brand-home"),
  modelPill: $("#model-pill"),
  modelDot: $("#model-dot"),
  modelPillName: $("#model-pill-name"),
  memoryBtn: $("#memory-btn"),
  memoryBadge: $("#memory-badge"),
  settingsBtn: $("#settings-btn"),
  newChat: $("#new-chat"),
  convSearch: $("#conv-search"),
  convList: $("#conv-list"),
  convEmpty: $("#conv-empty"),
  sidebarMemory: $("#sidebar-memory"),
  sidebarMemoryCount: $("#sidebar-memory-count"),
  sidebarModel: $("#sidebar-model"),
  sidebarModelName: $("#sidebar-model-name"),
  sidebarDot: $("#sidebar-dot"),
  transcript: $("#transcript"),
  chat: $("#chat"),
  empty: $("#empty-state"),
  scrim: $("#scrim"),
  scrollBottom: $("#scroll-bottom"),
  dockHint: $("#dock-hint"),
  dockHintText: $("#dock-hint-text"),
  dockRetry: $("#dock-retry"),
  composer: $("#composer"),
  input: $("#composer-input"),
  agentToggle: $("#agent-toggle"),
  contextRing: $("#context-ring"),
  ringValue: $("#ring-value"),
  sendBtn: $("#send-btn"),
  stopBtn: $("#stop-btn"),
  memorySheet: $("#memory-sheet"),
  settingsSheet: $("#settings-sheet"),
};

const STORAGE = {
  conversation: "kiwibro.conversation",
  sidebar: "kiwibro.sidebar",
};

const PHASE_LABELS = {
  thinking: "Thinking",
  web: "Searching the web",
  tool: "Using a tool",
  reading: "Reading",
  writing: "Writing",
  running: "Working",
};

const mobileQuery = window.matchMedia("(max-width: 860px)");

const state = {
  booted: false,
  unreachable: false,
  health: null,
  settings: null,
  models: [],
  currentModel: null,
  conversations: [],
  convFilter: "",
  conversationId: null,
  conversation: null,
  memoriesCount: 0,
  workspace: "",
  dataDir: "",
  streaming: false,
  controller: null,
  lastSend: null,
  drawerOpen: false,
};

const backgroundStreams = new Map();
let currentStream = null;

let renamingId = null;
let bootRetryTimer = 0;
let conversationRefreshTimer = 0;
let healthCheckAt = 0;
let memoryRefreshTimer = 0;
let memorySheet = null;
let settingsSheet = null;

/* -------------------------------------------------------------------------- */
/* Chat view                                                                  */
/* -------------------------------------------------------------------------- */

const chatView = createChatView({
  root: els.chat,
  transcript: els.transcript,
  resolveApproval: (approvalId, decision) => api.resolveApproval(approvalId, decision),
  onRetry: () => retryLastSend(),
  onScrollState: (pinned) => {
    const hasContent = !els.chat.classList.contains("is-empty");
    els.scrollBottom.hidden = pinned || !hasContent;
  },
});

const inferenceUI = createInferenceUI({
  agentToggle: els.agentToggle,
  contextRing: els.contextRing,
  transcript: els.transcript,
  getState: () => state,
  patchSettings: (patch) => patchSettingsAndSync(patch),
  onModelsChanged: () => refreshModels({ quiet: true }),
});

/* -------------------------------------------------------------------------- */
/* Bootstrap                                                                  */
/* -------------------------------------------------------------------------- */

function applyBootstrap(data) {
  state.health = data.health || null;
  state.settings = data.settings || null;
  state.models = Array.isArray(data.models) ? data.models : [];
  state.currentModel =
    data.current_model ||
    state.settings?.model ||
    state.models.find((model) => model.state === "loaded")?.id ||
    state.models[0]?.id ||
    null;
  state.conversations = Array.isArray(data.conversations) ? data.conversations : [];
  state.memoriesCount = Number(data.memories_count) || 0;
  state.workspace = data.workspace || state.settings?.workspace || "";
  state.lmstudio = Boolean(state.health?.lmstudio);
  state.unreachable = false;
}

async function boot() {
  window.clearTimeout(bootRetryTimer);
  try {
    const data = await api.bootstrap();
    state.booted = true;
    applyBootstrap(data);
    renderAll();
    await restoreConversation();
    focusComposerSoon();
  } catch (error) {
    state.booted = false;
    state.unreachable = true;
    state.lmstudio = false;
    state.models = [];
    renderOfflineState();
    if (!state.retryNotified) {
      state.retryNotified = true;
      toast("Can't reach the Kiwibro server", { tone: "error", iconName: "alert", timeout: 5000 });
    }
    window.clearTimeout(bootRetryTimer);
    bootRetryTimer = window.setTimeout(() => {
      if (state.unreachable) boot();
    }, 5000);
    void error;
  }
}

function isVisibleConversation(conversation) {
  return (Number(conversation?.message_count) || 0) > 0;
}

async function restoreConversation() {
  const saved = localStorage.getItem(STORAGE.conversation);
  if (saved && state.conversations.some((conversation) => conversation.id === saved && isVisibleConversation(conversation))) {
    const opened = await openConversation(saved, { silent: true });
    if (opened) return;
  }
  if (saved) localStorage.removeItem(STORAGE.conversation);
  showEmptyState();
}

function renderAll() {
  renderHeaderModel();
  renderSidebar();
  renderMemoryCounts();
  applySettingsUI();
  updateComposerState();
  updateContext(null);
}

function renderOfflineState() {
  renderHeaderModel();
  renderSidebar();
  renderMemoryCounts();
  updateComposerState();
  updateContext(null);
}

/* -------------------------------------------------------------------------- */
/* Header & model picker                                                      */
/* -------------------------------------------------------------------------- */

function renderHeaderModel() {
  inferenceUI.applySettings(state.settings);
  let name = "Kiwibro";
  let tone = "idle";
  let title = "Switch model";

  if (state.unreachable) {
    name = "Offline";
    tone = "warn";
    title = "Kiwibro server is unreachable";
  } else if (!state.lmstudio) {
    name = "LM Studio offline";
    tone = "warn";
    title = "Start LM Studio to chat";
  } else if (!state.models.length) {
    name = "No model";
    tone = "amber";
    title = "No chat model loaded in LM Studio";
  } else if (state.currentModel) {
    name = state.currentModel;
    tone = "loaded";
    title = `${state.currentModel} — switch model`;
  }

  els.modelPillName.textContent = name;
  els.modelDot.dataset.tone = tone;
  els.modelPill.title = title;
  els.sidebarModelName.textContent = state.unreachable
    ? "Offline"
    : !state.lmstudio
      ? "LM Studio off"
      : state.currentModel || "No model";
  els.sidebarDot.dataset.tone = tone;
}

function buildModelMenu(menu) {
  clear(menu);
  menu.append(el("div", { class: "menu-label", text: "Model" }));

  if (!state.lmstudio) {
    menu.append(
      el("div", { class: "menu-empty" }, [
        icon("alert"),
        el("div", { text: "LM Studio isn't running." }),
        el("div", { text: "Start it, then refresh." }),
      ]),
    );
  } else if (!state.models.length) {
    menu.append(
      el("div", { class: "menu-empty" }, [
        icon("info"),
        el("div", { text: "No chat models available." }),
      ]),
    );
  } else {
    for (const model of state.models) {
      const selected = model.id === state.currentModel;
      const row = el(
        "button",
        {
          class: `menu-item${selected ? " is-selected" : ""}`,
          type: "button",
          role: "menuitemradio",
          "aria-checked": String(selected),
          "data-model-id": model.id,
          title: model.id,
        },
        [
          el("span", {
            class: "status-dot",
            "data-tone": model.state === "loaded" ? "loaded" : "idle",
          }),
          el("span", { class: "menu-item-body" }, [
            el("span", { class: "menu-item-name", text: model.name || model.id }),
            el("span", { class: "menu-item-meta", text: modelMetaLine(model) }),
          ]),
          icon("check", "menu-item-check"),
        ],
      );
      row.addEventListener("click", async () => {
        closePopover({ restoreFocus: false });
        const ok = await selectModel(model.id);
        if (ok) focusComposerSoon();
      });
      menu.append(row);
    }

    const loaded = state.models.filter((model) => model.state === "loaded").length;
    const refresh = el(
      "button",
      {
        class: "icon-btn menu-refresh",
        type: "button",
        title: "Refresh model list",
        "aria-label": "Refresh model list",
      },
      [icon("refresh")],
    );
    refresh.addEventListener("click", async () => {
      refresh.disabled = true;
      await refreshModels({ quiet: true });
      buildModelMenu(menu);
      menu.querySelector(".menu-item")?.focus({ preventScroll: true });
    });
    menu.append(
      el("div", { class: "menu-sep" }),
      el("div", { class: "menu-foot" }, [
        el("span", {
          class: "menu-foot-text",
          text: `${loaded} loaded · ${state.models.length} available`,
        }),
        refresh,
      ]),
    );
  }

  menu.addEventListener("keydown", moveMenuFocus);
}

function moveMenuFocus(event) {
  if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
  const items = $$(".menu-item", event.currentTarget).filter((item) => !item.disabled);
  if (!items.length) return;
  event.preventDefault();
  const index = items.indexOf(document.activeElement);
  let next;
  if (event.key === "Home") next = items[0];
  else if (event.key === "End") next = items[items.length - 1];
  else {
    const delta = event.key === "ArrowDown" ? 1 : -1;
    next = items[(index + delta + items.length) % items.length];
  }
  next.focus({ preventScroll: true });
}

function toggleModelPopover() {
  if (popoverIsOpen()) {
    closePopover();
    return;
  }
  if (state.unreachable) {
    toast("The Kiwibro server isn't reachable", { tone: "error", iconName: "alert" });
    return;
  }
  const menu = el("div", { class: "menu", role: "menu", "aria-label": "Model" });
  buildModelMenu(menu);
  els.modelPill.setAttribute("aria-expanded", "true");
  openPopover({
    anchor: els.modelPill,
    content: menu,
    align: "center",
    label: "Choose model",
    onClose: () => els.modelPill.setAttribute("aria-expanded", "false"),
  });
}

async function selectModel(id) {
  if (!id) return true;
  if (id === state.currentModel) {
    renderHeaderModel();
    return true;
  }
  try {
    const result = await api.selectModel(id);
    state.currentModel = result.current || id;
    if (state.settings) state.settings.model = state.currentModel;
    renderHeaderModel();
    toast(`Using ${state.currentModel}`, { iconName: "sparkles" });
    announce(`Model switched to ${state.currentModel}`);
    return true;
  } catch (error) {
    toast(error?.message || "Couldn't switch model", { tone: "error", iconName: "alert" });
    return false;
  }
}

async function refreshModels({ quiet = false } = {}) {
  try {
    const data = await api.models();
    state.models = Array.isArray(data.models) ? data.models : [];
    state.currentModel = data.current || state.currentModel;
    state.lmstudio = Boolean(data.lmstudio);
    renderHeaderModel();
    updateComposerState();
    return true;
  } catch (error) {
    if (!quiet) toast(error?.message || "Couldn't refresh models", { tone: "error" });
    return false;
  }
}

/* -------------------------------------------------------------------------- */
/* Sidebar                                                                    */
/* -------------------------------------------------------------------------- */

function renderSidebar() {
  clear(els.convList);
  const q = (state.convFilter || "").trim().toLowerCase();
  const conversations = state.conversations
    .filter(isVisibleConversation)
    .filter((conversation) => {
      if (!q) return true;
      const title = (conversation.title || "").toLowerCase();
      const model = (conversation.model || "").toLowerCase();
      return title.includes(q) || model.includes(q);
    });
  els.convEmpty.textContent = q ? "No matching chats." : "No chats yet.";
  els.convEmpty.hidden = conversations.length > 0;

  for (const conversation of conversations) {
    const isStreaming =
      backgroundStreams.has(conversation.id) ||
      (state.streaming && state.conversationId === conversation.id) ||
      Boolean(conversation.busy);

    const row = el("div", {
      class: `conv-row${conversation.id === state.conversationId ? " is-active" : ""}${isStreaming ? " is-streaming" : ""}`,
      dataset: { id: conversation.id },
    });

    if (renamingId === conversation.id) {
      const input = el("input", {
        type: "text",
        value: conversation.title || "",
        "aria-label": "Chat title",
        maxlength: "120",
      });
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          commitRename(conversation, input.value);
        } else if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          renamingId = null;
          renderSidebar();
        }
      });
      input.addEventListener("blur", () => {
        if (renamingId === conversation.id) commitRename(conversation, input.value);
      });
      row.append(el("div", { class: "conv-rename" }, [input]));
      els.convList.append(row);
      window.requestAnimationFrame(() => {
        input.focus({ preventScroll: true });
        input.select();
      });
      continue;
    }

    const open = el("button", {
      class: "conv-open",
      type: "button",
      title: conversation.title,
      "aria-current": conversation.id === state.conversationId ? "page" : "false",
    });
    open.append(el("span", { class: "conv-title", text: conversation.title || "New chat" }));
    if (isStreaming) {
      open.append(
        el("span", {
          class: "status-dot conv-running-dot",
          "data-tone": "loaded",
          "aria-label": "Replying in background",
          title: "Replying in background",
        }),
      );
    }
    if (conversation.pinned) open.append(icon("pin", "conv-pin"));
    const time = el("span", { class: "conv-time", "data-ts": conversation.updated_at });
    time.textContent = relativeTime(conversation.updated_at);
    time.title = absoluteTime(conversation.updated_at);
    open.append(time);
    open.addEventListener("click", () => {
      closeDrawer();
      openConversation(conversation.id);
    });

    const menuButton = el(
      "button",
      {
        class: "conv-menu-btn",
        type: "button",
        title: "Chat options",
        "aria-label": `Chat options for ${conversation.title || "chat"}`,
      },
      [icon("more")],
    );
    menuButton.addEventListener("click", (event) => {
      event.stopPropagation();
      showConversationMenu(conversation, menuButton, row);
    });

    row.append(open, menuButton);
    els.convList.append(row);
  }
}

function showConversationMenu(conversation, anchor, row) {
  row.classList.add("is-menu-open");
  const menu = el("div", { class: "menu", role: "menu", "aria-label": "Chat options" });
  const addItem = (iconName, label, handler, className = "") => {
    const button = el("button", {
      class: `menu-item${className}`,
      type: "button",
      role: "menuitem",
    }, [icon(iconName), el("span", { text: label })]);
    button.addEventListener("click", handler);
    menu.append(button);
  };

  menu.append(el("div", { class: "menu-label", text: "Chat" }));
  addItem("pin", conversation.pinned ? "Unpin" : "Pin", () => {
    closePopover({ restoreFocus: false });
    togglePin(conversation);
  });
  addItem("pencil", "Rename", () => {
    closePopover({ restoreFocus: false });
    renamingId = conversation.id;
    renderSidebar();
  });
  addItem("copy", "Fork chat", async () => {
    closePopover({ restoreFocus: false });
    try {
      const data = await api.forkConversation(conversation.id);
      await refreshConversations({ silent: true });
      const forkedId = data?.conversation?.id || data?.id;
      if (forkedId) {
        await openConversation(forkedId);
      }
      toast("Chat forked");
    } catch (error) {
      toast(error?.message || "Couldn't fork chat", { tone: "error" });
    }
  });
  addItem("sparkles", "Compact context", async () => {
    closePopover({ restoreFocus: false });
    try {
      await api.compactConversation(conversation.id);
      if (state.conversationId === conversation.id) {
        await openConversation(conversation.id, { silent: true });
      }
      toast("Context compacted");
    } catch (error) {
      toast(error?.message || "Couldn't compact context", { tone: "error" });
    }
  });
  menu.append(el("div", { class: "menu-sep" }));
  addItem(
    "trash",
    "Delete",
    () => showDeleteConfirm(menu, conversation),
    " menu-danger",
  );
  menu.addEventListener("keydown", moveMenuFocus);

  openPopover({
    anchor,
    content: menu,
    align: "end",
    label: "Chat options",
    onClose: () => row.classList.remove("is-menu-open"),
  });
}

function showDeleteConfirm(menu, conversation) {
  const cancel = el("button", { class: "btn btn-ghost btn-compact", type: "button", text: "Cancel" });
  const confirm = el("button", {
    class: "btn btn-primary btn-compact",
    type: "button",
    text: "Delete",
  });
  menu.replaceChildren(
    el("div", { class: "confirm-row" }, [
      el("div", {
        class: "confirm-text",
        text: `Delete “${conversation.title || "this chat"}”?`,
      }),
      el("div", { class: "confirm-actions" }, [cancel, confirm]),
    ]),
  );
  cancel.addEventListener("click", () => closePopover());
  confirm.addEventListener("click", async () => {
    confirm.disabled = true;
    cancel.disabled = true;
    await deleteConversation(conversation);
  });
  cancel.focus({ preventScroll: true });
}

async function togglePin(conversation) {
  try {
    const data = await api.patchConversation(conversation.id, { pinned: !conversation.pinned });
    if (data?.conversation) Object.assign(conversation, data.conversation);
    else conversation.pinned = !conversation.pinned;
    await refreshConversations({ silent: true });
  } catch (error) {
    toast(error?.message || "Couldn't update chat", { tone: "error" });
  }
}

async function deleteConversation(conversation) {
  try {
    if (backgroundStreams.has(conversation.id)) {
      backgroundStreams.get(conversation.id)?.controller?.abort();
      backgroundStreams.delete(conversation.id);
    }
    await api.deleteConversation(conversation.id);
    closePopover();
    state.conversations = state.conversations.filter((entry) => entry.id !== conversation.id);
    if (state.conversationId === conversation.id) {
      state.conversationId = null;
      state.conversation = null;
      localStorage.removeItem(STORAGE.conversation);
      chatView.reset();
      updateContext(null);
    }
    renderSidebar();
    updateComposerState();
    toast("Chat deleted");
    refreshConversations({ silent: true });
  } catch (error) {
    toast(error?.message || "Couldn't delete chat", { tone: "error" });
  }
}

async function commitRename(conversation, rawValue) {
  const title = String(rawValue || "").trim();
  renamingId = null;
  if (!title || title === conversation.title) {
    renderSidebar();
    return;
  }
  try {
    const data = await api.patchConversation(conversation.id, { title });
    conversation.title = data?.conversation?.title || title;
    if (state.conversation?.id === conversation.id) state.conversation.title = conversation.title;
    renderSidebar();
  } catch (error) {
    renderSidebar();
    toast(error?.message || "Couldn't rename chat", { tone: "error" });
  }
}

async function refreshConversations({ silent = true } = {}) {
  try {
    const data = await api.conversations();
    state.conversations = Array.isArray(data.conversations) ? data.conversations : [];
    renderSidebar();
  } catch (error) {
    if (!silent) toast(error?.message || "Couldn't load chats", { tone: "error" });
  }
}

function scheduleConversationRefresh() {
  window.clearTimeout(conversationRefreshTimer);
  conversationRefreshTimer = window.setTimeout(() => {
    refreshConversations({ silent: true });
  }, 250);
}

function updateConversationTitle(id, title) {
  const conversation = state.conversations.find((entry) => entry.id === id);
  if (conversation) {
    conversation.title = title;
    conversation.updated_at = new Date().toISOString();
  }
  if (state.conversation?.id === id) state.conversation.title = title;
  renderSidebar();
}

/* -------------------------------------------------------------------------- */
/* Conversations                                                              */
/* -------------------------------------------------------------------------- */

function detachCurrentStream() {
  if (!state.streaming || !currentStream) return;
  currentStream.isForeground = false;
  const cid = currentStream.conversationId || state.conversationId;
  if (cid) {
    const conv = state.conversations.find((c) => c.id === cid);
    const title = conv?.title || state.conversation?.title || "Chat";
    backgroundStreams.set(cid, {
      controller: currentStream.controller,
      title,
      status: "running",
      needsApproval: false,
    });
  }
  state.streaming = false;
  state.controller = null;
  setStreamingUI(false);
  renderSidebar();
}

async function openConversation(id, { silent = false } = {}) {
  if (!id) return false;
  if (id === state.conversationId && state.conversation) {
    closeDrawer();
    focusComposerSoon();
    return true;
  }
  if (state.streaming) {
    detachCurrentStream();
  }
  try {
    const data = await api.conversation(id);
    state.conversationId = id;
    state.conversation = data.conversation || null;
    localStorage.setItem(STORAGE.conversation, id);
    chatView.renderConversation(data.messages, data.conversation);
    updateContext(data.context);
    renderSidebar();
    updateComposerState();
    setStreamingUI(backgroundStreams.has(id));
    closeDrawer();
    if (!silent) focusComposerSoon();
    return true;
  } catch (error) {
    if (localStorage.getItem(STORAGE.conversation) === id) {
      localStorage.removeItem(STORAGE.conversation);
    }
    if (!silent) toast(error?.message || "Couldn't open that chat", { tone: "error" });
    if (state.conversationId === id) {
      state.conversationId = null;
      state.conversation = null;
      chatView.reset();
      renderSidebar();
    }
    return false;
  }
}

function showEmptyState() {
  state.conversationId = null;
  state.conversation = null;
  chatView.reset();
  renderSidebar();
  updateContext(null);
  updateComposerState();
}

function newChat() {
  if (state.streaming) {
    detachCurrentStream();
  }
  state.conversationId = null;
  state.conversation = null;
  localStorage.removeItem(STORAGE.conversation);
  chatView.reset();
  renderSidebar();
  updateContext(null);
  updateComposerState();
  setStreamingUI(false);
  closeDrawer();
  focusComposerSoon();
}

/* -------------------------------------------------------------------------- */
/* Composer                                                                   */
/* -------------------------------------------------------------------------- */

function composerReady() {
  return !state.unreachable && state.lmstudio && state.models.length > 0;
}

function updateComposerState() {
  const ready = composerReady();
  els.composer.classList.toggle("is-disabled", !ready);
  els.input.disabled = !ready;
  els.agentToggle.disabled = !state.settings;

  let hint = "";
  if (state.unreachable) hint = "Can't reach the Kiwibro server. Is it running?";
  else if (!state.lmstudio) hint = "LM Studio isn't running. Start it to chat.";
  else if (!state.models.length) hint = "No chat model loaded in LM Studio.";

  els.dockHint.hidden = !hint;
  if (hint) els.dockHintText.textContent = hint;
  updateSendEnabled();
}

function updateSendEnabled() {
  // While a reply runs, sending steers it (see sendMessage) — keep Enter enabled.
  els.sendBtn.disabled = !composerReady() || !els.input.value.trim();
  els.contextRing.classList.toggle("is-idle", !state.conversationId);
}

function autosize() {
  els.input.style.height = "auto";
  const height = Math.min(els.input.scrollHeight, 200);
  els.input.style.height = `${Math.max(46, height)}px`;
}

function focusComposerSoon() {
  if (mobileQuery.matches) return;
  window.requestAnimationFrame(() => {
    if (document.querySelector("dialog[open]")) return;
    els.input.focus({ preventScroll: true });
  });
}

async function sendMessage(rawText, { echo = true } = {}) {
  const text = String(rawText || "").trim();
  if (!text) return;
  if (
    state.conversationId &&
    (state.streaming || backgroundStreams.has(state.conversationId))
  ) {
    // Sending while a reply runs steers it instead of queueing a new turn.
    try {
      await api.steerChat(state.conversationId, text);
      els.input.value = "";
      autosize();
      toast("Steering note queued — it applies at the next step");
    } catch (error) {
      toast(error?.message || "Couldn't queue the steering note", { tone: "error" });
    }
    return;
  }
  if (state.streaming) return;
  if (state.conversationId && backgroundStreams.has(state.conversationId)) {
    toast("This chat is already replying", { tone: "error" });
    return;
  }
  if (!composerReady()) {
    toast(
      !state.lmstudio ? "LM Studio isn't running" : "No model is available",
      { tone: "error", iconName: "alert" },
    );
    return;
  }

  if (echo) chatView.beginUser(text);
  else chatView.beginTurn();
  state.streaming = true;
  state.lastSend = { text };
  els.input.value = "";
  autosize();
  setStreamingUI(true);
  announce("Kiwibro is replying");

  const controller = new AbortController();
  state.controller = controller;
  let sawDone = false;

  const activeStream = {
    conversationId: state.conversationId,
    controller,
    isForeground: true,
  };
  currentStream = activeStream;

  try {
    const response = await api.chat(
      {
        conversation_id: state.conversationId,
        message: text,
        model: state.currentModel || undefined,
      },
      { signal: controller.signal },
    );
    if (!response.ok) throw await chatError(response);
    await consumeSSE(response, (event, data) => {
      if (event === "done") sawDone = true;
      if (activeStream.isForeground) {
        handleStreamEvent(event, data, activeStream);
      } else {
        handleBackgroundStreamEvent(activeStream, event, data);
      }
    });
    if (!sawDone && activeStream.isForeground) chatView.finishDone({ aborted: false });
  } catch (error) {
    if (activeStream.isForeground) {
      if (error?.name === "AbortError" || controller.signal.aborted) {
        chatView.finishDone({ aborted: true });
      } else {
        if (error?.code === "busy") {
          toast("This chat is already replying", { tone: "error" });
          chatView.finishDone({ aborted: false });
        } else {
          chatView.showError(error?.message || "Something went wrong. Try again.", { retry: true });
        }
      }
    } else {
      if (error?.name !== "AbortError" && !controller.signal.aborted) {
        const bgTitle = backgroundStreams.get(activeStream.conversationId)?.title || "Chat";
        if (error?.code === "busy") {
          toast(`“${bgTitle}” is already replying`, { tone: "error" });
        } else {
          toast(error?.message || `Error in “${bgTitle}”`, { tone: "error" });
        }
      }
    }
  } finally {
    const cid = activeStream.conversationId;
    if (cid) {
      backgroundStreams.delete(cid);
    }
    if (activeStream.isForeground) {
      state.streaming = false;
      state.controller = null;
      setStreamingUI(false);
      scheduleConversationRefresh();
      if (sawDone) {
        announce("Reply complete");
        // Memory extraction runs after done and may not emit an event.
        scheduleMemoryRefresh();
      }
    } else {
      if (state.conversationId && state.conversationId === cid) {
        api.conversation(cid).then((data) => {
          if (state.conversationId === cid) {
            chatView.renderConversation(data.messages, data.conversation);
            updateContext(data.context);
            setStreamingUI(false);
          }
        }).catch(() => {});
      }
      renderSidebar();
      scheduleConversationRefresh();
      if (sawDone) scheduleMemoryRefresh();
    }
    if (currentStream === activeStream) {
      currentStream = null;
    }
  }
}

function handleStreamEvent(event, data, activeStream) {
  switch (event) {
    case "meta": {
      const isNew = !state.conversationId;
      if (data.conversation_id) {
        state.conversationId = data.conversation_id;
        if (activeStream) activeStream.conversationId = data.conversation_id;
        localStorage.setItem(STORAGE.conversation, data.conversation_id);
      }
      if (data.model) {
        state.currentModel = data.model;
        renderHeaderModel();
      }
      if (data.context) updateContext(data.context);
      if (isNew) {
        state.conversations.unshift({
          id: data.conversation_id,
          title: data.title || "New chat",
          model: data.model || state.currentModel,
          pinned: false,
          message_count: 1,
          updated_at: new Date().toISOString(),
        });
        renderSidebar();
        refreshConversations({ silent: true });
      }
      updateSendEnabled();
      break;
    }
    case "status": {
      const label = data.label || PHASE_LABELS[data.phase] || "";
      if (label) chatView.setStatus(label);
      break;
    }
    case "delta":
      if (typeof data.text === "string" && data.text) chatView.appendDelta(data.text);
      break;
    case "reasoning":
      if (typeof data.text === "string" && data.text) chatView.appendReasoning(data.text);
      break;
    case "tool_call":
      chatView.addToolCall(data);
      break;
    case "tool_result":
      chatView.addToolResult(data);
      break;
    case "approval":
      chatView.addApproval(data);
      break;
    case "context":
      updateContext(data);
      if (data.compacted) chatView.addContextChip();
      break;
    case "steering":
      chatView.addSteeringChip?.(data.text || "");
      break;
    case "memory": {
      const added = Number(data.added) || 0;
      const removed = Number(data.removed) || 0;
      if (added || removed || data.updated) {
        state.memoriesCount = Math.max(0, state.memoriesCount + added - removed);
        renderMemoryCounts();
        chatView.addMemoryChip(data);
        memorySheet?.refreshIfOpen();
      }
      break;
    }
    case "title":
      if (data.id && data.title) updateConversationTitle(data.id, data.title);
      break;
    case "done":
      chatView.finishDone({ aborted: Boolean(data.aborted) });
      inferenceUI.onDone(data);
      scheduleConversationRefresh();
      break;
    case "error":
      chatView.showError(data.message || "Something went wrong. Try again.", { retry: true });
      if (data.code === "busy") toast("This chat is already replying", { tone: "error" });
      break;
    default:
      break;
  }
}

function handleBackgroundStreamEvent(activeStream, event, data) {
  let cid = activeStream.conversationId;
  const bg = cid ? backgroundStreams.get(cid) : null;
  const title = bg?.title || "Chat";

  switch (event) {
    case "meta":
      if (data.conversation_id) {
        activeStream.conversationId = data.conversation_id;
        cid = data.conversation_id;
        if (!backgroundStreams.has(data.conversation_id)) {
          backgroundStreams.set(data.conversation_id, {
            controller: activeStream.controller,
            title: data.title || "New chat",
            status: "running",
            needsApproval: false,
          });
          renderSidebar();
        }
      }
      if (data.title && cid) {
        updateConversationTitle(cid, data.title);
        if (bg) bg.title = data.title;
      }
      break;
    case "status":
      if (bg) bg.status = data.label || PHASE_LABELS[data.phase] || "running";
      break;
    case "title":
      if (data.id && data.title) {
        updateConversationTitle(data.id, data.title);
        if (bg) bg.title = data.title;
      }
      break;
    case "approval":
      if (bg) bg.needsApproval = true;
      toast(`Approval needed in “${title}”`, { tone: "info" });
      renderSidebar();
      break;
    case "context":
      break;
    case "memory": {
      const added = Number(data.added) || 0;
      const removed = Number(data.removed) || 0;
      if (added || removed || data.updated) {
        state.memoriesCount = Math.max(0, state.memoriesCount + added - removed);
        renderMemoryCounts();
        memorySheet?.refreshIfOpen();
      }
      break;
    }
    case "done":
      toast(`Reply complete in “${title}”`);
      break;
    case "error":
      if (data.code === "busy") {
        toast(`“${title}” is already replying`, { tone: "error" });
      } else {
        toast(data.message || `Error in “${title}”`, { tone: "error" });
      }
      break;
    default:
      break;
  }
}

function setStreamingUI(streaming) {
  els.sendBtn.hidden = streaming;
  els.stopBtn.hidden = !streaming;
  els.input.placeholder = streaming ? "Steer this reply… (Enter applies it)" : "Ask Kiwibro…";
  updateSendEnabled();
}

function stopStreaming() {
  if (state.streaming) {
    const conversationId = state.conversationId;
    if (conversationId) api.stopChat(conversationId).catch(() => {});
    chatView.finishDone({ aborted: true });
    state.controller?.abort();
    announce("Reply stopped");
    return;
  }
  if (state.conversationId && backgroundStreams.has(state.conversationId)) {
    const bg = backgroundStreams.get(state.conversationId);
    api.stopChat(state.conversationId).catch(() => {});
    bg?.controller?.abort();
    backgroundStreams.delete(state.conversationId);
    setStreamingUI(false);
    renderSidebar();
    announce("Reply stopped");
  }
}

function retryLastSend() {
  if (state.streaming) return;
  const text = state.lastSend?.text;
  if (!text) return;
  chatView.prepareRetry();
  sendMessage(text, { echo: false });
}

async function toggleAgent() {
  if (!state.settings) return;
  const next = state.settings.agent_enabled === false;
  els.agentToggle.setAttribute("aria-pressed", String(next));
  try {
    await patchSettingsAndSync({ agent_enabled: next });
    announce(next ? "Agent mode on" : "Agent mode off");
  } catch (error) {
    els.agentToggle.setAttribute("aria-pressed", String(!next));
    toast(error?.message || "Couldn't change agent mode", { tone: "error" });
  }
}

/* -------------------------------------------------------------------------- */
/* Settings & memory helpers                                                  */
/* -------------------------------------------------------------------------- */

async function patchSettingsAndSync(patch) {
  const data = await api.patchSettings(patch);
  const next = data?.settings || { ...(state.settings || {}), ...patch };
  state.settings = next;
  if (data?.data_dir) state.dataDir = data.data_dir;
  if (next.workspace) state.workspace = next.workspace;
  applySettingsUI();
  updateComposerState();
  return next;
}

function applySettingsUI() {
  const agentEnabled = state.settings?.agent_enabled !== false;
  els.agentToggle.setAttribute("aria-pressed", String(agentEnabled));
  els.agentToggle.title = agentEnabled
    ? "Agent mode — tools and memory"
    : "Agent mode is off — tools disabled";
  inferenceUI.applySettings(state.settings);
}

async function forgetAllMemories() {
  const result = await api.forgetMemories();
  state.memoriesCount = 0;
  renderMemoryCounts();
  memorySheet?.refreshIfOpen();
  return result?.deleted;
}

function scheduleMemoryRefresh(delay = 4200) {
  window.clearTimeout(memoryRefreshTimer);
  memoryRefreshTimer = window.setTimeout(refreshMemoryCount, delay);
}

async function refreshMemoryCount() {
  if (!state.booted || state.unreachable || state.streaming) return;
  try {
    const count = await api.memoryCount();
    if (!Number.isFinite(count) || count === state.memoriesCount) return;
    state.memoriesCount = count;
    renderMemoryCounts();
    memorySheet?.refreshIfOpen();
  } catch {
    /* memory extraction is best-effort; stay quiet */
  }
}

function renderMemoryCounts() {
  const count = state.memoriesCount;
  els.memoryBadge.hidden = count <= 0;
  els.memoryBadge.textContent = count > 99 ? "99+" : String(count);
  els.sidebarMemoryCount.textContent = count === 1 ? "1 memory" : `${count} memories`;
}

function updateContext(context) {
  const used = Number(context?.used) || 0;
  const budget = Number(context?.budget) || 0;
  let percent = Number(context?.percent);
  if (!Number.isFinite(percent)) percent = budget > 0 ? (used / budget) * 100 : 0;
  percent = Math.min(100, Math.max(0, percent));
  els.ringValue.style.strokeDashoffset = String(56.55 * (1 - percent / 100));
  els.contextRing.dataset.level = percent >= 80 ? "high" : percent >= 55 ? "medium" : "low";
  els.contextRing.classList.toggle("is-idle", !state.conversationId);
  const description = budget
    ? `Context ${Math.round(percent)}% used — ${formatTokens(used)} of ${formatTokens(budget)} tokens`
    : `Context ${Math.round(percent)}% used`;
  els.contextRing.title = description;
  els.contextRing.setAttribute("aria-label", description);
  inferenceUI.setContext({ used, budget, percent });
}

/* -------------------------------------------------------------------------- */
/* Sidebar drawer / collapse                                                  */
/* -------------------------------------------------------------------------- */

function isMobile() {
  return mobileQuery.matches;
}

function setDrawer(open) {
  state.drawerOpen = Boolean(open) && isMobile();
  els.app.dataset.drawer = state.drawerOpen ? "open" : "closed";
  els.scrim.hidden = !state.drawerOpen;
  els.sidebarToggle.setAttribute("aria-expanded", String(state.drawerOpen));
  if (state.drawerOpen) {
    window.requestAnimationFrame(() => {
      els.sidebar.querySelector(".new-chat")?.focus({ preventScroll: true });
    });
  } else if (document.activeElement === els.sidebarToggle) {
    els.sidebarToggle.focus({ preventScroll: true });
  }
}

function closeDrawer() {
  if (state.drawerOpen) setDrawer(false);
}

function toggleSidebar() {
  if (isMobile()) {
    setDrawer(!state.drawerOpen);
    return;
  }
  const hidden = els.app.dataset.sidebar === "hidden";
  const next = hidden ? "shown" : "hidden";
  els.app.dataset.sidebar = next;
  localStorage.setItem(STORAGE.sidebar, next);
  els.sidebarToggle.setAttribute("aria-expanded", String(next !== "hidden"));
}

function applyViewport() {
  if (isMobile()) {
    els.app.dataset.sidebar = "shown";
    setDrawer(false);
    els.sidebarToggle.setAttribute("aria-expanded", "false");
  } else {
    state.drawerOpen = false;
    els.app.dataset.drawer = "closed";
    els.scrim.hidden = true;
    const preference = localStorage.getItem(STORAGE.sidebar) || "shown";
    els.app.dataset.sidebar = preference;
    els.sidebarToggle.setAttribute("aria-expanded", String(preference !== "hidden"));
  }
}

/* -------------------------------------------------------------------------- */
/* Health                                                                     */
/* -------------------------------------------------------------------------- */

async function reconnect() {
  els.dockRetry.disabled = true;
  try {
    if (state.unreachable) {
      await boot();
      return;
    }
    const health = await api.health();
    state.health = health;
    state.lmstudio = Boolean(health.lmstudio);
    if (state.lmstudio) await refreshModels({ quiet: true });
    renderHeaderModel();
    updateComposerState();
    if (state.lmstudio) toast("LM Studio connected");
  } catch (error) {
    toast(error?.message || "Still offline", { tone: "error" });
  } finally {
    els.dockRetry.disabled = false;
  }
}

async function maybeCheckHealth() {
  if (!state.booted || state.unreachable || state.streaming) return;
  if (Date.now() - healthCheckAt < 15_000) return;
  healthCheckAt = Date.now();
  try {
    const health = await api.health();
    const wasOnline = state.lmstudio;
    state.health = health;
    state.lmstudio = Boolean(health.lmstudio);
    if (state.lmstudio !== wasOnline) {
      if (state.lmstudio) await refreshModels({ quiet: true });
      renderHeaderModel();
      updateComposerState();
      if (state.lmstudio) toast("LM Studio connected");
    }
    refreshMemoryCount();
  } catch {
    /* offline: keep the calm hint */
  }
}

/* -------------------------------------------------------------------------- */
/* Event wiring                                                               */
/* -------------------------------------------------------------------------- */

function initSheets() {
  memorySheet = createMemorySheet({
    dialog: els.memorySheet,
    api,
    onChange: (count) => {
      if (!Number.isFinite(count) || state.unreachable) return;
      state.memoriesCount = count;
      renderMemoryCounts();
    },
  });

  settingsSheet = createSettingsSheet({
    dialog: els.settingsSheet,
    api,
    isMock: IS_MOCK,
    actions: {
      selectModel,
      patchSettings: patchSettingsAndSync,
      forgetMemories: forgetAllMemories,
    },
  });

  wireDialog(els.memorySheet);
  wireDialog(els.settingsSheet);
}

function initComposer() {
  els.input.addEventListener("input", () => {
    autosize();
    updateSendEnabled();
  });

  els.input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    if (!els.sendBtn.disabled) sendMessage(els.input.value);
  });

  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!els.sendBtn.disabled) sendMessage(els.input.value);
  });

  els.agentToggle.addEventListener("click", toggleAgent);
  els.stopBtn.addEventListener("click", stopStreaming);
  autosize();
}

function initChrome() {
  els.sidebarToggle.addEventListener("click", toggleSidebar);
  els.brandHome.addEventListener("click", newChat);
  els.newChat.addEventListener("click", newChat);
  els.convSearch?.addEventListener("input", (event) => {
    state.convFilter = event.target.value || "";
    renderSidebar();
  });
  els.convSearch?.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && els.convSearch.value) {
      event.preventDefault();
      event.stopPropagation();
      els.convSearch.value = "";
      state.convFilter = "";
      renderSidebar();
    }
  });
  els.memoryBtn.addEventListener("click", () => memorySheet?.open());
  els.sidebarMemory.addEventListener("click", () => memorySheet?.open());
  els.settingsBtn.addEventListener("click", () => settingsSheet?.open());
  els.sidebarModel.addEventListener("click", () => settingsSheet?.open());
  els.modelPill.addEventListener("click", toggleModelPopover);
  els.dockRetry.addEventListener("click", reconnect);
  els.scrim.addEventListener("click", closeDrawer);
  els.scrollBottom.addEventListener("click", () => chatView.scrollToBottom({ smooth: true }));

  els.empty.addEventListener("click", (event) => {
    const suggestion = event.target.closest("[data-prompt]");
    if (!suggestion) return;
    sendMessage(suggestion.dataset.prompt);
  });

  mobileQuery.addEventListener("change", applyViewport);
  window.addEventListener("resize", () => {
    if (state.drawerOpen && !isMobile()) closeDrawer();
  });
  window.addEventListener("focus", maybeCheckHealth);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) maybeCheckHealth();
  });

  window.setInterval(() => updateRelativeTimes(), 30_000);
  updateRelativeTimes();
}

function initShortcuts() {
  document.addEventListener("keydown", (event) => {
    const mod = event.ctrlKey || event.metaKey;
    if (mod && !event.altKey && event.key.toLowerCase() === "n") {
      event.preventDefault();
      newChat();
      return;
    }
    if (mod && !event.altKey && event.key.toLowerCase() === "k") {
      event.preventDefault();
      if (!state.drawerOpen && isMobile()) setDrawer(true);
      if (els.app.dataset.sidebar === "hidden" && !isMobile()) toggleSidebar();
      els.convSearch?.focus();
      els.convSearch?.select();
      return;
    }
    if (mod && !event.altKey && (event.key === "," || event.code === "Comma")) {
      event.preventDefault();
      settingsSheet?.open();
      return;
    }
    if (event.key !== "Escape") return;
    if (document.querySelector("dialog[open]")) return;
    if (popoverIsOpen()) {
      closePopover();
      return;
    }
    if (state.drawerOpen) {
      closeDrawer();
      return;
    }
    if (chatView.hasPendingApproval()) {
      chatView.denyPending();
      return;
    }
    if (state.streaming) stopStreaming();
  });
}

/* -------------------------------------------------------------------------- */
/* Boot                                                                       */
/* -------------------------------------------------------------------------- */

function init() {
  applyViewport();
  initSheets();
  initComposer();
  initChrome();
  initShortcuts();
  boot();
}

init();
