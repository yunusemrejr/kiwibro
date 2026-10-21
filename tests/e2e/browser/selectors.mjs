/**
 * Central selector registry for Kiwibro browser E2E.
 *
 * Keep every id/aria/text lookup here so a markup refactor only breaks one file.
 * Ids come from templates/index.html; dynamic pieces from static/js/*.js.
 */

export const SEL = {
  app: "#app",
  headerbar: ".headerbar",
  sidebar: "#sidebar",
  sidebarToggle: "#sidebar-toggle",
  brandHome: "#brand-home",
  newChat: "#new-chat",
  convList: "#conv-list",
  convRow: ".conv-row",
  convOpen: ".conv-open",
  convEmpty: "#conv-empty",
  transcript: "#transcript",
  chat: "#chat",
  emptyState: "#empty-state",
  emptyTitle: "#empty-title",
  transcriptLog: '[role="log"]',

  modelPill: "#model-pill",
  modelPillName: "#model-pill-name",
  modelDot: "#model-dot",
  modelPopover: ".popover",
  modelItem: ".popover .menu-item",
  modelItemActive: ".popover .menu-item.is-selected",

  memoryBtn: "#memory-btn",
  memoryBadge: "#memory-badge",
  settingsBtn: "#settings-btn",

  composer: "#composer",
  composerInput: "#composer-input",
  sendBtn: "#send-btn",
  stopBtn: "#stop-btn",
  agentToggle: "#agent-toggle",
  contextRing: "#context-ring",
  dockHint: "#dock-hint",
  dockRetry: "#dock-retry",

  userBubble: ".msg-user .bubble",
  userMessage: ".msg-user",
  assistantMessage: ".msg-assistant",
  assistantMarkdown: ".msg-assistant .md",
  streamCaret: ".stream-caret",
  reasoningToggle: ".reasoning-toggle",
  toolRow: ".tool-row",
  toolList: ".tool-list",

  approval: ".approval",
  approvalAllow: ".approval .btn-primary",
  approvalAlways: ".approval .btn-ghost",
  approvalDeny: ".approval .btn-deny",
  approvalResolved: ".approval.is-resolved",

  memorySheet: "#memory-sheet",
  memoryList: "#memory-list",
  memorySearch: "#memory-search",
  memoryNew: "#memory-new",
  memoryKind: "#memory-new-kind",
  memoryAddForm: "#memory-add-form",
  memoryAddButton: '#memory-add-form button[type="submit"]',
  forgetAll: "#forget-all-btn",
  kindChip: ".kind-chip",
  memoryRow: ".memory-row",

  settingsSheet: "#settings-sheet",
  settingsBody: "#settings-body",

  toastStack: "#toast-stack",
  toast: ".toast",
  liveStatus: "#live-status",
  dialogOpen: "dialog[open]",
};

/** Labels used for text-based waits (kept separate from CSS selectors). */
export const TEXT = {
  lmOfflineHints: ["LM Studio isn't running", "No chat model loaded", "Can't reach the Kiwibro server"],
  send: "Send",
};

export const APPROVAL_PROMPT =
  "Search the web for kiwi conservation news and save a note to a file";
export const SIMPLE_PROMPT = "In one short sentence, what is a kiwi bird?";
