/**
 * The transcript: message rendering, streaming segments, tool activity,
 * approvals, reasoning, chips and error rows.
 */

import { renderMarkdown } from "./markdown.js";
import {
  el,
  icon,
  copyText,
  toast,
  announce,
  toolLabel,
  toolIcon,
  toolSummary,
  normalizeToolCall,
  prettyArgs,
  riskLevel,
  summarizeArgs,
} from "./ui.js";

const ASSISTANT_PHRASE = {
  web_search: "search the web",
  fetch_page: "fetch a web page",
  multi_search: "search the web with several queries",
  fetch_pages: "fetch several web pages",
  web_research: "research a question on the web",
  browse_page: "browse a web page",
  browser: "use the browser",
  read_file: "read a file from your workspace",
  read_files: "read files from your workspace",
  list_files: "look through your workspace",
  glob_files: "find files in your workspace",
  grep_files: "search your files",
  file_outline: "outline a file",
  git: "inspect the git repository",
  write_file: "write a file in your workspace",
  replace_in_file: "edit a file in your workspace",
  apply_patch: "patch a file in your workspace",
  file_manage: "move, copy or delete files",
  run_command: "run a shell command",
  multi_command: "run several shell commands",
  terminal_session: "use a terminal session",
  run_python: "run a Python snippet",
  run_javascript: "run a JavaScript snippet",
  system_info: "check this computer",
  open_path: "open something on your desktop",
  clipboard: "use your clipboard",
  notify: "show a notification",
  screenshot: "take a screenshot",
  model_info: "list your local models",
  delegate: "ask another local model",
  describe_image: "look at an image with a vision model",
  study_file: "study a long file with a helper model",
  task_board: "update the task board",
  recall: "search memory",
  remember: "save a memory",
  calculate: "calculate an exact result",
  analyze_table: "analyze a data table",
  doc_search: "search your documents",
  doc_read: "read a document",
  doc_add: "index documents in your workspace",
  doc_list: "list your documents",
  undo_changes: "undo its recent file changes",
};

function caretSpan() {
  const caret = document.createElement("span");
  caret.className = "stream-caret";
  caret.setAttribute("aria-hidden", "true");
  return caret;
}

function injectCaret(container) {
  const last = container.lastElementChild;
  if (!last) {
    container.append(caretSpan());
    return;
  }
  if (last.classList.contains("code-block")) {
    const code = last.querySelector("code");
    (code || last).append(caretSpan());
    return;
  }
  if (/^(P|H1|H2|H3|H4|LI|BLOCKQUOTE)$/.test(last.tagName)) {
    last.append(caretSpan());
    return;
  }
  container.append(caretSpan());
}

function truncate(value, max) {
  const text = String(value ?? "");
  return text.length > max ? `${text.slice(0, max)}\n…` : text;
}

function riskBadge(risk) {
  const level = riskLevel(risk);
  return el("span", {
    class: "risk-badge",
    "data-risk": level,
    text: `${level[0].toUpperCase()}${level.slice(1)} risk`,
  });
}

function approvalDetailRows(approval) {
  let args = approval.args;
  if (typeof args === "string") {
    try {
      args = JSON.parse(args);
    } catch {
      args = { raw: args };
    }
  }
  if (!args || typeof args !== "object") return [{ label: "Arguments", value: String(args ?? "") }];

  const rows = [];
  switch (approval.tool) {
    case "run_command":
      rows.push({ label: "Command", value: truncate(args.command, 2000), pre: true });
      if (args.cwd) rows.push({ label: "Working directory", value: args.cwd, pre: false });
      break;
    case "write_file":
      if (args.path) rows.push({ label: "Path", value: args.path, pre: false });
      if (args.content != null) {
        rows.push({ label: "Content preview", value: truncate(args.content, 1600), pre: true });
      }
      break;
    case "replace_in_file":
      if (args.path) rows.push({ label: "Path", value: args.path, pre: false });
      if (args.old != null) rows.push({ label: "Replace", value: truncate(args.old, 600), pre: true });
      if (args.new != null) rows.push({ label: "With", value: truncate(args.new, 600), pre: true });
      break;
    case "fetch_page":
      if (args.url) rows.push({ label: "URL", value: args.url, pre: false });
      break;
    case "apply_patch":
      if (args.path) rows.push({ label: "Path", value: args.path, pre: false });
      if (Array.isArray(args.edits)) {
        for (const [index, edit] of args.edits.entries()) {
          rows.push({ label: `Edit ${index + 1} — replace`, value: truncate(edit.old, 500), pre: true });
          rows.push({ label: `Edit ${index + 1} — with`, value: truncate(edit.new, 500), pre: true });
        }
      }
      break;
    case "file_manage":
      if (args.action) rows.push({ label: "Action", value: args.action, pre: false });
      if (args.path) rows.push({ label: "Path", value: args.path, pre: false });
      if (args.destination) rows.push({ label: "Destination", value: args.destination, pre: false });
      if (args.recursive) rows.push({ label: "Recursive", value: "yes", pre: false });
      break;
    case "multi_command":
      if (Array.isArray(args.commands)) {
        rows.push({ label: "Commands", value: truncate(args.commands.join("\n"), 2000), pre: true });
      }
      if (args.cwd) rows.push({ label: "Working directory", value: args.cwd, pre: false });
      break;
    case "terminal_session":
      if (args.action) rows.push({ label: "Action", value: args.action, pre: false });
      if (args.command) rows.push({ label: "Command", value: truncate(args.command, 1000), pre: true });
      if (args.session_id) rows.push({ label: "Session", value: args.session_id, pre: false });
      if (args.input) rows.push({ label: "Input", value: truncate(args.input, 500), pre: true });
      break;
    case "run_python":
    case "run_javascript":
      if (args.code != null) rows.push({ label: "Code", value: truncate(args.code, 2000), pre: true });
      break;
    case "open_path":
      if (args.target) rows.push({ label: "Open", value: args.target, pre: false });
      break;
    case "clipboard":
      rows.push({ label: "Action", value: args.action || "", pre: false });
      if (args.text != null) rows.push({ label: "Text", value: truncate(args.text, 800), pre: true });
      break;
    case "screenshot":
      if (args.filename) rows.push({ label: "Save as", value: args.filename, pre: false });
      break;
    case "remember":
      if (args.content) rows.push({ label: "Memory", value: args.content, pre: false });
      break;
    default:
      rows.push({ label: "Arguments", value: prettyArgs(args), pre: true });
  }
  return rows;
}

function buildApprovalCard(approval, resolve) {
  const tool = approval.tool || "tool";
  // The safety level (when present) grades this specific call, not just the tool.
  const level = riskLevel(approval.level || approval.risk);
  const summary = approval.summary || summarizeArgs(approval.args) || "";
  const phrase = ASSISTANT_PHRASE[tool] || "use a tool";
  const reasons = Array.isArray(approval.reasons) ? approval.reasons.filter(Boolean) : [];

  const more = el(
    "button",
    { class: "approval-more", type: "button", "aria-expanded": "false" },
    [icon("chevron-right"), el("span", { text: "Details" })],
  );
  const detail = el("div", { class: "approval-detail", hidden: true });
  for (const row of approvalDetailRows(approval)) {
    if (row.pre) {
      detail.append(
        el("div", {}, [
          el("div", { class: "detail-label", text: row.label }),
          el("pre", { text: row.value }),
        ]),
      );
    } else {
      detail.append(
        el("div", {}, [
          el("div", { class: "detail-label", text: row.label }),
          el("div", { class: "detail-value", text: row.value }),
        ]),
      );
    }
  }
  more.addEventListener("click", () => {
    const expanded = more.getAttribute("aria-expanded") === "true";
    more.setAttribute("aria-expanded", String(!expanded));
    detail.hidden = expanded;
  });

  const allow = el("button", { class: "btn btn-primary", type: "button", text: "Allow" });
  const always = el(
    "button",
    {
      class: "btn btn-ghost",
      type: "button",
      title: `Always allow ${toolLabel(tool).toLowerCase()} without asking`,
      text: "Always",
    },
  );
  const deny = el("button", { class: "btn btn-deny", type: "button", text: "Deny" });

  const card = el(
    "div",
    {
      class: "approval",
      "data-risk": level,
      role: "group",
      "aria-label": `Approval required: ${toolLabel(tool)}`,
    },
    [
      el("div", { class: "approval-head" }, [
        el("span", { class: "approval-tool-icon" }, [icon(toolIcon(tool))]),
        el("div", { class: "approval-title" }, [
          el("span", { class: "approval-name", text: toolLabel(tool) }),
          el("span", { class: "approval-sub", text: `Kiwibro wants to ${phrase}` }),
        ]),
        riskBadge(level),
      ]),
      el("div", { class: "approval-summary", text: summary }),
      reasons.length
        ? el(
            "ul",
            { class: "approval-reasons", "aria-label": "Why Kiwibro is asking" },
            reasons.map((text) => el("li", { text })),
          )
        : null,
      more,
      detail,
      el("div", { class: "approval-actions" }, [deny, always, allow]),
    ],
  );

  const setBusy = (busy) => {
    for (const button of [allow, always, deny, more]) button.disabled = busy;
    card.setAttribute("aria-busy", String(busy));
  };

  const markResolved = (decision, note) => {
    if (card.classList.contains("is-resolved")) return;
    card.classList.add("is-resolved");
    card.classList.toggle("is-approved", decision !== "deny");
    card.classList.toggle("is-denied", decision === "deny");
    card.querySelector(".approval-actions")?.remove();
    const approved = decision !== "deny";
    const label =
      note ||
      (decision === "allow" ? "Allowed" : decision === "always" ? "Always allowed" : "Denied");
    card.append(
      el("div", { class: "approval-resolved" }, [
        icon(approved ? "check" : "ban"),
        el("span", { text: label }),
      ]),
    );
    if (approved) announce("Tool call approved");
    else announce("Tool call denied");
  };

  let settled = false;
  const choose = async (decision) => {
    if (settled) return;
    settled = true;
    setBusy(true);
    try {
      await resolve(decision);
      markResolved(decision);
    } catch (error) {
      settled = false;
      setBusy(false);
      if (error?.status === 404) {
        markResolved("deny", "Request expired");
        toast("That approval expired", { tone: "error" });
      } else {
        toast(error?.message || "Could not send the decision", { tone: "error" });
      }
    }
  };

  allow.addEventListener("click", () => choose("allow"));
  always.addEventListener("click", () => choose("always"));
  deny.addEventListener("click", () => choose("deny"));

  return {
    node: card,
    markResolved,
    expire() {
      if (!card.classList.contains("is-resolved")) markResolved("deny", "Expired");
      settled = true;
    },
  };
}

function createToolRow(entry) {
  const name = entry.name || "tool";
  const args = entry.args;
  const summary = summarizeArgs(args, "");
  const row = el("button", { class: "tool-row", type: "button", "aria-expanded": "false" }, [
    el("span", { class: "tool-icon" }, [icon(toolIcon(name))]),
    el("span", { class: "tool-name", text: toolLabel(name) }),
    el("span", { class: "tool-summary", text: toolSummary(name, args) || summary }),
    el("span", { class: "tool-duration", hidden: true }),
    el("span", { class: "tool-status" }, [el("span", { class: "spinner" })]),
    icon("chevron-right", "tool-chevron"),
  ]);
  const detail = el("div", { class: "tool-detail", hidden: true });
  const node = el("div", { class: "tool is-running" }, [row, detail]);
  let result = null;
  let expanded = false;

  const renderDetail = () => {
    detail.replaceChildren();
    if (args != null) {
      detail.append(
        el("div", {}, [
          el("div", { class: "detail-label", text: "Arguments" }),
          el("pre", { text: prettyArgs(args) }),
        ]),
      );
    }
    if (result) {
      detail.append(
        el("div", {}, [
          el("div", { class: "detail-label", text: "Result" }),
          el("pre", { class: "tool-result-text", text: truncate(result.preview, 4000) }),
        ]),
      );
    }
  };

  row.addEventListener("click", () => {
    expanded = !expanded;
    row.setAttribute("aria-expanded", String(expanded));
    node.classList.toggle("is-expanded", expanded);
    detail.hidden = !expanded;
    if (expanded) renderDetail();
  });

  const api = {
    id: entry.id,
    node,
    addResult(next) {
      result = next;
      node.classList.remove("is-running");
      node.classList.add(next.ok === false ? "is-fail" : "is-ok");
      const status = node.querySelector(".tool-status");
      status.replaceChildren(icon(next.ok === false ? "x" : "check"));
      const duration = node.querySelector(".tool-duration");
      if (Number.isFinite(next.duration_ms) && next.duration_ms >= 0) {
        duration.textContent =
          next.duration_ms >= 1000
            ? `${(next.duration_ms / 1000).toFixed(1)}s`
            : `${Math.round(next.duration_ms)}ms`;
        duration.hidden = false;
      }
      if (expanded) renderDetail();
    },
    fail() {
      node.classList.remove("is-running");
      node.classList.add("is-fail");
      node.querySelector(".tool-status")?.replaceChildren(icon("x"));
      if (expanded) renderDetail();
    },
  };
  return api;
}

export function createChatView({ root, transcript, resolveApproval, onRetry, onScrollState }) {
  const empty = root.querySelector(".empty");
  let current = null;
  let pendingApprovals = new Map();
  let lastRender = 0;
  let pinned = true;

  function refreshEmpty() {
    const hasMessages = root.querySelector(".msg, .conversation-summary");
    if (empty) empty.hidden = Boolean(hasMessages);
    root.classList.toggle("is-empty", !hasMessages);
  }

  function isPinned() {
    return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 96;
  }

  function scrollToBottom({ smooth = false } = {}) {
    transcript.scrollTo({
      top: transcript.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  }

  function scrollIfPinned() {
    if (!pinned) return;
    window.requestAnimationFrame(() => {
      if (pinned) transcript.scrollTop = transcript.scrollHeight;
    });
  }

  function finishReasoning(turn) {
    if (!turn.reasoningActive) return;
    turn.reasoningActive = false;
    turn.reasoning.classList.remove("is-thinking");
    const seconds = Math.max(1, Math.round((performance.now() - turn.reasoningStarted) / 1000));
    turn.reasoningLabel.textContent = `Thought for ${seconds}s`;
    turn.reasoningToggle.setAttribute("aria-expanded", String(turn.reasoningExpanded));
    turn.reasoningBody.hidden = !turn.reasoningExpanded;
  }

  function renderSegment(turn, withCaret) {
    const segment = turn.segment;
    if (!segment) return;
    segment.element.innerHTML = renderMarkdown(segment.text);
    if (withCaret) injectCaret(segment.element);
  }

  function flushSegment(turn) {
    if (turn.raf) {
      window.clearTimeout(turn.raf);
      turn.raf = 0;
    }
    if (!turn.segment) return;
    renderSegment(turn, false);
  }

  function closeSegment(turn) {
    if (!turn.segment) return;
    flushSegment(turn);
    turn.segment = null;
  }

  function scheduleRender(turn) {
    if (turn.raf) return;
    const run = () => {
      turn.raf = 0;
      lastRender = performance.now();
      if (!turn.segment) return;
      renderSegment(turn, true);
      scrollIfPinned();
    };
    const delay = Math.max(0, 70 - (performance.now() - lastRender));
    turn.raf = window.setTimeout(run, delay);
  }

  function createTurnShell() {
    if (empty) empty.hidden = true;
    root.classList.remove("is-empty");

    const reasoningLabel = el("span", { class: "reasoning-label", text: "Thinking…" });
    const reasoningToggle = el(
      "button",
      {
        class: "reasoning-toggle",
        type: "button",
        "aria-expanded": "false",
        "aria-controls": "",
        title: "Show reasoning",
      },
      [icon("sparkles", "reasoning-icon"), reasoningLabel, icon("chevron-right", "reasoning-chevron")],
    );
    const reasoningBody = el("div", { class: "reasoning-body", hidden: true });
    const reasoning = el("div", { class: "reasoning", hidden: true }, [
      reasoningToggle,
      reasoningBody,
    ]);

    const activityLabel = el("span", { class: "activity-label", text: "" });
    const activity = el("div", { class: "activity", hidden: true }, [
      el("span", { class: "spinner" }),
      activityLabel,
    ]);

    const flow = el("div", { class: "assistant-flow" });
    const article = el("article", { class: "msg msg-assistant" }, [reasoning, flow]);
    root.append(article);

    const turn = {
      article,
      flow,
      reasoning,
      reasoningToggle,
      reasoningBody,
      reasoningLabel,
      reasoningText: "",
      reasoningActive: false,
      reasoningStarted: 0,
      reasoningExpanded: false,
      activity,
      activityLabel,
      segment: null,
      tools: new Map(),
      chips: null,
      error: null,
      raf: 0,
      finished: false,
    };

    reasoningToggle.addEventListener("click", () => {
      turn.reasoningExpanded = !turn.reasoningExpanded;
      reasoningToggle.setAttribute("aria-expanded", String(turn.reasoningExpanded));
      reasoningBody.hidden = !turn.reasoningExpanded;
      if (turn.reasoningExpanded) scrollIfPinned();
    });

    return turn;
  }

  function ensureTurn() {
    if (!current) current = createTurnShell();
    return current;
  }

  function finishTurn({ aborted = false } = {}) {
    if (!current) return;
    const turn = current;
    if (turn.finished) return;
    turn.finished = true;
    if (turn.raf) {
      window.clearTimeout(turn.raf);
      turn.raf = 0;
    }
    finishReasoning(turn);
    closeSegment(turn);
    turn.activity.hidden = true;
    for (const approval of pendingApprovals.values()) approval.expire();
    pendingApprovals.clear();
    if (aborted) {
      ensureChips(turn).append(
        el("span", { class: "chip chip-stopped" }, [icon("stop"), el("span", { text: "Stopped" })]),
      );
    }
    current = null;
    scrollIfPinned();
  }

  function ensureChips(turn) {
    if (!turn.chips) {
      turn.chips = el("div", { class: "chips" });
      turn.flow.append(turn.chips);
    }
    return turn.chips;
  }

  function appendChip(turn, chip) {
    const chips = ensureChips(turn);
    chips.append(chip);
    scrollIfPinned();
  }

  function startUser(text) {
    if (empty) empty.hidden = true;
    root.classList.remove("is-empty");
    current = null;
    const bubble = el("div", { class: "bubble", text });
    root.append(el("div", { class: "msg msg-user" }, [bubble]));
    pinned = true;
    scrollToBottom();
  }

  function showError(message, { retry = true } = {}) {
    const turn = ensureTurn();
    finishReasoning(turn);
    closeSegment(turn);
    turn.activity.hidden = true;
    if (turn.error) {
      turn.error.querySelector(".msg-error-text").textContent = message;
      return;
    }
    const retryButton = el("button", { class: "text-btn", type: "button", text: "Retry" });
    retryButton.addEventListener("click", () => {
      if (turn.error) {
        turn.error.remove();
        turn.error = null;
      }
      const hasContent = Boolean(
        turn.article.querySelector(".md, .tool, .approval, .chips"),
      );
      if (!hasContent) {
        turn.article.remove();
        if (current === turn) current = null;
        refreshEmpty();
      }
      onRetry?.();
    });
    const row = el("div", { class: "msg-error", role: "alert" }, [
      icon("alert"),
      el("span", { class: "msg-error-text", text: message }),
      retry ? retryButton : null,
    ]);
    turn.error = row;
    turn.flow.append(row);
    scrollIfPinned();
  }

  function reset() {
    if (current?.raf) window.clearTimeout(current.raf);
    current = null;
    pendingApprovals.clear();
    for (const node of Array.from(root.children)) {
      if (node !== empty) node.remove();
    }
    refreshEmpty();
    transcript.scrollTop = 0;
    pinned = true;
    onScrollState?.(true);
  }

  function openConversation(messages, conversation) {
    reset();
    const list = Array.isArray(messages) ? messages : [];
    if (!list.length) {
      refreshEmpty();
      return;
    }
    if (empty) empty.hidden = true;
    root.classList.remove("is-empty");

    if (conversation?.compacted && conversation.summary) {
      const panel = el("div", { class: "summary-panel", hidden: true, text: conversation.summary });
      const chip = el(
        "button",
        {
          class: "chip chip-context",
          type: "button",
          "aria-expanded": "false",
          title: "Older messages were summarized to fit the model's context",
        },
        [icon("layers"), el("span", { text: "Earlier conversation summarized" })],
      );
      chip.addEventListener("click", () => {
        const expanded = chip.getAttribute("aria-expanded") === "true";
        chip.setAttribute("aria-expanded", String(!expanded));
        panel.hidden = expanded;
      });
      root.append(el("div", { class: "conversation-summary" }, [chip, panel]));
    }

    const results = new Map();
    for (const message of list) {
      if (message.role === "tool" && message.tool_call_id) results.set(message.tool_call_id, message);
    }

    for (const message of list) {
      if (message.role === "user") {
        root.append(
          el("div", { class: "msg msg-user" }, [
            el("div", { class: "bubble", text: message.content || "" }),
          ]),
        );
        continue;
      }
      if (message.role !== "assistant") continue;

      const flow = el("div", { class: "assistant-flow" });
      const article = el("article", { class: "msg msg-assistant" }, [flow]);

      if (message.reasoning) {
        const label = el("span", { class: "reasoning-label", text: "Thought for a moment" });
        const toggle = el(
          "button",
          { class: "reasoning-toggle", type: "button", "aria-expanded": "false" },
          [icon("sparkles", "reasoning-icon"), label, icon("chevron-right", "reasoning-chevron")],
        );
        const body = el("div", { class: "reasoning-body", hidden: true, text: message.reasoning });
        toggle.addEventListener("click", () => {
          const expanded = toggle.getAttribute("aria-expanded") === "true";
          toggle.setAttribute("aria-expanded", String(!expanded));
          body.hidden = expanded;
        });
        article.prepend(el("div", { class: "reasoning" }, [toggle, body]));
      }

      if (message.content) {
        const md = el("div", { class: "md" });
        md.innerHTML = renderMarkdown(message.content);
        flow.append(md);
      }

      let calls = message.tool_calls;
      if (typeof calls === "string") {
        try {
          calls = JSON.parse(calls);
        } catch {
          calls = [];
        }
      }
      if (Array.isArray(calls) && calls.length) {
        const list = el("div", { class: "tool-list" });
        for (const raw of calls) {
          const call = normalizeToolCall(raw);
          if (!call) continue;
          const toolEntry = createToolRow(call);
          const result = results.get(call.id);
          if (result) {
            toolEntry.addResult({
              ok: result.meta?.ok !== false,
              duration_ms: result.meta?.duration_ms,
              preview: result.content || "",
            });
          } else {
            toolEntry.fail();
          }
          list.append(toolEntry.node);
        }
        if (list.children.length) flow.append(list);
      }

      if (article.querySelector(".md, .tool, .reasoning")) {
        root.append(article);
      }
    }
    if (!root.querySelector(".msg")) {
      refreshEmpty();
      return;
    }
    transcript.scrollTop = transcript.scrollHeight;
    pinned = true;
    onScrollState?.(true);
  }

  transcript.addEventListener(
    "scroll",
    () => {
      const next = isPinned();
      if (next !== pinned) {
        pinned = next;
        onScrollState?.(pinned);
      }
    },
    { passive: true },
  );

  root.addEventListener("click", async (event) => {
    const button = event.target instanceof Element ? event.target.closest(".code-copy") : null;
    if (!button) return;
    const code = button.closest(".code-block")?.querySelector("pre code");
    if (!code) return;
    const ok = await copyText(code.textContent || "");
    if (!ok) {
      toast("Could not copy code", { tone: "error" });
      return;
    }
    button.classList.add("is-copied");
    const label = button.querySelector("span");
    if (label) label.textContent = "Copied";
    button.querySelector("use")?.setAttribute("href", "#i-check");
    window.setTimeout(() => {
      if (!button.isConnected) return;
      button.classList.remove("is-copied");
      if (label) label.textContent = "Copy";
      button.querySelector("use")?.setAttribute("href", "#i-copy");
    }, 1600);
  });

  const view = {
    reset,
    renderConversation: openConversation,
    showError,
    beginUser: startUser,
    beginTurn() {
      ensureTurn();
    },
    appendReasoning(text) {
      const turn = ensureTurn();
      turn.reasoning.hidden = false;
      if (!turn.reasoningActive) {
        turn.reasoningActive = true;
        turn.reasoningStarted = performance.now();
        turn.reasoning.classList.add("is-thinking");
        turn.reasoningLabel.textContent = "Thinking…";
        turn.reasoningToggle.setAttribute("aria-expanded", String(turn.reasoningExpanded));
        turn.reasoningBody.hidden = !turn.reasoningExpanded;
      }
      turn.reasoningText += text;
      turn.reasoningBody.textContent = turn.reasoningText;
      scrollIfPinned();
    },
    appendDelta(text) {
      const turn = ensureTurn();
      if (!turn.segment) {
        turn.segment = { element: el("div", { class: "md" }), text: "" };
        turn.flow.append(turn.segment.element);
      }
      if (turn.reasoningActive) finishReasoning(turn);
      turn.activity.hidden = true;
      turn.segment.text += text;
      scheduleRender(turn);
      scrollIfPinned();
    },
    setStatus(label) {
      if (!label) return;
      const turn = ensureTurn();
      turn.activityLabel.textContent = label;
      turn.activity.hidden = false;
      turn.flow.append(turn.activity);
      scrollIfPinned();
    },
    addToolCall(data) {
      const turn = ensureTurn();
      finishReasoning(turn);
      closeSegment(turn);
      turn.activity.hidden = true;
      const entry = createToolRow({
        id: data.id || "",
        name: data.name || data.tool || "tool",
        args: data.args,
      });
      const list = el("div", { class: "tool-list" }, [entry.node]);
      turn.flow.append(list);
      if (data.id) turn.tools.set(data.id, entry);
      scrollIfPinned();
    },
    addToolResult(data) {
      if (!current) return;
      const entry = current.tools.get(data.id);
      if (entry) entry.addResult(data);
      current.activity.hidden = true;
      scrollIfPinned();
    },
    addApproval(data) {
      const turn = ensureTurn();
      finishReasoning(turn);
      closeSegment(turn);
      turn.activity.hidden = true;
      const approvalId = data.approval_id || data.id || "";
      const card = buildApprovalCard(data, (decision) => resolveApproval(approvalId, decision));
      turn.flow.append(card.node);
      pendingApprovals.set(approvalId, card);
      announce("Kiwibro needs approval to continue");
      scrollIfPinned();
    },
    addSteeringChip(text) {
      const turn = ensureTurn();
      const chip = el("span", {
        class: "chip chip-context",
        title: text || "Steering note sent while the reply was running",
      });
      chip.append(icon("message"), el("span", { text: "Steering note applied" }));
      appendChip(turn, chip);
    },
    addContextChip() {
      const turn = ensureTurn();
      if (turn.contextChip) return;
      const chip = el("span", {
        class: "chip chip-context",
        title: "Older messages were summarized to fit the model's context",
      });
      chip.append(icon("layers"), el("span", { text: "Context compacted" }));
      turn.contextChip = chip;
      appendChip(turn, chip);
    },
    addMemoryChip(ops) {
      const turn = ensureTurn();
      const added = Number(ops?.added) || 0;
      const updated = Number(ops?.updated) || 0;
      const removed = Number(ops?.removed) || 0;
      let label = "Memory updated";
      if (added && !updated && !removed) label = added === 1 ? "Memory saved" : `${added} memories saved`;
      else if (updated && !added) label = updated === 1 ? "Memory updated" : `${updated} memories updated`;
      else if (removed && !added && !updated) label = removed === 1 ? "Memory removed" : `${removed} memories removed`;
      const chip = el("span", { class: "chip chip-memory" }, [icon("sparkles"), el("span", { text: label })]);
      appendChip(turn, chip);
    },
    finishDone({ aborted = false } = {}) {
      finishTurn({ aborted });
    },
    prepareRetry() {
      if (!current) return;
      current.error?.remove();
      current.error = null;
      const hasContent = Boolean(current.article.querySelector(".md, .tool, .approval, .chips"));
      if (!hasContent) {
        current.article.remove();
        current = null;
        refreshEmpty();
      } else {
        finishTurn({ aborted: false });
      }
    },
    scrollToBottom(options) {
      pinned = true;
      onScrollState?.(true);
      scrollToBottom(options);
    },
    hasPendingApproval() {
      return pendingApprovals.size > 0;
    },
    denyPending() {
      const first = pendingApprovals.values().next().value;
      if (!first) return false;
      first.node.querySelector(".btn-deny")?.click();
      return true;
    },
    get isPinned() {
      return pinned;
    },
  };

  return view;
}
