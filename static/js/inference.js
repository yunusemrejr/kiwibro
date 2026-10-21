// Inference controls: how much the local model thinks, how fast each reply was,
// and how big the model's context window is. Measured values only — nothing
// here is shown unless the server reported it.

import { api } from "./api.js";
import { el, formatTokens, openPopover, closePopover, popoverIsOpen, toast } from "./ui.js";

const MODES = ["auto", "on", "off"];
const MODE_HELP = {
  auto: "Thinking: auto — thinks first only when a request needs it",
  on: "Thinking: on — always thinks before answering (slower)",
  off: "Thinking: off — answers directly (fastest)",
};
const BULB =
  '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M9.5 18.5h5M10.5 21h3"/>' +
  '<path d="M12 3a6 6 0 0 0-3.7 10.7c.7.6 1.2 1.4 1.2 2.3h5c0-.9.5-1.7 1.2-2.3A6 6 0 0 0 12 3Z"/></svg>';

function seconds(ms) {
  return ms < 10000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms / 1000)} s`;
}

/** One quiet line of measured facts for a finished reply ("" when nothing was measured). */
export function statsLine(inference) {
  if (!inference) return "";
  const parts = [];
  if (Number.isFinite(inference.ttft_ms)) parts.push(`${seconds(inference.ttft_ms)} to first token`);
  if (Number.isFinite(inference.tokens_per_s)) parts.push(`${Math.round(inference.tokens_per_s)} tok/s`);
  if (inference.reuse_percent > 0) parts.push(`${inference.reuse_percent}% cached`);
  if (inference.thinking) parts.push("thought first");
  return parts.join(" · ");
}

function windowLabel(tokens) {
  return tokens >= 1048576 ? `${Math.round(tokens / 1048576)}M` : `${Math.round(tokens / 1024)}K`;
}

export function createInferenceUI({ agentToggle, contextRing, transcript, getState, patchSettings, onModelsChanged }) {
  let context = null;
  const thinkButton = el("button", { class: "agent-toggle think-toggle", type: "button", hidden: "" });
  thinkButton.innerHTML = `${BULB}<span>Think</span><span class="think-mode"></span>`;
  agentToggle?.after(thinkButton);

  function currentModel(models = getState().models || []) {
    return models.find((model) => model.id === getState().currentModel) || null;
  }

  function applySettings(settings = {}) {
    const mode = MODES.includes(settings.reasoning) ? settings.reasoning : "auto";
    // Only models that LM Studio can switch thinking off for get the control.
    thinkButton.hidden = !currentModel()?.reasoning?.includes?.("off");
    thinkButton.dataset.mode = mode;
    thinkButton.querySelector(".think-mode").textContent = mode;
    thinkButton.title = MODE_HELP[mode];
    thinkButton.setAttribute("aria-label", MODE_HELP[mode]);
  }

  thinkButton.addEventListener("click", async () => {
    const previous = thinkButton.dataset.mode || "auto";
    const next = MODES[(MODES.indexOf(previous) + 1) % MODES.length];
    applySettings({ ...getState().settings, reasoning: next });
    try {
      await patchSettings({ reasoning: next });
    } catch {
      applySettings({ ...getState().settings, reasoning: previous });
      toast("Couldn't change thinking mode", { tone: "error", iconName: "alert" });
    }
  });

  function onDone(data) {
    const line = data?.aborted ? "" : statsLine(data?.inference);
    const article = transcript?.querySelector("article.msg-assistant:last-of-type");
    if (!line || !article || article.querySelector(".msg-stats")) return;
    const stats = el("div", { class: "msg-stats" });
    stats.textContent = line;
    stats.title =
      "Measured for this reply on this machine. Cached: share of the prompt LM Studio " +
      "could reuse from its KV cache (estimate).";
    (article.querySelector(".assistant-flow") || article).append(stats);
  }

  function usageText() {
    const used = Number(context?.used) || 0;
    const budget = Number(context?.budget) || 0;
    if (!budget) return "Nothing measured yet";
    if (!used) return `Empty chat · ${formatTokens(budget)} tokens available`;
    return `${formatTokens(used)} of ${formatTokens(budget)} tokens used`;
  }

  // The context ring becomes the place to understand and enlarge the window.
  if (contextRing) {
    contextRing.setAttribute("role", "button");
    contextRing.setAttribute("tabindex", "0");
    contextRing.setAttribute("aria-haspopup", "dialog");
    contextRing.addEventListener("click", toggleContextPopover);
    contextRing.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggleContextPopover();
      }
    });
  }

  async function toggleContextPopover() {
    if (popoverIsOpen()) {
      closePopover();
      return;
    }
    let model = currentModel();
    try {
      const data = await api.models();
      model = currentModel(data.models || []) || model;
    } catch {
      /* offline: show what is cached */
    }
    const body = el("div", { class: "ctx-pop" }, [
      el("div", { class: "menu-label" }, ["Context"]),
      el("p", { class: "ctx-usage" }, [usageText()]),
    ]);
    if (model) appendWindow(body, model);
    // Built completely before opening so the popover is placed for its final size.
    openPopover({ anchor: contextRing, content: body, align: "end", label: "Context window" });
  }

  function appendWindow(body, model) {
    const loaded = model.loaded_config?.context_length || model.context;
    body.append(
      el("p", { class: "ctx-window" }, [
        `${model.display_name || model.id} · ${windowLabel(loaded)} window`,
        model.max_context > loaded ? el("span", { class: "ctx-max" }, [` · supports ${windowLabel(model.max_context)}`]) : "",
      ]),
    );
    const choices = model.context_choices || [];
    if (!choices.length) return;
    const row = el("div", { class: "ctx-choices", role: "group", "aria-label": "Load a larger window" });
    for (const size of choices) {
      const button = el("button", { class: "chip ctx-choice", type: "button" }, [windowLabel(size)]);
      button.addEventListener("click", () => resize(model, size, row));
      row.append(button);
    }
    body.append(
      row,
      el("p", { class: "ctx-hint" }, [
        `Reloads the model with a larger window for long chats. Uses more memory; ` +
          `if it doesn't fit, ${windowLabel(loaded)} is kept.`,
      ]),
    );
  }

  async function resize(model, size, row) {
    const buttons = row.querySelectorAll("button");
    buttons.forEach((button) => (button.disabled = true));
    row.setAttribute("aria-busy", "true");
    try {
      await api.resizeContext(model.id, size);
      closePopover();
      toast(`${model.display_name || model.id} now has a ${windowLabel(size)} window`);
      onModelsChanged?.();
    } catch (error) {
      buttons.forEach((button) => (button.disabled = false));
      row.removeAttribute("aria-busy");
      toast(error?.message || "LM Studio couldn't reload the model", { tone: "error", iconName: "alert" });
    }
  }

  return { applySettings, onDone, setContext: (stats) => (context = stats) };
}
