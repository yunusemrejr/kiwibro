#!/usr/bin/env node
/**
 * Kiwibro browser E2E rig (playwright-core, real headless Chrome).
 *
 * Modes
 *   --mode mock   drive the UI against its in-browser mock backend (?mock=1).
 *                 No backend state needed; used to validate the rig and UI.
 *   --mode fake   drive the real Flask app against tests/e2e/fake_lm.py
 *                 (deterministic streaming + approval, no LM Studio).
 *   --mode live   drive the real Flask app against the real LM Studio
 *                 (slow: first call can JIT-load for 30–120 s).
 *   --base URL    attach to an already-running server instead of spawning one.
 *
 * What it does
 *   checks  – console/page errors, failed requests, XSS renderer probes, SSE
 *             parser probes, API/UI contract, a11y basics, responsive overflow.
 *   flows   – hero, models, chat, approval, memory, settings, mobile; each flow
 *             saves a screenshot named exactly docs/screenshots/<name>.png.
 *
 * Usage
 *   node tests/e2e/browser/run_browser.mjs --mode mock --out .dev/qa/browser-shots
 *   node tests/e2e/browser/run_browser.mjs --mode live --out docs/screenshots
 */

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

import { REPO_ROOT, killAll, sleep, startApp, startFakeLm } from "./app_server.mjs";
import { APPROVAL_PROMPT, SEL, SIMPLE_PROMPT } from "./selectors.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

/* ------------------------------------------------------------------ args */

function parseArgs(argv) {
  const options = {
    mode: "mock",
    base: "",
    out: ".dev/qa/browser/shots",
    only: "checks,hero,models,chat,approval,memory,settings,mobile",
    settle: 550,
    chrome: process.env.CHROME_PATH || "/usr/bin/google-chrome",
    headless: true,
    viewport: "1440x900",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const [key, inline] = arg.split("=", 2);
    const next = () => inline ?? argv[++index];
    switch (key) {
      case "--mode":
        options.mode = next();
        break;
      case "--base":
        options.base = next();
        break;
      case "--out":
        options.out = next();
        break;
      case "--only":
        options.only = next();
        break;
      case "--settle":
        options.settle = Number(next());
        break;
      case "--chrome":
        options.chrome = next();
        break;
      case "--viewport":
        options.viewport = next();
        break;
      case "--headed":
        options.headless = false;
        break;
      case "--help":
      case "-h":
        console.log(fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("*/")[0]);
        process.exit(0);
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  if (!["mock", "fake", "live"].includes(options.mode)) {
    throw new Error(`--mode must be mock|fake|live, got ${options.mode}`);
  }
  return options;
}

function loadPlaywright() {
  const candidates = [
    process.env.PLAYWRIGHT_CORE_PATH,
    path.join(REPO_ROOT, ".dev", "qa", "node_modules", "playwright-core"),
    "playwright-core",
  ].filter(Boolean);
  const errors = [];
  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch (error) {
      errors.push(String(error.message).split("\n")[0]);
    }
  }
  throw new Error(
    `playwright-core not found. Install with:\n  cd ${path.join(REPO_ROOT, ".dev", "qa")} && npm i playwright-core\n` +
      `or set PLAYWRIGHT_CORE_PATH. Tried: ${errors.join(" | ")}`,
  );
}

/* --------------------------------------------------------------- results */

const results = [];

function record(area, name, ok, detail = "") {
  results.push({ area, name, ok, detail });
  const status = ok ? "ok  " : "FAIL";
  console.log(`  ${status} [${area}] ${name}${detail ? ` — ${detail}` : ""}`);
}

async function guarded(area, name, fn) {
  const started = Date.now();
  try {
    const detail = await fn();
    record(area, name, true, `${detail || ""} (${Date.now() - started}ms)`);
    return true;
  } catch (error) {
    record(area, name, false, String(error?.message || error).split("\n")[0]);
    return false;
  }
}

/* --------------------------------------------------------------- screenshot */

async function settle(page, ms) {
  await page.evaluate(() => (document.fonts ? document.fonts.ready : undefined));
  await page.waitForTimeout(ms);
}

async function shot(page, outDir, name, settleMs) {
  fs.mkdirSync(outDir, { recursive: true });
  await settle(page, settleMs);
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
  });
  const file = path.join(outDir, `${name}.png`);
  await page.screenshot({ path: file, type: "png", animations: "disabled" });
  return file;
}

/* ------------------------------------------------------------------- checks */

async function checkConsole(page, state) {
  const problems = [...state.consoleErrors, ...state.pageErrors];
  if (problems.length) throw new Error(`${problems.length} console/page error(s): ${problems.slice(0, 3).join(" | ")}`);
  return "0 console errors / 0 page errors";
}

async function checkNoExternal(page, state) {
  if (state.externalRequests.length) {
    throw new Error(`external requests: ${state.externalRequests.slice(0, 5).join(", ")}`);
  }
  return "zero external network requests (offline-capable, no CDNs/fonts)";
}

async function checkRequests(_page, state) {
  if (state.requestFailures.length) {
    throw new Error(`failed request(s): ${state.requestFailures.slice(0, 4).join(" | ")}`);
  }
  const bad = state.badResponses.filter((entry) => entry.status >= 500);
  if (bad.length) throw new Error(`5xx responses: ${bad.slice(0, 3).join(" | ")}`);
  return "no failed requests, no 5xx responses";
}

async function checkBootstrapContract(page) {
  return page.evaluate(async () => {
    // go through the app's own api client so ?mock=1 exercises the mock backend
    const { api } = await import("/static/js/api.js");
    const data = await api.bootstrap();
    const pill = document.querySelector("#model-pill-name")?.textContent?.trim() || "";
    const expectedModel = data.current_model || (data.models?.[0]?.id ?? "");
    const problems = [];
    if (!data.health || data.health.ok !== true) problems.push("health.ok");
    if (!Array.isArray(data.models)) problems.push("models[]");
    if (!data.settings || !data.settings.permissions) problems.push("settings.permissions");
    if (typeof data.memories_count !== "number") problems.push("memories_count");
    if (!data.workspace) problems.push("workspace");
    if (expectedModel && pill && !pill.includes(expectedModel)) {
      problems.push(`model pill ${JSON.stringify(pill)} != ${expectedModel}`);
    }
    if (problems.length) throw new Error(problems.join(", "));
    return `model=${expectedModel || "(offline)"} memories=${data.memories_count}`;
  });
}

async function checkMarkdownXss(page) {
  return page.evaluate(async () => {
    window.__xss = undefined;
    const module = await import("/static/js/markdown.js");
    const payloads = [
      '<img src=x onerror="window.__xss=1">',
      "<script>window.__xss=2</script>",
      "[click](javascript:window.__xss=3)",
      '<a href="javascript:alert(1)">x</a>',
      "![remote](http://127.0.0.1:9/pixel.png)",
      "<svg onload=window.__xss=4>",
      '<iframe src="data:text/html,<script>parent.__xss=5</script>"></iframe>',
      '**bold** `<img src=x onerror=window.__xss=6>`',
      "[ok](https://example.com/) [bad](data:text/html,x)",
    ];
    const host = document.createElement("div");
    host.id = "xss-probe";
    host.style.position = "absolute";
    host.style.left = "-10000px";
    document.body.append(host);
    for (const payload of payloads) {
      const node = document.createElement("div");
      node.innerHTML = module.renderMarkdown(payload);
      host.append(node);
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
    const problems = [];
    if (window.__xss !== undefined) problems.push(`script executed (${window.__xss})`);
    for (const bad of host.querySelectorAll("script, img, svg, iframe, object, embed")) {
      problems.push(`dangerous tag rendered: <${bad.tagName.toLowerCase()}>`);
    }
    for (const anchor of host.querySelectorAll("a")) {
      const href = anchor.getAttribute("href") || "";
      if (!/^https?:\/\//i.test(href)) problems.push(`unsafe href: ${href.slice(0, 40)}`);
      if (!(anchor.getAttribute("rel") || "").includes("noopener")) problems.push("link missing noopener");
      if ((anchor.getAttribute("target") || "") !== "_blank") problems.push("link missing target=_blank");
    }
    const code = host.querySelector("code");
    if (!code || !code.textContent.includes("<img")) problems.push("inline code did not stay escaped");
    host.remove();
    if (problems.length) throw new Error(problems.slice(0, 5).join("; "));
    return `${payloads.length} hostile markdown payloads stayed inert`;
  });
}

async function checkSseParser(page) {
  return page.evaluate(async () => {
    const { consumeSSE } = await import("/static/js/api.js");
    const encoder = new TextEncoder();
    const text =
      'event: meta\ndata: {"a":"Grüße 🥝"}\n\n' +
      ': ping\n\n' +
      'event: delta\ndata: {"text":"hi "}\n\n' +
      'event: delta\ndata: {"text":"there"}\n\n' +
      'event: done\ndata: {"ok":true,"n":3}\n\n';
    const bytes = encoder.encode(text);
    // split on every possible boundary class, including inside the emoji bytes
    const cuts = [1, 7, 19, 31, 45, 60, 79, bytes.length - 3];
    const chunks = [];
    let last = 0;
    for (const cut of cuts) {
      if (cut > last && cut < bytes.length) {
        chunks.push(bytes.slice(last, cut));
        last = cut;
      }
    }
    chunks.push(bytes.slice(last));
    const stream = new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    });
    const events = [];
    const saw = await consumeSSE(new Response(stream, { headers: { "Content-Type": "text/event-stream" } }), (event, data) => {
      events.push([event, data]);
    });
    const problems = [];
    if (!saw) problems.push("consumeSSE returned false");
    if (events.length !== 4) problems.push(`expected 4 events, got ${events.length}`);
    if (events[0]?.[0] !== "meta" || events[0]?.[1]?.a !== "Grüße 🥝") problems.push("meta frame/UTF-8 split failed");
    if (events.filter((entry) => entry[0] === "delta").length !== 2) problems.push("delta frames lost");
    if (events.at(-1)?.[0] !== "done" || events.at(-1)?.[1]?.ok !== true) problems.push("done frame lost");
    if (problems.length) throw new Error(problems.join("; "));
    return `SSE frames survived ${chunks.length} arbitrary chunk cuts incl. UTF-8 emoji split`;
  });
}

async function checkA11y(page) {
  return page.evaluate(() => {
    const problems = [];
    const name = (node) =>
      (node.getAttribute("aria-label") || node.getAttribute("title") || node.textContent || "").trim();
    for (const button of document.querySelectorAll("button")) {
      if (!name(button)) problems.push(`unnamed button: ${button.outerHTML.slice(0, 80)}`);
    }
    for (const input of document.querySelectorAll("input, textarea, select")) {
      const labelled =
        input.getAttribute("aria-label") ||
        input.getAttribute("aria-labelledby") ||
        (input.id && document.querySelector(`label[for="${CSS.escape(input.id)}"]`)) ||
        input.closest("label");
      if (!labelled) problems.push(`unlabelled control: ${input.id || input.tagName}`);
    }
    for (const image of document.querySelectorAll("img")) {
      if (!image.hasAttribute("alt")) problems.push(`img without alt: ${image.getAttribute("src")}`);
    }
    if (document.documentElement.lang !== "en") problems.push("html lang missing");
    if (!document.querySelector('meta[name="viewport"]')) problems.push("viewport meta missing");
    if (!document.querySelector('meta[name="color-scheme"]')) problems.push("color-scheme meta missing");
    for (const dialog of document.querySelectorAll("dialog")) {
      const labelledBy = dialog.getAttribute("aria-labelledby");
      if (!labelledBy || !document.getElementById(labelledBy)) problems.push(`dialog without label: ${dialog.id}`);
    }
    for (const node of document.querySelectorAll("[tabindex]")) {
      if (Number(node.getAttribute("tabindex")) > 0) problems.push(`positive tabindex: ${node.id || node.tagName}`);
    }
    const headings = document.querySelectorAll("h1");
    if (!headings.length) problems.push("no h1");
    if (problems.length) throw new Error(problems.slice(0, 6).join("; "));
    return "buttons/inputs/images/dialogs labelled, lang/viewport/color-scheme set";
  });
}

async function checkResponsive(page) {
  const viewport = page.viewportSize();
  return page.evaluate(() => {
    const root = document.documentElement;
    const overflow = root.scrollWidth - root.clientWidth;
    const wide = [];
    for (const node of document.querySelectorAll("header, main, aside, .composer, .dock, .transcript")) {
      const rect = node.getBoundingClientRect();
      if (rect.right > window.innerWidth + 2 || rect.left < -2) wide.push(node.className || node.tagName);
    }
    const problems = [];
    if (overflow > 2) problems.push(`horizontal overflow ${overflow}px`);
    if (wide.length) problems.push(`elements past viewport: ${wide.slice(0, 3).join(", ")}`);
    if (!document.querySelector("#composer-input")) problems.push("composer missing");
    if (problems.length) throw new Error(problems.join("; "));
    return `no horizontal overflow at ${window.innerWidth}x${window.innerHeight}`;
  }).then((detail) => `${detail} (viewport ${viewport.width})`);
}

async function checkShortcuts(page) {
  const problems = [];
  // Ctrl+N -> new chat / empty state
  await page.keyboard.press("Control+n");
  await page.waitForSelector(`${SEL.emptyState}:not([hidden])`, { timeout: 5000 }).catch(() => {
    problems.push("Ctrl+N did not open a new chat");
  });
  // Ctrl+, -> settings sheet; Esc closes it
  await page.keyboard.press("Control+,");
  try {
    await page.waitForSelector(`${SEL.settingsSheet}[open]`, { timeout: 6000 });
  } catch {
    problems.push("Ctrl+, did not open settings");
  }
  await page.keyboard.press("Escape");
  await page
    .waitForSelector(`${SEL.settingsSheet}[open]`, { state: "detached", timeout: 6000 })
    .catch(() => problems.push("Esc did not close settings"));
  // Enter sends, Shift+Enter inserts a newline
  const enabled = await page.evaluate(() => !document.querySelector("#composer-input")?.disabled);
  if (enabled) {
    await page.fill(SEL.composerInput, "shortcut line one");
    await page.keyboard.press("Shift+Enter");
    await page.keyboard.type("line two");
    const value = await page.inputValue(SEL.composerInput);
    if (!value.includes("\n")) problems.push("Shift+Enter did not insert a newline");
    await page.fill(SEL.composerInput, "");
  }
  if (problems.length) throw new Error(problems.join("; "));
  return "Ctrl+N, Ctrl+,, Esc and Shift+Enter behave as documented";
}

/* -------------------------------------------------------------------- flows */

async function ensureEmptyState(page) {
  if (await page.locator(`${SEL.emptyState}:not([hidden])`).count()) return;
  await page.click(SEL.newChat);
  await page.waitForSelector(`${SEL.emptyState}:not([hidden])`, { timeout: 10000 });
}

async function resetUi(page) {
  // Close popovers/sheets and stop any in-flight stream so one failed flow cannot
  // cascade into the next one.
  await page.keyboard.press("Escape").catch(() => {});
  await settle(page, 120);
  await page.keyboard.press("Escape").catch(() => {});
  const stop = page.locator(SEL.stopBtn);
  if (await stop.count()) {
    const hidden = await stop.first().evaluate((node) => node.hidden);
    if (!hidden) {
      await page.keyboard.press("Escape").catch(() => {});
      await waitStreamIdle(page, 30000).catch(() => {});
    }
  }
  for (const dialog of await page.locator("dialog[open]").all()) {
    await page.keyboard.press("Escape").catch(() => {});
    await settle(page, 120);
    void dialog;
  }
}

async function openSeededConversation(page, title) {
  const row = page.locator(".conv-row", { hasText: title }).first();
  if (!(await row.count())) return false;
  await row.locator(".conv-open").click();
  await page.waitForSelector(SEL.assistantMarkdown, { state: "visible", timeout: 20000 });
  await settle(page, 250);
  return true;
}

async function waitStreamIdle(page, timeoutMs) {
  await page.waitForFunction(
    () => {
      const stop = document.querySelector("#stop-btn");
      return !stop || stop.hidden;
    },
    null,
    { timeout: timeoutMs },
  );
}

async function flowHero(page, outDir, options) {
  await resetUi(page);
  await ensureEmptyState(page);
  await page.waitForFunction(
    () => {
      const text = document.querySelector("#model-pill-name")?.textContent?.trim() || "";
      return text && text !== "Loading…";
    },
    null,
    { timeout: 15000 },
  );
  return shot(page, outDir, "hero", options.settle);
}

async function flowModels(page, outDir, options) {
  await resetUi(page);
  await page.click(SEL.modelPill);
  await page.waitForSelector(SEL.modelPopover, { timeout: 8000 });
  await page.waitForFunction(
    () => document.querySelectorAll(".popover .menu-item").length > 0,
    null,
    { timeout: 8000 },
  );
  const file = await shot(page, outDir, "models", options.settle);
  // switch away from the current model and back, proving selection works
  const current = (await page.textContent(SEL.modelPillName)).trim();
  const items = page.locator(`${SEL.modelPopover} .menu-item`);
  const count = await items.count();
  let switched = false;
  for (let index = 0; index < count; index += 1) {
    const label = (await items.nth(index).innerText()).split("\n")[0].trim();
    if (label && label !== current) {
      await items.nth(index).click();
      await page.waitForFunction(
        (previous) => document.querySelector("#model-pill-name")?.textContent?.trim() !== previous,
        current,
        { timeout: 10000 },
      );
      switched = true;
      break;
    }
  }
  await page.keyboard.press("Escape");
  await page.waitForSelector(SEL.modelPopover, { state: "detached", timeout: 5000 }).catch(() => {});
  if (!switched) throw new Error("model popover had no alternative model to switch to");
  // restore the original selection so later flows/screenshots stay canonical
  await page.click(SEL.modelPill);
  await page.waitForSelector(SEL.modelPopover, { timeout: 5000 });
  const rows = page.locator(`${SEL.modelPopover} .menu-item`);
  for (let index = 0; index < (await rows.count()); index += 1) {
    const label = (await rows.nth(index).innerText()).split("\n")[0].trim();
    if (label === current) {
      await rows.nth(index).click();
      break;
    }
  }
  await page.waitForFunction(
    (expected) => document.querySelector("#model-pill-name")?.textContent?.trim() === expected,
    current,
    { timeout: 10000 },
  ).catch(() => {});
  await page.keyboard.press("Escape");
  return file;
}

async function flowChat(page, outDir, options) {
  await resetUi(page);
  if (options.mode === "live") {
    const opened = await openSeededConversation(page, "Write a Python script");
    if (!opened) throw new Error("seeded conversation 'Write a Python script' not found");
    const before = await page.locator(`${SEL.assistantMarkdown}`).count();
    await page.fill(SEL.composerInput, "Add two short comments to the code and show the snippet again.");
    await page.keyboard.press("Enter");
    await page.waitForFunction(
      (previous) => document.querySelectorAll(".msg-assistant .md").length > previous,
      before,
      { timeout: 240000 },
    );
    await waitStreamIdle(page, 240000);
    await page.evaluate(() => {
      const transcript = document.querySelector("#transcript");
      if (transcript) transcript.scrollTop = transcript.scrollHeight;
    });
    return shot(page, outDir, "chat", options.settle);
  }
  await ensureEmptyState(page);
  const timeout = options.mode === "live" ? 240000 : 45000;
  const chatPrompt =
    options.mode === "live"
      ? "In one short sentence, what is a kiwi bird? Answer from your own knowledge, no tools needed."
      : SIMPLE_PROMPT;
  await page.fill(SEL.composerInput, chatPrompt);
  await page.keyboard.press("Enter");
  await page.waitForSelector(SEL.userBubble, { timeout: 15000 });
  await page.waitForSelector(SEL.assistantMarkdown, { state: "visible", timeout });
  await page.waitForFunction(
    () => (document.querySelector(".msg-assistant .md")?.textContent?.trim().length || 0) > 10,
    null,
    { timeout },
  );
  await waitStreamIdle(page, timeout);
  await page.waitForSelector(SEL.streamCaret, { state: "detached", timeout: 15000 }).catch(() => {});
  await page.evaluate(() => {
    const transcript = document.querySelector("#transcript");
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  });
  return shot(page, outDir, "chat", options.settle);
}

async function flowApproval(page, outDir, options) {
  await resetUi(page);
  const timeout = options.mode === "live" ? 90000 : 45000;
  let prompt = APPROVAL_PROMPT;
  if (options.mode === "live") {
    const opened = await openSeededConversation(page, "Write a Python script");
    if (!opened) throw new Error("seeded conversation 'Write a Python script' not found");
    prompt =
      "Use the write_file tool to save this Python script as scripts/rename_by_date.py, " +
      "then tell me it is saved:\n\n```python\nfrom datetime import date\nfrom pathlib import Path\n\n" +
      "for path in sorted(Path('.').glob('*.txt')):\n    " +
      "path.rename(path.with_name(f'{date.today()}-{path.name}'))\n```";
  } else if (options.mode === "fake") {
    prompt = `${APPROVAL_PROMPT} [[scenario:tool_write]]`;
  } else {
    await page.click(SEL.newChat).catch(() => {});
    await settle(page, 150);
  }
  await page.fill(SEL.composerInput, prompt);
  await page.keyboard.press("Enter");
  await page.waitForSelector(SEL.approval, { state: "visible", timeout });
  await page.waitForFunction(
    () => {
      const card = document.querySelector(".approval");
      return card && card.querySelectorAll(".btn").length >= 3;
    },
    null,
    { timeout: 10000 },
  );
  const file = await shot(page, outDir, "approval", options.settle);
  // Continue the flow so the UI ends in a clean state, and prove Allow works.
  await page.click(SEL.approvalAllow);
  await page.waitForSelector(SEL.approvalResolved, { timeout: 30000 });
  await waitStreamIdle(page, timeout);
  const evidenceDir = path.resolve(REPO_ROOT, ".dev/qa/browser/screens");
  await shot(page, evidenceDir, "approval-resolved", options.settle).catch(() => {});
  return file;
}

async function flowMemory(page, outDir, options) {
  await resetUi(page);
  await page.click(SEL.memoryBtn);
  await page.waitForSelector(`${SEL.memorySheet}[open]`, { timeout: 8000 });
  await page.waitForFunction(
    () => {
      const list = document.querySelector("#memory-list");
      return list && list.children.length > 0;
    },
    null,
    { timeout: 8000 },
  ).catch(async () => {
    // add one through the UI so the screenshot is not an empty sheet
    await page.fill(SEL.memoryNew, "Prefers dark themes and kiwi green");
    await page.fill(SEL.memoryNew, "Prefers dark themes and kiwi green");
    await page.click(SEL.memoryAddButton);
    await page.waitForSelector(SEL.memoryRow, { timeout: 8000 });
  });
  const file = await shot(page, outDir, "memory", options.settle);
  // CRUD through the real sheet: add -> pin -> delete
  const unique = `QA browser memory ${Date.now()}`;
  await page.fill(SEL.memoryNew, unique);
  await page.selectOption(SEL.memoryKind, "preference");
  await page.click(SEL.memoryAddButton);
  await page.waitForFunction(
    (text) => Array.from(document.querySelectorAll(".memory-item")).some((row) => row.textContent.includes(text)),
    unique,
    { timeout: 10000 },
  );
  const row = page.locator(SEL.memoryItem, { hasText: unique }).first();
  await row.locator(SEL.memoryPin).click();
  await page.waitForFunction(
    (text) => Array.from(document.querySelectorAll(".memory-item")).some(
      (item) => item.textContent.includes(text) && item.classList.contains("is-pinned"),
    ),
    unique,
    { timeout: 10000 },
  );
  await row.locator(SEL.memoryDelete).click();
  await page.waitForFunction(
    (text) => !Array.from(document.querySelectorAll(".memory-item")).some((item) => item.textContent.includes(text)),
    unique,
    { timeout: 10000 },
  );
  await page.keyboard.press("Escape");
  await page.waitForSelector(SEL.dialogOpen, { state: "detached", timeout: 5000 }).catch(() => {});
  return file;
}

async function flowSettings(page, outDir, options) {
  await resetUi(page);
  await page.click(SEL.settingsBtn);
  await page.waitForSelector(`${SEL.settingsSheet}[open]`, { timeout: 8000 });
  await page.waitForFunction(
    () => {
      const body = document.querySelector("#settings-body");
      return body && body.textContent.trim().length > 30;
    },
    null,
    { timeout: 10000 },
  );
  const file = await shot(page, outDir, "settings", options.settle);
  // toggle Agent mode off and on again, proving the switch round-trips the API
  const agentSwitch = page.locator(SEL.agentSwitch).first();
  if (await agentSwitch.count()) {
    const before = await agentSwitch.getAttribute("aria-checked");
    await agentSwitch.click();
    await page.waitForFunction(
      (previous) =>
        document.querySelector('button[role="switch"][aria-label="Agent mode"]')?.getAttribute("aria-checked") !==
        previous,
      before,
      { timeout: 10000 },
    );
    const after = await agentSwitch.getAttribute("aria-checked");
    const settings = await page.evaluate(async () => {
      const { api } = await import("/static/js/api.js");
      const data = await api.settings();
      return data.settings;
    });
    const expected = after === "true";
    if (Boolean(settings.agent_enabled) !== expected) throw new Error("agent toggle did not persist");
    await agentSwitch.click();
    await page.waitForFunction(
      (previous) =>
        document.querySelector('button[role="switch"][aria-label="Agent mode"]')?.getAttribute("aria-checked") !==
        previous,
      after,
      { timeout: 10000 },
    );
  }
  await page.keyboard.press("Escape");
  await page.waitForSelector(SEL.dialogOpen, { state: "detached", timeout: 5000 }).catch(() => {});
  return file;
}

async function flowMobile(page, outDir, options) {
  await resetUi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await settle(page, 300);
  if (options.mode === "live") {
    // committed mobile shot = clean empty state at 390 px (Lead's preference)
    await page.click(SEL.newChat).catch(() => {});
    await page.waitForSelector(`${SEL.emptyState}:not([hidden])`, { timeout: 10000 });
    return shot(page, outDir, "mobile", options.settle);
  }
  const hasMessages = await page.locator(`${SEL.assistantMessage}`).count();
  if (!hasMessages) {
    const timeout = options.mode === "live" ? 180000 : 30000;
    await page.fill(SEL.composerInput, SIMPLE_PROMPT);
    await page.keyboard.press("Enter");
    await page.waitForSelector(SEL.assistantMarkdown, { state: "visible", timeout });
    await waitStreamIdle(page, timeout);
  }
  await page.evaluate(() => {
    const transcript = document.querySelector("#transcript");
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  });
  await settle(page, 400);
  return shot(page, outDir, "mobile", options.settle);
}

/* --------------------------------------------------------------------- main */

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, options.out);
  const wanted = new Set(options.only.split(",").map((entry) => entry.trim()).filter(Boolean));
  const { chromium } = loadPlaywright();
  if (!fs.existsSync(options.chrome)) {
    throw new Error(`Chrome binary not found: ${options.chrome} (set CHROME_PATH)`);
  }

  let app = null;
  let fake = null;
  let baseUrl = options.base;
  try {
    if (!baseUrl) {
      if (options.mode === "mock") {
        app = await startApp({ label: "browser-mock" });
      } else if (options.mode === "fake") {
        fake = await startFakeLm({ models: "default" });
        app = await startApp({ label: "browser-fake", lmstudioUrl: fake.url });
      } else {
        app = await startApp({ label: "browser-live", lmstudioUrl: process.env.KIWIBRO_LMSTUDIO_URL || "http://127.0.0.1:1234" });
      }
      baseUrl = app.baseUrl;
    }
    const query = options.mode === "mock" ? "?mock=1" : "";
    const [width, height] = options.viewport.split("x").map(Number);

    console.log(`Kiwibro browser rig — mode=${options.mode} base=${baseUrl}${query}`);

    const browser = await chromium.launch({
      executablePath: options.chrome,
      headless: options.headless,
      args: ["--no-sandbox", "--disable-gpu", "--force-color-profile=srgb", "--font-render-hinting=none"],
    });
    const context = await browser.newContext({
      viewport: { width, height },
      deviceScaleFactor: 2,
      colorScheme: "dark",
    });
    const page = await context.newPage();
    page.setDefaultTimeout(15000);

    const state = { consoleErrors: [], pageErrors: [], requestFailures: [], badResponses: [], externalRequests: [] };
    const origin = new URL(baseUrl).origin;
    page.on("request", (request) => {
      const url = request.url();
      if (!url.startsWith(origin) && !url.startsWith("data:") && !url.startsWith("about:")) {
        state.externalRequests.push(url);
      }
    });
    page.on("console", (message) => {
      if (message.type() === "error") state.consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => state.pageErrors.push(String(error.message || error)));
    page.on("requestfailed", (request) => {
      const failure = request.failure()?.errorText || "failed";
      if (!request.url().includes("favicon")) state.requestFailures.push(`${request.url()} (${failure})`);
    });
    page.on("response", (response) => {
      if (response.status() >= 400) state.badResponses.push({ url: response.url(), status: response.status() });
    });

    await page.goto(`${baseUrl}/${query}`, { waitUntil: "networkidle" });
    await page.waitForSelector(SEL.app, { timeout: 15000 });
    await page.evaluate(() => localStorage.clear());
    await page.goto(`${baseUrl}/${query}`, { waitUntil: "networkidle" });
    await page.waitForSelector(`${SEL.app} .headerbar`, { timeout: 15000 });
    await settle(page, 400);

    const checks = [
      ["console", checkConsole],
      ["resources", checkRequests],
      ["no-external", checkNoExternal],
      ["contract", checkBootstrapContract],
      ["xss", checkMarkdownXss],
      ["sse", checkSseParser],
      ["a11y", checkA11y],
      ["responsive-desktop", checkResponsive],
      ["shortcuts", checkShortcuts],
    ];
    if (wanted.has("checks")) {
      console.log("\n== checks ==");
      for (const [name, fn] of checks) await guarded("check", name, () => fn(page, state));
      await page.setViewportSize({ width: 390, height: 844 });
      await guarded("check", "responsive-mobile", () => checkResponsive(page));
      await page.setViewportSize({ width, height });
    }

    const flows = [
      ["hero", flowHero],
      ["models", flowModels],
      ["chat", flowChat],
      ["approval", flowApproval],
      ["memory", flowMemory],
      ["settings", flowSettings],
      ["mobile", flowMobile],
    ];
    const shotFlows = flows.filter(([name]) => wanted.has(name));
    if (shotFlows.length) {
      console.log(`\n== flows (screenshots -> ${outDir}) ==`);
      for (const [name, fn] of shotFlows) {
        await guarded("shot", name, async () => {
          const file = await fn(page, outDir, options);
          return path.relative(REPO_ROOT, file);
        });
      }
      // Re-check console after the flows too.
      await guarded("check", "console-after-flows", () => checkConsole(page, state));
    }

    await context.close();
    await browser.close();
  } finally {
    if (app) await app.stop();
    if (fake) await fake.stop();
    killAll();
  }

  const failures = results.filter((entry) => !entry.ok);
  const summaryPath = path.resolve(REPO_ROOT, ".dev", "qa", "browser", "browser-evidence.json");
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(
    summaryPath,
    JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        mode: options.mode,
        base: options.base || "(spawned)",
        total: results.length,
        passed: results.length - failures.length,
        failed: failures.length,
        results,
      },
      null,
      2,
    ),
  );
  console.log(`\n${results.length - failures.length}/${results.length} browser checks passed — ${summaryPath}`);
  if (failures.length) {
    console.log("\nFAILURES:");
    for (const failure of failures) console.log(`  - [${failure.area}] ${failure.name}: ${failure.detail}`);
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(`browser rig failed: ${error?.stack || error}`);
  killAll();
  process.exit(1);
});
