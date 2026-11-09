/**
 * Models panel: the models on this computer, the engine that runs them, and
 * downloads from Hugging Face. Renders into #models-body of the models sheet.
 */
import { el, icon, debounce, toast, openSheet } from "./ui.js";

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;
const POLL_MS = 1000;
const SEARCH_MS = 350;
const BUSY_INSTALL = ["checking", "downloading", "verifying", "unpacking"];
const ACTIVE_JOB = ["queued", "downloading"];
const FINISHED_JOB = ["done", "cancelled", "error"];
const FIT_TEXT = { good: "Fits in memory", tight: "Tight fit", too_big: "Too big for this computer" };
const PHASE_TEXT = {
  checking: "Looking for the newest build…",
  verifying: "Checking the download…",
  unpacking: "Unpacking…",
};
const WAIT_FOR_REPLY = "Wait for the current reply to finish.";
const BUILD_KIND = { vulkan: "Vulkan", cpu: "CPU" };

/* ---------- Formatting ---------------------------------------------------- */

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value <= 0) return "0 MB";
  const mb = value / MIB;
  if (mb < 1) return "< 1 MB";
  if (mb < 1024) return `${Math.round(mb)} MB`;
  const gb = value / GIB;
  return gb < 10 ? `${gb.toFixed(1)} GB` : `${Math.round(gb)} GB`;
}

function formatCount(count) {
  const value = Number(count) || 0;
  if (value < 1000) return String(value);
  const thousands = Math.round(value / 1000);
  if (thousands < 1000) return `${thousands}K`;
  return `${(value / 1e6).toFixed(1)}M`;
}

function formatEta(seconds) {
  const total = Math.round(Number(seconds) || 0);
  if (total <= 0) return "";
  if (total < 60) return `${total} s`;
  if (total < 3600) return `${Math.round(total / 60)} min`;
  let hours = Math.floor(total / 3600);
  let minutes = Math.round((total % 3600) / 60);
  if (minutes === 60) {
    hours += 1;
    minutes = 0;
  }
  return minutes ? `${hours} h ${minutes} min` : `${hours} h`;
}

function countOf(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

function errorText(error) {
  return error?.message || "Something went wrong. Try again.";
}

function isAbort(error) {
  return error?.name === "AbortError";
}

function isBusy(installer) {
  return BUSY_INSTALL.includes(installer?.state);
}

function installerShape(installer) {
  if (!installer) return "none";
  return isBusy(installer) ? "busy" : `${installer.state}:${installer.supported}`;
}

function sourceKind(source) {
  return source === "kiwibro" || source === "lmstudio" ? source : "other";
}

function jobPercent(job) {
  return job.total > 0 ? Math.min(100, (job.received / job.total) * 100) : 0;
}

function jobsKey(jobs) {
  return jobs.map((job) => `${job.id}:${job.state}`).join("|");
}

function installPercent(installer) {
  if (installer.state === "downloading" && installer.total > 0) {
    return Math.min(100, (installer.received / installer.total) * 100);
  }
  return 100;
}

function installText(installer) {
  if (installer.state === "downloading") {
    return `Downloading… ${formatBytes(installer.received)} of ${formatBytes(installer.total)}`;
  }
  return PHASE_TEXT[installer.state] || "Working…";
}

/** The file to suggest: Q4_K_M when it fits, else the largest good-fit file. Split parts are never suggested. */
function pickRecommended(files) {
  const usable = files.filter((file) => !file.projector && !file.split);
  const q4 = usable.find(
    (file) => String(file.quant || "").toUpperCase() === "Q4_K_M" && file.fit !== "too_big",
  );
  if (q4) return q4;
  const good = usable
    .filter((file) => file.fit === "good")
    .sort((a, b) => (b.size_bytes || 0) - (a.size_bytes || 0));
  return good[0] || null;
}

/* ---------- The panel ------------------------------------------------------ */

export function createModelsView({ dialog, api, getState, actions }) {
  const body = dialog.querySelector("#models-body");
  body.classList.add("mdl-panel");

  const state = {
    session: 0, // bumps on every open and close, so late answers are dropped
    library: null,
    jobs: [],
    installer: null,
    slots: null, // local mode: the blocks of the panel
    rows: new Map(), // model id -> row, for the inline delete confirm
    confirmId: "",
    expandToken: null, // identifies the repository whose file list is open
  };
  let mode = ""; // "local" or "attached", as last built
  let hub = null; // { input, results }, rebuilt on every open
  let hubController = null;
  let hubSeq = 0;
  let fileListSeq = 0;
  let filesCache = new Map(); // repo -> file list, per open
  const fileInfo = new WeakMap(); // file row -> { repo, file, recommended }
  let pollTimer = 0;
  let ticking = false;
  const search = debounce(() => runSearch(hub?.input.value.trim() || ""), SEARCH_MS);

  dialog.addEventListener("close", () => {
    state.session += 1;
    stopPolling();
    search.cancel();
    hubController?.abort();
    hubController = null;
    state.expandToken = null;
  });

  return { open, refreshIfOpen };

  /* ---------- Open, load, mode ------------------------------------------- */

  function open() {
    if (dialog.open) {
      refreshIfOpen();
      return;
    }
    state.session += 1;
    state.library = null;
    state.jobs = [];
    state.installer = null;
    state.slots = null;
    state.rows = new Map();
    state.confirmId = "";
    state.expandToken = null;
    filesCache = new Map();
    mode = "";
    hub = null;
    openSheet(dialog);
    start(state.session);
  }

  function refreshIfOpen() {
    if (dialog.open && state.library) reload().catch(() => {});
  }

  async function start(session) {
    showLoading();
    try {
      const library = await api.library();
      if (session === state.session) applyLibrary(library);
    } catch (error) {
      if (session === state.session) showLoadError(error);
    }
  }

  async function reload() {
    const session = state.session;
    const library = await api.library();
    if (session === state.session) applyLibrary(library);
  }

  function applyLibrary(library) {
    state.library = library;
    state.jobs = library.downloads || [];
    state.installer = library.installer || null;
    for (const job of state.jobs) if (job.state === "done") markInstalled(job);
    if (library.mode === "attached") {
      mode = "attached";
      state.slots = null;
      renderAttached();
    } else {
      if (mode !== "local") buildLocal();
      renderAll();
    }
    refreshHubActions();
    ensurePolling();
  }

  function showLoading() {
    body.replaceChildren(
      el("div", { class: "mdl-loading" }, [el("span", { class: "spinner", role: "status", "aria-label": "Loading models" })]),
    );
  }

  function showLoadError(error) {
    body.replaceChildren(
      el("div", { class: "mdl-loading-error" }, [
        notice(errorText(error)),
        el("div", { class: "mdl-actions" }, [
          actionButton("Try again", "btn btn-primary", () => start(state.session)),
        ]),
      ]),
    );
  }

  function buildLocal() {
    mode = "local";
    const slots = {
      setup: block("setup"),
      library: block("library"),
      downloads: block("downloads"),
      hub: block("hub"),
      engine: block("engine"),
    };
    state.slots = slots;
    slots.hub.replaceChildren(hubSection());
    body.replaceChildren(slots.setup, slots.library, slots.downloads, slots.hub, slots.engine);
    runSearch("");
  }

  function block(name) {
    return el("div", { class: "mdl-block", dataset: { slot: name } });
  }

  function renderAll() {
    renderSetup();
    renderLibrary();
    renderJobs();
    renderEngine();
  }

  function renderAttached() {
    const server = getState().settings?.server_url || "";
    body.replaceChildren(
      el("section", { class: "panel-section" }, [
        el("div", { class: "panel-title", text: "Kiwibro is using a model server" }),
        server ? el("div", { class: "mdl-address", text: server }) : null,
        el("p", { class: "mdl-text", text: "Models are added and removed on that server." }),
        el("div", { class: "mdl-actions" }, [
          actionButton("Open Settings", "btn btn-ghost", () => actions.openSettings()),
        ]),
      ]),
    );
  }

  /* ---------- Shared small parts ----------------------------------------- */

  function currentModel() {
    return getState().currentModel || "";
  }

  function isStreaming() {
    return Boolean(getState().streaming);
  }

  function ramBytes() {
    return state.library?.ram_bytes || 0;
  }

  function notice(text) {
    return el("div", { class: "panel-error", role: "alert" }, [icon("alert"), el("span", { text })]);
  }

  function panelEmpty(text) {
    return el("div", { class: "panel-empty", text });
  }

  function spinnerLine(label) {
    return el("div", { class: "mdl-state" }, [el("span", { class: "spinner", role: "status", "aria-label": label })]);
  }

  function actionButton(label, className, onPress, extra = {}) {
    const node = el("button", { class: className, type: "button", ...extra }, label);
    node.addEventListener("click", () => onPress(node));
    return node;
  }

  function fitLine(fit, sizeBytes) {
    if (!(ramBytes() > 0)) return null;
    const percent = Math.min(100, (((sizeBytes || 0) * 1.25 + GIB) / ramBytes()) * 100);
    const tone = fit === "tight" ? "tan" : fit === "too_big" ? "warn" : undefined;
    return el("div", { class: "mdl-fit", dataset: { fit: fit || "unknown" } }, [
      el("span", { class: "bar", "data-tone": tone }, [el("span", { style: { width: `${percent}%` } })]),
      el("span", { text: FIT_TEXT[fit] || "" }),
    ]);
  }

  /**
   * Runs one request started from a button. The button stays disabled until the
   * request settles; a failure shows a toast. Returns true when the task finished.
   */
  async function attempt(button, task) {
    if (button) button.disabled = true;
    try {
      await task();
      return true;
    } catch (error) {
      if (!isAbort(error)) toast(errorText(error), { tone: "error", iconName: "alert" });
      return false;
    } finally {
      if (button) button.disabled = false;
    }
  }

  function notify() {
    Promise.resolve(actions.modelsChanged?.()).catch(() => {});
  }

  /* ---------- Engine setup (A) and engine (E) ----------------------------- */

  function renderSetup() {
    if (!state.slots) return;
    const nodes = state.library.runtime ? [] : [setupSection()];
    state.slots.setup.replaceChildren(...nodes);
  }

  function setupSection() {
    const installer = state.installer || {};
    const children = [el("div", { class: "panel-title", text: "Kiwibro needs its engine" })];
    if (installer.supported === false) {
      children.push(
        notice(
          `There is no ready-made engine for this computer (${installer.arch || "unknown"}). Install llama.cpp yourself and Kiwibro will find it.`,
        ),
      );
      return el("section", { class: "panel-section" }, children);
    }
    children.push(
      el("p", {
        class: "mdl-text",
        text: "The engine is llama.cpp, the program that runs models. Kiwibro downloads the official build from GitHub and keeps it in its own folder.",
      }),
    );
    if (isBusy(installer)) {
      children.push(progressRow(installer));
    } else {
      if (installer.state === "error") children.push(notice(installer.error || "The install did not finish."));
      const label = installer.state === "error" ? "Try again" : "Install engine";
      children.push(el("div", { class: "mdl-actions" }, [actionButton(label, "btn btn-primary", installEngine)]));
    }
    return el("section", { class: "panel-section" }, children);
  }

  function renderEngine() {
    if (!state.slots) return;
    const runtime = state.library.runtime;
    state.slots.engine.replaceChildren(...(runtime ? [engineSection(runtime)] : []));
  }

  function engineSection(runtime) {
    const installer = state.installer || {};
    const line = el("div", { class: "mdl-engine-line" }, [el("span", { text: runtime.label || "llama.cpp" })]);
    if (installer.supported) {
      line.append(actionButton("Check for a newer build", "text-btn", installEngine, { disabled: isBusy(installer) }));
    }
    const children = [el("div", { class: "panel-title", text: "Engine" }), line];
    if (isBusy(installer)) children.push(progressRow(installer));
    else if (installer.state === "error") children.push(notice(installer.error || "The check did not finish."));
    const builds = installer.installed || [];
    if (builds.length > 1) children.push(el("div", { class: "mdl-list" }, builds.map(buildRow)));
    return el("section", { class: "panel-section" }, children);
  }

  function buildRow(build) {
    const end = build.current
      ? el("span", { class: "mdl-inuse" }, [icon("check"), "In use"])
      : [
          actionButton("Use", "text-btn", (button) => useBuild(build, button)),
          actionButton("Delete", "text-btn danger", (button) => deleteBuild(build, button)),
        ];
    return el("div", { class: "mdl-row mdl-build" }, [
      el("div", { class: "mdl-main" }, [
        el("div", { class: "mdl-name", text: `${build.tag} · ${BUILD_KIND[build.kind] || build.kind}` }),
        el("div", { class: "mdl-meta", text: formatBytes(build.size_bytes) }),
      ]),
      el("div", { class: "mdl-end" }, end),
    ]);
  }

  async function installEngine(button) {
    await attempt(button, async () => {
      const result = await api.installRuntime("auto");
      setInstaller(result?.installer || (await api.runtime()).installer);
    });
  }

  async function cancelInstall(button) {
    await attempt(button, async () => {
      await api.cancelRuntimeInstall();
      setInstaller((await api.runtime()).installer);
    });
  }

  async function useBuild(build, button) {
    await attempt(button, async () => {
      await api.useRuntime(build.name);
      await reload();
      notify();
    });
  }

  async function deleteBuild(build, button) {
    await attempt(button, async () => {
      await api.deleteRuntime(build.name);
      await reload();
      notify();
    });
  }

  function progressRow(installer) {
    const node = el("div", { class: "mdl-progress" }, [
      el("div", { class: "mdl-progress-top" }, [
        el("span", { class: "mdl-progress-label" }),
        actionButton("Cancel", "btn btn-ghost", cancelInstall),
      ]),
      el("div", { class: "bar" }, [el("span")]),
    ]);
    updateProgress(node, installer);
    return node;
  }

  function updateProgress(node, installer) {
    node.querySelector(".mdl-progress-label").textContent = installText(installer);
    node.querySelector(".bar > span").style.width = `${installPercent(installer)}%`;
  }

  /**
   * Stores the installer state. Progress rows move in place; the setup and
   * engine blocks are rebuilt only when the kind of state changes.
   */
  function setInstaller(installer) {
    if (!installer) return;
    const before = installerShape(state.installer);
    state.installer = installer;
    if (state.slots && installerShape(installer) !== before) {
      renderSetup();
      renderEngine();
    } else {
      for (const node of body.querySelectorAll(".mdl-progress")) updateProgress(node, installer);
    }
    ensurePolling();
  }

  /* ---------- Models on this computer (B) --------------------------------- */

  function renderLibrary() {
    if (!state.slots) return;
    const library = state.library;
    const models = library.models || [];
    const streaming = isStreaming();
    const total = models.reduce((sum, model) => sum + (model.size_bytes || 0), 0);
    const anyLoaded = models.some((model) => model.loaded);
    state.rows = new Map();
    state.confirmId = "";

    const children = [
      el("div", { class: "panel-title" }, [
        "On this computer",
        el("small", { text: `${countOf(models.length, "model")} · ${formatBytes(total)}` }),
        anyLoaded
          ? actionButton("Unload all", "text-btn mdl-title-end", unloadAll, {
              disabled: streaming,
              title: streaming ? WAIT_FOR_REPLY : undefined,
            })
          : null,
      ]),
      storageBlock(library),
      models.length
        ? el("div", { class: "mdl-list" }, sortModels(models).map(modelRow))
        : panelEmpty(library.runtime ? "No models yet. Pick one below." : "No models yet."),
    ];
    state.slots.library.replaceChildren(el("section", { class: "panel-section" }, children));
  }

  function storageBlock(library) {
    const free = library.disk?.free_bytes || 0;
    const sources = (library.sources || []).filter((source) => source.bytes > 0);
    const used = sources.reduce((sum, source) => sum + source.bytes, 0);
    const total = used + free;
    if (total <= 0) return null;
    return el("div", { class: "mdl-storage" }, [
      el(
        "div",
        { class: "mdl-disk", "aria-hidden": "true" },
        sources.map((source) =>
          el("span", {
            class: "mdl-seg",
            dataset: { source: sourceKind(source.source) },
            style: { width: `${(source.bytes / total) * 100}%` },
          }),
        ),
      ),
      el("div", { class: "mdl-legend" }, [
        ...sources.map((source) =>
          el("span", { class: "mdl-legend-item", title: source.path || undefined }, [
            el("span", { class: "mdl-swatch", dataset: { source: sourceKind(source.source) } }),
            `${source.label || source.source} ${formatBytes(source.bytes)}`,
          ]),
        ),
        el("span", { class: "mdl-free", text: `${formatBytes(free)} free` }),
      ]),
    ]);
  }

  function sortModels(models) {
    const current = currentModel();
    const rank = (model) => (model.id === current ? 0 : model.loaded ? 1 : 2);
    return [...models].sort((a, b) => rank(a) - rank(b) || String(a.name).localeCompare(String(b.name)));
  }

  function metaParts(model) {
    const facts = [model.params, model.quantization, model.size_bytes ? formatBytes(model.size_bytes) : ""];
    const parts = [];
    const text = facts.filter(Boolean).join(" · ");
    if (text) parts.push(text);
    if (model.source_label) {
      if (parts.length) parts.push(" · ");
      if (!model.deletable) parts.push(lockMark(model));
      parts.push(model.source_label);
    }
    return parts;
  }

  function lockMark(model) {
    const title = model.source === "lmstudio" ? "Remove it in LM Studio" : "In a folder Kiwibro does not manage";
    return el("span", { class: "mdl-lock", title, role: "img", "aria-label": title }, [icon("lock")]);
  }

  function modelRow(model) {
    const row = el("div", { class: "mdl-row", dataset: { model: model.id } }, [
      el("div", { class: "mdl-main" }, [
        el("div", { class: "mdl-name", text: model.name }),
        el("div", { class: "mdl-meta" }, metaParts(model)),
        fitLine(model.fit, model.size_bytes),
      ]),
      el("div", { class: "mdl-end" }, rowEnd(model)),
    ]);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.confirmId === model.id) {
        event.preventDefault();
        event.stopPropagation();
        keepModel(model.id);
      }
    });
    state.rows.set(model.id, row);
    return row;
  }

  function rowEnd(model) {
    const streaming = isStreaming();
    const nodes = [];
    if (model.id === currentModel()) {
      nodes.push(el("span", { class: "mdl-inuse" }, [icon("check"), "In use"]));
    } else {
      if (model.loaded) nodes.push(el("span", { class: "mdl-muted", text: "Loaded" }));
      nodes.push(
        actionButton("Use", "btn btn-ghost", (button) => useModel(model, button), {
          disabled: streaming,
          title: streaming ? WAIT_FOR_REPLY : undefined,
        }),
      );
    }
    if (model.deletable) {
      nodes.push(
        actionButton(icon("trash"), "icon-btn", () => askDelete(model), {
          title: "Delete",
          "aria-label": `Delete ${model.name}`,
          disabled: streaming,
        }),
      );
    }
    return nodes;
  }

  function askDelete(model) {
    if (state.confirmId && state.confirmId !== model.id) keepModel(state.confirmId, { focus: false });
    state.confirmId = model.id;
    const row = state.rows.get(model.id);
    if (!row) return;
    const end = row.querySelector(".mdl-end");
    end.replaceChildren(
      el("span", { class: "mdl-confirm-text", text: `Delete? Frees ${formatBytes(model.size_bytes)}` }),
      actionButton("Delete", "btn btn-danger", (button) => deleteModel(model, button)),
      actionButton("Keep", "btn btn-ghost mdl-keep", () => keepModel(model.id)),
    );
    end.querySelector(".mdl-keep").focus();
  }

  function keepModel(id, { focus = true } = {}) {
    if (state.confirmId === id) state.confirmId = "";
    const model = (state.library?.models || []).find((item) => item.id === id);
    const row = state.rows.get(id);
    if (!model || !row) return;
    const end = row.querySelector(".mdl-end");
    end.replaceChildren(...rowEnd(model));
    if (focus) end.querySelector(".icon-btn")?.focus();
  }

  async function deleteModel(model, button) {
    const removed = await attempt(button, async () => {
      const result = await api.deleteModel(model.id);
      filesCache.clear();
      toast(`Deleted ${model.name} · freed ${formatBytes(result?.freed_bytes ?? model.size_bytes)}`);
      await reload();
      notify();
    });
    if (!removed) keepModel(model.id, { focus: false });
  }

  async function useModel(model, button) {
    await attempt(button, async () => {
      await actions.selectModel(model.id);
      await reload();
    });
  }

  async function unloadAll(button) {
    await attempt(button, async () => {
      await api.engineStop();
      await reload();
      notify();
    });
  }

  /* ---------- Downloads (C) ----------------------------------------------- */

  function renderJobs() {
    if (!state.slots) return;
    const jobs = state.jobs;
    if (!jobs.length) {
      state.slots.downloads.replaceChildren();
      return;
    }
    const finished = jobs.some((job) => FINISHED_JOB.includes(job.state));
    state.slots.downloads.replaceChildren(
      el("section", { class: "panel-section" }, [
        el("div", { class: "panel-title" }, [
          "Downloads",
          finished ? actionButton("Clear finished", "text-btn mdl-title-end", clearFinished) : null,
        ]),
        el("div", { class: "mdl-list" }, jobs.map(jobRow)),
      ]),
    );
  }

  function jobRow(job) {
    const children = [
      el("div", { class: "mdl-name", text: job.name || job.file }),
      el("div", { class: "mdl-job-state", text: jobStateText(job), "data-tone": jobTone(job) }),
    ];
    if (job.state === "downloading") {
      children.push(el("div", { class: "bar" }, [el("span", { style: { width: `${jobPercent(job)}%` } })]));
    }
    return el("div", { class: "mdl-row mdl-job", dataset: { job: job.id } }, [
      el("div", { class: "mdl-main" }, children),
      el("div", { class: "mdl-end" }, jobActions(job)),
    ]);
  }

  function jobStateText(job) {
    switch (job.state) {
      case "queued":
        return "Waiting…";
      case "downloading": {
        const parts = [`${formatBytes(job.received)} of ${formatBytes(job.total)}`];
        if (job.speed_bps > 0) parts.push(`${formatBytes(job.speed_bps)}/s`);
        const eta = formatEta(job.eta_s);
        if (eta) parts.push(`${eta} left`);
        return parts.join(" · ");
      }
      case "done":
        return "Ready";
      case "cancelled":
        return "Cancelled";
      case "error":
        return job.error || "The download failed.";
      default:
        return job.state || "";
    }
  }

  function jobTone(job) {
    if (job.state === "done") return "ok";
    if (job.state === "error") return "error";
    return undefined;
  }

  function jobActions(job) {
    if (ACTIVE_JOB.includes(job.state)) {
      return [actionButton("Cancel", "btn btn-ghost", (button) => cancelJob(job, button))];
    }
    if (job.state === "done") {
      return [actionButton("Use", "btn btn-ghost", (button) => useFinished(job, button))];
    }
    if (job.state === "cancelled") {
      return [actionButton("Resume", "btn btn-ghost", (button) => startDownload(job.repo, job.file, button))];
    }
    if (job.state === "error") {
      return [actionButton("Try again", "btn btn-ghost", (button) => startDownload(job.repo, job.file, button))];
    }
    return [];
  }

  function updateJobRow(row, job) {
    row.querySelector(".mdl-job-state").textContent = jobStateText(job);
    const fill = row.querySelector(".bar > span");
    if (fill) fill.style.width = `${jobPercent(job)}%`;
  }

  /** Applies a new job list: rebuilds the section when a job changed state, otherwise moves the bars. */
  function setJobs(jobs) {
    if (!state.slots) return;
    const before = jobsKey(state.jobs);
    state.jobs = jobs;
    for (const job of jobs) if (job.state === "done") markInstalled(job);
    if (jobsKey(jobs) !== before) {
      renderJobs();
      refreshHubActions();
    } else {
      for (const row of state.slots.downloads.querySelectorAll("[data-job]")) {
        const job = jobs.find((item) => item.id === row.dataset.job);
        if (job) updateJobRow(row, job);
      }
    }
    ensurePolling();
  }

  async function refreshJobs() {
    const session = state.session;
    const result = await api.downloads();
    if (session === state.session) setJobs(result.jobs || []);
  }

  async function cancelJob(job, button) {
    await attempt(button, async () => {
      await api.cancelDownload(job.id);
      await refreshJobs();
    });
  }

  async function clearFinished(button) {
    await attempt(button, async () => {
      await api.clearDownloads();
      await refreshJobs();
    });
  }

  async function useFinished(job, button) {
    await attempt(button, async () => {
      await reload();
      const base = job.file.replace(/\.gguf$/i, "").toLowerCase();
      const model = (state.library?.models || []).find((item) => item.id.toLowerCase().endsWith(base));
      if (!model) {
        state.slots?.library.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
      await actions.selectModel(model.id);
      await reload();
    });
  }

  function markInstalled(job) {
    for (const file of filesCache.get(job.repo) || []) {
      if (file.file === job.file) file.installed = true;
    }
  }

  function jobActiveFor(repo, file) {
    return state.jobs.some((job) => job.repo === repo && job.file === file && ACTIVE_JOB.includes(job.state));
  }

  /* ---------- Get a model (D) --------------------------------------------- */

  function hubSection() {
    const input = el("input", {
      type: "search",
      placeholder: "Search models, e.g. qwen 4b",
      "aria-label": "Search Hugging Face",
      autocomplete: "off",
    });
    input.addEventListener("input", () => search());
    const results = el("div", { class: "mdl-hub-results" });
    hub = { input, results };
    return el("section", { class: "panel-section" }, [
      el("div", { class: "panel-title" }, ["Get a model", el("small", { text: "from Hugging Face" })]),
      el("div", { class: "search-field" }, [icon("search"), input]),
      results,
    ]);
  }

  async function runSearch(query) {
    if (!hub) return;
    hubController?.abort();
    const controller = new AbortController();
    hubController = controller;
    const session = state.session;
    const seq = ++hubSeq;
    const { results } = hub;
    results.replaceChildren(spinnerLine("Searching"));
    try {
      const data = await api.hubSearch(query, { signal: controller.signal });
      if (session !== state.session || seq !== hubSeq) return;
      const items = data.results || [];
      results.replaceChildren(
        items.length ? el("div", { class: "mdl-hub" }, items.map(hubItem)) : panelEmpty("Nothing found."),
      );
    } catch (error) {
      if (isAbort(error) || session !== state.session || seq !== hubSeq) return;
      results.replaceChildren(notice(errorText(error)));
    }
  }

  function hubMeta(item) {
    const parts = [`${formatCount(item.downloads)} downloads`];
    if (item.vision) parts.push("sees images");
    if (item.license) parts.push(item.license);
    return parts.join(" · ");
  }

  function hubItem(item) {
    fileListSeq += 1;
    const filesId = `mdl-files-${fileListSeq}`;
    const files = el("div", { class: "mdl-files", id: filesId, hidden: true });
    const head = el(
      "button",
      { class: "mdl-hub-head", type: "button", "aria-expanded": "false", "aria-controls": filesId },
      [
        el("span", { class: "mdl-hub-name" }, [
          item.author ? el("span", { class: "mdl-muted", text: `${item.author}/` }) : null,
          item.name,
        ]),
        el("span", { class: "mdl-hub-meta", text: hubMeta(item) }),
        icon("chevron-down", "mdl-chev"),
      ],
    );
    const entry = el("div", { class: "mdl-hub-item" }, [head, files]);
    head.addEventListener("click", () => toggleRepo(entry, item.repo));
    return entry;
  }

  function setExpanded(entry, open) {
    entry.querySelector(".mdl-hub-head").setAttribute("aria-expanded", String(open));
    const files = entry.querySelector(".mdl-files");
    files.hidden = !open;
    if (!open) files.replaceChildren();
  }

  function toggleRepo(entry, repo) {
    const wasOpen = entry.querySelector(".mdl-hub-head").getAttribute("aria-expanded") === "true";
    for (const other of hub.results.querySelectorAll(".mdl-hub-item")) setExpanded(other, false);
    state.expandToken = null;
    if (!wasOpen) expandRepo(entry, repo);
  }

  async function expandRepo(entry, repo) {
    setExpanded(entry, true);
    const token = {};
    state.expandToken = token;
    const session = state.session;
    const files = entry.querySelector(".mdl-files");
    files.replaceChildren(spinnerLine("Loading files"));
    try {
      const list = await loadFiles(repo);
      if (session !== state.session || token !== state.expandToken) return;
      files.replaceChildren(
        ...(list.length ? fileRows(repo, list) : [panelEmpty("No model files in this repository.")]),
      );
    } catch (error) {
      if (session !== state.session || token !== state.expandToken) return;
      files.replaceChildren(notice(errorText(error)));
    }
  }

  async function loadFiles(repo) {
    if (!filesCache.has(repo)) {
      const data = await api.hubFiles(repo);
      filesCache.set(repo, data.files || []);
    }
    return filesCache.get(repo);
  }

  function fileRows(repo, files) {
    const recommended = pickRecommended(files);
    return files.map((file) => fileRow(repo, file, file === recommended));
  }

  function fileRow(repo, file, recommended) {
    const top = [el("span", { class: "mdl-quant", text: file.quant || file.name || file.file })];
    if (file.size_bytes) top.push(el("span", { class: "mdl-muted", text: formatBytes(file.size_bytes) }));
    if (recommended) top.push(el("span", { class: "mdl-note", text: "Good balance of size and quality" }));
    if (file.projector) top.push(el("span", { class: "mdl-note", text: "Vision add-on" }));
    const row = el("div", { class: "mdl-file", dataset: { file: file.file } }, [
      el("div", { class: "mdl-file-main" }, [
        el("div", { class: "mdl-file-top" }, top),
        fitLine(file.fit, file.size_bytes),
      ]),
      el("div", { class: "mdl-end" }, fileAction(repo, file, recommended)),
    ]);
    fileInfo.set(row, { repo, file, recommended });
    return row;
  }

  function fileAction(repo, file, recommended) {
    if (file.installed) return [el("span", { class: "mdl-done" }, [icon("check"), "Installed"])];
    if (jobActiveFor(repo, file.file)) return [el("span", { class: "mdl-muted", text: "Downloading…" })];
    if (file.split) return [el("span", { class: "mdl-muted", text: "Split file" })];
    const primary = recommended && file.fit !== "too_big";
    return [
      actionButton("Download", primary ? "btn btn-primary" : "btn btn-ghost", (button) =>
        startDownload(repo, file.file, button),
      ),
    ];
  }

  function refreshHubActions() {
    if (!hub) return;
    for (const row of hub.results.querySelectorAll(".mdl-file")) {
      const info = fileInfo.get(row);
      if (info) row.querySelector(".mdl-end").replaceChildren(...fileAction(info.repo, info.file, info.recommended));
    }
  }

  async function startDownload(repo, file, button) {
    await attempt(button, async () => {
      await api.startDownload(repo, file);
      toast("Download started");
      await refreshJobs();
    });
  }

  /* ---------- Polling ----------------------------------------------------- */

  function wantsPolling() {
    if (!dialog.open || !state.library) return false;
    return state.jobs.some((job) => ACTIVE_JOB.includes(job.state)) || isBusy(state.installer);
  }

  function ensurePolling() {
    if (wantsPolling()) {
      if (!pollTimer) pollTimer = window.setInterval(tick, POLL_MS);
    } else {
      stopPolling();
    }
  }

  function stopPolling() {
    window.clearInterval(pollTimer);
    pollTimer = 0;
  }

  async function tick() {
    if (ticking || !state.library) return;
    ticking = true;
    const session = state.session;
    try {
      const result = await api.downloads();
      if (session !== state.session) return;
      const previous = new Map(state.jobs.map((job) => [job.id, job.state]));
      const jobDone = (result.jobs || []).some(
        (job) => job.state === "done" && ACTIVE_JOB.includes(previous.get(job.id)),
      );
      let installDone = false;
      if (isBusy(state.installer)) {
        const runtime = await api.runtime();
        if (session !== state.session) return;
        installDone = runtime.installer?.state === "done";
        if (!installDone) setInstaller(runtime.installer);
      }
      if (jobDone || installDone) {
        await reload();
        notify();
      } else {
        setJobs(result.jobs || []);
      }
    } catch {
      // A failed poll is simply tried again on the next tick.
    } finally {
      ticking = false;
    }
  }
}
