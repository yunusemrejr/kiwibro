/** Fake model library, Hugging Face catalogue, downloads and engine builds for ?mock=1. */

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;
const RAM_BYTES = 16 * GIB;
const FREE_BYTES = 96 * GIB;
const ENGINE_BYTES = 84 * MIB;
const BUILD_BYTES = 86_000_000;
const ACTIVE_JOB = ["queued", "downloading"];
const BUSY_INSTALL = ["checking", "downloading", "verifying", "unpacking"];
// Milliseconds from the start of an action at which each step begins.
const INSTALL_AT = { download: 800, verify: 4800, unpack: 5500, done: 6500 };
const JOB_QUEUE_MS = 500;
const JOB_RUN_MS = 5500;

// Models the mock starts with: where each one lives and how big it is.
const SEED_MODELS = {
  "qwen3.8-2b": { source: "kiwibro", size: 1.4 * GIB },
  "spark-x2.6-4b": { source: "kiwibro", size: 2.6 * GIB },
  "gemma-3-12b": { source: "lmstudio", size: 7.3 * GIB },
  "llama-vision-11b": { source: "lmstudio", size: 4.1 * GIB },
};

// Files are [name, quantization, size in GiB, extra]. Repos are invented names on public model families.
const CATALOGUE = [
  {
    repo: "lumen-ai/Qwen3-4B-GGUF",
    author: "lumen-ai",
    name: "Qwen3-4B-GGUF",
    downloads: 182345,
    likes: 41,
    vision: false,
    license: "apache-2.0",
    files: [
      ["Qwen3-4B-Q2_K.gguf", "Q2_K", 1.67],
      ["Qwen3-4B-Q4_K_M.gguf", "Q4_K_M", 2.5],
      ["Qwen3-4B-Q5_K_M.gguf", "Q5_K_M", 2.89],
      ["Qwen3-4B-Q8_0.gguf", "Q8_0", 4.28],
    ],
  },
  {
    repo: "pebble-labs/Llama-3.2-3B-Instruct-GGUF",
    author: "pebble-labs",
    name: "Llama-3.2-3B-Instruct-GGUF",
    downloads: 96210,
    likes: 27,
    vision: false,
    license: "llama3.2",
    files: [
      ["Llama-3.2-3B-Instruct-Q4_K_M.gguf", "Q4_K_M", 2.02],
      ["Llama-3.2-3B-Instruct-Q6_K.gguf", "Q6_K", 2.74],
      ["Llama-3.2-3B-Instruct-Q8_0.gguf", "Q8_0", 3.42],
      ["Llama-3.2-3B-Instruct-F16.gguf", "F16", 6.43],
    ],
  },
  {
    repo: "orchard-ml/Gemma-3-12B-it-GGUF",
    author: "orchard-ml",
    name: "Gemma-3-12B-it-GGUF",
    downloads: 1250000,
    likes: 310,
    vision: true,
    license: "gemma",
    files: [
      ["gemma-3-12b-it-Q2_K.gguf", "Q2_K", 4.52],
      ["gemma-3-12b-it-Q3_K_M.gguf", "Q3_K_M", 5.61],
      ["gemma-3-12b-it-Q4_K_M.gguf", "Q4_K_M", 7.3],
      ["gemma-3-12b-it-Q8_0.gguf", "Q8_0", 12.5],
      ["mmproj-gemma-3-12b-it-F16.gguf", "F16", 0.82, { projector: true }],
    ],
  },
  {
    repo: "reed-models/Phi-4-mini-GGUF",
    author: "reed-models",
    name: "Phi-4-mini-GGUF",
    downloads: 40512,
    likes: 12,
    vision: false,
    license: "mit",
    files: [
      ["Phi-4-mini-Q4_K_M.gguf", "Q4_K_M", 2.49],
      ["Phi-4-mini-Q8_0.gguf", "Q8_0", 4.08],
    ],
  },
  {
    repo: "sable-lab/Mistral-Small-24B-GGUF",
    author: "sable-lab",
    name: "Mistral-Small-24B-GGUF",
    downloads: 58790,
    likes: 77,
    vision: false,
    license: "apache-2.0",
    files: [
      ["Mistral-Small-24B-Q2_K.gguf", "Q2_K", 8.9],
      ["Mistral-Small-24B-Q3_K_M.gguf", "Q3_K_M", 10.8],
      ["Mistral-Small-24B-Q4_K_M-00001-of-00002.gguf", "Q4_K_M", 10.0, { split: true }],
      ["Mistral-Small-24B-Q4_K_M-00002-of-00002.gguf", "Q4_K_M", 4.3, { split: true }],
    ],
  },
  {
    repo: "fernhill/SmolLM3-3B-GGUF",
    author: "fernhill",
    name: "SmolLM3-3B-GGUF",
    downloads: 13904,
    likes: 9,
    vision: false,
    license: "apache-2.0",
    files: [
      ["SmolLM3-3B-Q4_K_M.gguf", "Q4_K_M", 1.92],
      ["SmolLM3-3B-Q6_K.gguf", "Q6_K", 2.5],
      ["SmolLM3-3B-Q8_0.gguf", "Q8_0", 3.1],
    ],
  },
];

const STORES = new WeakMap();

/* ---------- The fake state, per backend ------------------------------------ */

function storeFor(backend) {
  let store = STORES.get(backend);
  if (!store) {
    const noEngine = new URLSearchParams(window.location.search).get("engine") === "0";
    store = {
      meta: new Map(Object.entries(SEED_MODELS)),
      builds: noEngine ? [] : [makeBuild("b6123", true)],
      installer: { state: "idle", startedAt: 0, received: 0, total: 0, error: "", tag: "" },
      jobs: [],
      installedFiles: new Map(), // "repo|file" -> model id
      nextId: 1,
    };
    STORES.set(backend, store);
  }
  return store;
}

function makeBuild(tag, current = false) {
  return {
    name: `${tag}-vulkan`,
    tag,
    kind: "vulkan",
    path: `~/.local/share/kiwibro/runtime/${tag}-vulkan/llama-server`,
    size_bytes: BUILD_BYTES,
    installed_at: 1794000000.0,
    current,
  };
}

function activeBuild(store) {
  return store.builds.find((build) => build.current) || null;
}

function runtimeView(build) {
  if (!build) return null;
  return {
    path: build.path,
    label: `llama.cpp ${build.tag} (Vulkan)`,
    kind: build.kind,
    source: "kiwibro",
    version: build.tag,
    own: true,
  };
}

function fitOf(sizeBytes) {
  const need = sizeBytes * 1.25 + GIB;
  if (need <= RAM_BYTES * 0.6) return "good";
  if (need <= RAM_BYTES * 0.85) return "tight";
  return "too_big";
}

function paramsOf(name) {
  const match = /(\d+(?:\.\d+)?)B(?![a-z])/i.exec(name);
  return match ? `${match[1]}B` : "";
}

/* ---------- Time-driven progress ------------------------------------------- */
// Nothing runs on timers: every request first moves the fake jobs to where the clock says they should be.

function advance(store, backend) {
  const now = Date.now();
  const install = store.installer;
  if (BUSY_INSTALL.includes(install.state)) {
    const elapsed = now - install.startedAt;
    if (elapsed < INSTALL_AT.download) {
      install.state = "checking";
    } else if (elapsed < INSTALL_AT.verify) {
      install.state = "downloading";
      install.received = Math.round((install.total * (elapsed - INSTALL_AT.download)) / (INSTALL_AT.verify - INSTALL_AT.download));
    } else if (elapsed < INSTALL_AT.unpack) {
      install.state = "verifying";
      install.received = install.total;
    } else if (elapsed < INSTALL_AT.done) {
      install.state = "unpacking";
    } else {
      install.state = "done";
      finishInstall(store);
    }
  }

  for (const job of store.jobs) {
    if (!ACTIVE_JOB.includes(job.state)) continue;
    const elapsed = now - job.startedAt;
    if (elapsed < JOB_QUEUE_MS) {
      job.state = "queued";
      continue;
    }
    const share = Math.min(1, (elapsed - JOB_QUEUE_MS) / JOB_RUN_MS);
    job.received = Math.round(job.total * share);
    if (share >= 1) {
      job.state = "done";
      job.received = job.total;
      job.speed_bps = 0;
      job.eta_s = 0;
      job.path = `~/.local/share/kiwibro/models/${job.file}`;
      job.finished_at = now / 1000;
      publish(store, backend, job);
    } else {
      job.state = "downloading";
      job.speed_bps = Math.round(job.total / (JOB_RUN_MS / 1000));
      job.eta_s = Math.round((job.total - job.received) / job.speed_bps);
    }
  }
}

function finishInstall(store) {
  const tag = store.builds.length ? "b6140" : "b6123";
  let build = store.builds.find((item) => item.tag === tag);
  if (!build) {
    build = makeBuild(tag);
    store.builds.push(build);
  }
  for (const item of store.builds) item.current = item === build;
  store.installer.tag = tag;
}

// A finished download becomes one model in the library, once.
function publish(store, backend, job) {
  const stem = job.file.replace(/\.gguf$/i, "");
  const id = `kiwibro/${stem}`;
  store.installedFiles.set(`${job.repo}|${job.file}`, id);
  store.meta.set(id, { source: "kiwibro", size: job.total });
  if (!backend.models.some((model) => model.id === id)) {
    backend.models.push({
      id,
      name: stem,
      type: "llm",
      state: "not-loaded",
      context: 32768,
      max_context: 131072,
      tools: false,
      vision: false,
      capabilities: [],
      publisher: job.repo.split("/")[0],
      quantization: job.quant,
    });
  }
}

/* ---------- Views ---------------------------------------------------------- */

function modelView(model, meta, running) {
  return {
    id: model.id,
    name: model.name,
    publisher: model.publisher || "",
    params: paramsOf(model.name),
    quantization: model.quantization || "",
    size_bytes: meta.size,
    context_length: model.context || model.max_context || 0,
    kind: model.type === "vlm" ? "vlm" : "llm",
    source: meta.source,
    source_label: meta.source === "kiwibro" ? "Kiwibro" : "LM Studio",
    deletable: meta.source === "kiwibro",
    loaded: running && model.state === "loaded",
    fit: fitOf(meta.size),
  };
}

function sourceView(source, label, path, models) {
  const mine = models.filter((model) => model.source === source);
  return { source, label, path, models: mine.length, bytes: mine.reduce((sum, model) => sum + model.size_bytes, 0) };
}

function buildView(build) {
  return { ...build };
}

function installerView(store) {
  const install = store.installer;
  return {
    state: install.state,
    supported: true,
    arch: "x64",
    tag: install.tag,
    kind: install.state === "idle" ? "" : "vulkan",
    received: install.received,
    total: install.total,
    error: install.error,
    installed: store.builds.map(buildView),
    current: activeBuild(store) ? buildView(activeBuild(store)) : null,
  };
}

function jobView(job) {
  return {
    id: job.id,
    repo: job.repo,
    file: job.file,
    name: job.file,
    state: job.state,
    received: job.received,
    total: job.total,
    speed_bps: job.speed_bps,
    eta_s: job.eta_s,
    error: job.error,
    path: job.path,
    started_at: job.startedAt / 1000,
    finished_at: job.finished_at || 0,
  };
}

function libraryView(backend, store) {
  const models = backend.models.map((model) => {
    const meta = store.meta.get(model.id) || { source: "lmstudio", size: 4 * GIB };
    return modelView(model, meta, Boolean(activeBuild(store)));
  });
  const modelBytes = models.reduce((sum, model) => sum + model.size_bytes, 0);
  const partial = store.jobs
    .filter((job) => job.state === "downloading")
    .reduce((sum, job) => sum + job.received, 0);
  return {
    mode: "local",
    models,
    sources: [
      sourceView("kiwibro", "Kiwibro", "~/.local/share/kiwibro/models", models),
      sourceView("lmstudio", "LM Studio", "~/.lmstudio/models", models),
    ],
    disk: { free_bytes: FREE_BYTES, models_bytes: modelBytes, partial_bytes: partial },
    ram_bytes: RAM_BYTES,
    runtime: runtimeView(activeBuild(store)),
    installer: installerView(store),
    downloads: store.jobs.map(jobView),
  };
}

function fileView(entry, repo, store) {
  const [file, quant, gib, extra = {}] = entry;
  const size = Math.round(gib * GIB);
  return {
    file,
    name: file,
    size_bytes: size,
    sha256: "",
    quant,
    projector: Boolean(extra.projector),
    split: Boolean(extra.split),
    installed: store.installedFiles.has(`${repo}|${file}`),
    fit: fitOf(size),
  };
}

function findEntry(repo, file) {
  const repoEntry = CATALOGUE.find((item) => item.repo === repo);
  const fileEntry = repoEntry?.files.find((item) => item[0] === file);
  return repoEntry && fileEntry ? { repoEntry, fileEntry } : null;
}

/* ---------- Actions -------------------------------------------------------- */

function startInstall(store) {
  if (BUSY_INSTALL.includes(store.installer.state)) return;
  store.installer = { state: "checking", startedAt: Date.now(), received: 0, total: ENGINE_BYTES, error: "", tag: "" };
}

function cancelInstall(store) {
  if (BUSY_INSTALL.includes(store.installer.state)) store.installer = { ...store.installer, state: "cancelled" };
}

function startJob(store, body, fail) {
  const found = findEntry(body?.repo, body?.file);
  if (!found) fail(404, "not_found", "That file is not in the catalogue.");
  const [, quant, gib] = found.fileEntry;
  const id = `dl_${(store.nextId++).toString(16).padStart(8, "0")}`;
  const job = {
    id,
    repo: body.repo,
    file: body.file,
    quant,
    state: "queued",
    received: 0,
    total: Math.round(gib * GIB),
    speed_bps: 0,
    eta_s: 0,
    error: "",
    path: "",
    startedAt: Date.now(),
  };
  store.jobs.push(job);
  return { job: jobView(job) };
}

function deleteModel(backend, store, id, fail) {
  const model = backend.models.find((item) => item.id === id);
  if (!model) fail(404, "not_found", "That model is not in the library.");
  const meta = store.meta.get(id) || { source: "lmstudio", size: 0 };
  if (meta.source !== "kiwibro") {
    const message =
      meta.source === "lmstudio"
        ? `${model.name} is in LM Studio's folder. Remove it in LM Studio.`
        : `${model.name} is in a folder Kiwibro does not manage.`;
    fail(409, "not_deletable", message);
  }
  backend.models.splice(backend.models.indexOf(model), 1);
  store.meta.delete(id);
  for (const [key, value] of store.installedFiles) if (value === id) store.installedFiles.delete(key);
  if (backend.currentModel === id) backend.currentModel = backend.models[0]?.id ?? null;
  return { ok: true, freed_bytes: meta.size };
}

function searchHub(backend, url, fail) {
  if (backend.offline) fail(502, "offline", "Could not reach Hugging Face. Check the internet connection.");
  const tokens = (url.searchParams.get("q") || "").toLowerCase().split(/\s+/).filter(Boolean);
  const limit = Number(url.searchParams.get("limit")) || 20;
  const results = CATALOGUE.filter((entry) => {
    const text = `${entry.repo} ${entry.name}`.toLowerCase();
    return tokens.every((token) => text.includes(token));
  })
    .sort((a, b) => b.downloads - a.downloads)
    .slice(0, limit)
    .map((entry) => ({
      repo: entry.repo,
      author: entry.author,
      name: entry.name,
      downloads: entry.downloads,
      likes: entry.likes,
      updated: "2026-05-01T00:00:00Z",
      vision: entry.vision,
      license: entry.license,
    }));
  return { results };
}

function hubFiles(store, url, fail) {
  const repo = url.searchParams.get("repo") || "";
  const entry = CATALOGUE.find((item) => item.repo === repo);
  if (!entry) fail(404, "not_found", "That repository is not in the catalogue.");
  const files = entry.files
    .map((item) => fileView(item, repo, store))
    .sort((a, b) => Number(a.projector) - Number(b.projector) || a.size_bytes - b.size_bytes);
  return { repo, files };
}

/* ---------- Routes --------------------------------------------------------- */

export function mockModels(backend, { route, verb, body, url, fail }) {
  const store = storeFor(backend);
  advance(store, backend);

  if (route === "/api/library" && verb === "GET") return libraryView(backend, store);
  if (route === "/api/library/models" && verb === "DELETE") return deleteModel(backend, store, body?.id, fail);

  if (route === "/api/hub/search" && verb === "GET") return searchHub(backend, url, fail);
  if (route === "/api/hub/files" && verb === "GET") return hubFiles(store, url, fail);
  if (route === "/api/hub/downloads" && verb === "GET") {
    return { jobs: store.jobs.map(jobView), disk: { free_bytes: FREE_BYTES } };
  }
  if (route === "/api/hub/downloads" && verb === "POST") return startJob(store, body, fail);
  if (route === "/api/hub/downloads/clear" && verb === "POST") {
    store.jobs = store.jobs.filter((job) => ACTIVE_JOB.includes(job.state));
    return { ok: true };
  }
  const cancel = route.match(/^\/api\/hub\/downloads\/([^/]+)\/cancel$/);
  if (cancel && verb === "POST") {
    const job = store.jobs.find((item) => item.id === decodeURIComponent(cancel[1]));
    if (!job) fail(404, "not_found", "That download is gone.");
    if (ACTIVE_JOB.includes(job.state)) job.state = "cancelled";
    return { ok: true, job: jobView(job) };
  }

  if (route === "/api/runtime" && verb === "GET") {
    return { mode: "local", active: runtimeView(activeBuild(store)), installer: installerView(store) };
  }
  if (route === "/api/runtime/install" && verb === "POST") {
    startInstall(store);
    return { installer: installerView(store) };
  }
  if (route === "/api/runtime/cancel" && verb === "POST") {
    cancelInstall(store);
    return { installer: installerView(store) };
  }
  if (route === "/api/runtime/use" && verb === "POST") {
    const build = store.builds.find((item) => item.name === body?.name);
    if (!build) fail(404, "not_found", "That engine build is not installed.");
    for (const item of store.builds) item.current = item === build;
    return { ok: true, active: runtimeView(build) };
  }
  const remove = route.match(/^\/api\/runtime\/([^/]+)$/);
  if (remove && verb === "DELETE") {
    const build = store.builds.find((item) => item.name === decodeURIComponent(remove[1]));
    if (!build) fail(404, "not_found", "That engine build is not installed.");
    if (build.current && backend.models.some((model) => model.state === "loaded")) {
      fail(409, "busy", "Unload the running model first.");
    }
    store.builds = store.builds.filter((item) => item !== build);
    if (build.current && store.builds.length) store.builds[0].current = true;
    return { ok: true };
  }

  if (route === "/api/engine/stop" && verb === "POST") {
    for (const model of backend.models) model.state = "not-loaded";
    return { ok: true };
  }

  return undefined;
}
