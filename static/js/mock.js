/**
 * Development-only fake backend, activated with ?mock=1.
 * It mirrors docs/API.md (REST shapes and SSE events) so the whole UI can be
 * exercised without Flask or a real engine. It has no effect in normal mode.
 */

import { mockModels } from "./mock_models.js";
import { mockNotes } from "./mock_notes.js";

const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
const clone = (value) => JSON.parse(JSON.stringify(value));
const nowIso = () => new Date().toISOString();
const isoAgo = (ms) => new Date(Date.now() - ms).toISOString();

// Kiwi Mind: the page's ?mind= parameter picks the state; a download runs for about MIND_SIM_MS.
const MIND_BYTES = 486110000;
const MIND_PARTIAL = 201000000;
const MIND_SIM_MS = 6000;
const MIND_SKILLS = ["chat", "web", "code", "terminal", "computer", "tools", "write"];

class MockApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

const MINUTE = 60_000;
const HOUR = 3_600_000;

const MODELS = [
  {
    id: "qwen3.8-2b",
    name: "qwen3.8-2b",
    type: "llm",
    state: "loaded",
    context: 32768,
    max_context: 262144,
    tools: true,
    vision: false,
    capabilities: ["tool_use"],
    publisher: "empero-ai",
    quantization: "Q4_K_M",
  },
  {
    id: "spark-x2.6-4b",
    name: "spark-x2.6-4b",
    type: "llm",
    state: "loaded",
    context: 32768,
    max_context: 1048576,
    tools: true,
    vision: false,
    capabilities: ["tool_use"],
    publisher: "bartowski",
    quantization: "IQ2_M",
  },
  {
    id: "gemma-3-12b",
    name: "gemma-3-12b",
    type: "llm",
    state: "not-loaded",
    context: 32768,
    max_context: 131072,
    tools: false,
    vision: false,
    capabilities: [],
    publisher: "google",
    quantization: "Q4_0",
  },
  {
    id: "llama-vision-11b",
    name: "llama-vision-11b",
    type: "vlm",
    state: "not-loaded",
    context: 32768,
    max_context: 131072,
    tools: false,
    vision: true,
    capabilities: ["vision"],
    publisher: "meta",
    quantization: "Q4_K_M",
  },
];

const DEFAULT_PERMISSIONS = {
  web_search: "auto",
  fetch_page: "auto",
  multi_search: "auto",
  fetch_pages: "auto",
  web_research: "auto",
  browse_page: "auto",
  browser: "auto",
  read_file: "auto",
  read_files: "auto",
  list_files: "auto",
  glob_files: "auto",
  grep_files: "auto",
  file_outline: "auto",
  git: "auto",
  write_file: "ask",
  replace_in_file: "ask",
  apply_patch: "ask",
  file_manage: "ask",
  run_command: "ask",
  multi_command: "ask",
  terminal_session: "ask",
  run_python: "ask",
  run_javascript: "ask",
  system_info: "auto",
  open_path: "ask",
  clipboard: "ask",
  notify: "auto",
  screenshot: "ask",
  model_info: "auto",
  delegate: "auto",
  describe_image: "auto",
  study_file: "auto",
  task_board: "auto",
  recall: "auto",
  remember: "auto",
  calculate: "auto",
  analyze_table: "auto",
  doc_search: "auto",
  doc_read: "auto",
  doc_add: "auto",
  doc_list: "auto",
  convert_document: "auto",
  remote: "ask",
  workspace: "auto",
  undo_changes: "auto",
  component: "ask",
};

// The llama.cpp build the local engine would pick, and a server it already started.
const DIRECT_RUNTIME = {
  path: "/usr/local/bin/llama-server",
  label: "llama.cpp · Vulkan",
  kind: "system",
  source: "PATH",
};

const SEED_MEMORIES = [
  {
    id: "mem_mock_1",
    kind: "preference",
    content: "Prefers dark, low-noise interfaces with generous spacing.",
    pinned: true,
    confidence: 1,
    created_at: isoAgo(12 * 24 * HOUR),
    updated_at: isoAgo(2 * HOUR),
    last_used_at: isoAgo(2 * HOUR),
  },
  {
    id: "mem_mock_2",
    kind: "project",
    content: "Building Kiwibro, a local-first AI agent for Linux.",
    pinned: true,
    confidence: 1,
    created_at: isoAgo(9 * 24 * HOUR),
    updated_at: isoAgo(6 * HOUR),
    last_used_at: isoAgo(6 * HOUR),
  },
  {
    id: "mem_mock_3",
    kind: "fact",
    content: "Lives in Istanbul and works mostly in the evenings.",
    pinned: false,
    confidence: 0.9,
    created_at: isoAgo(8 * 24 * HOUR),
    updated_at: isoAgo(3 * 24 * HOUR),
    last_used_at: null,
  },
  {
    id: "mem_mock_4",
    kind: "fact",
    content: "Prefers small local models that fit in 8 GB of memory.",
    pinned: false,
    confidence: 1,
    created_at: isoAgo(7 * 24 * HOUR),
    updated_at: isoAgo(2 * 24 * HOUR),
    last_used_at: isoAgo(3 * HOUR),
  },
  {
    id: "mem_mock_5",
    kind: "preference",
    content: "Wants Python examples to stay dependency-free where possible.",
    pinned: false,
    confidence: 0.85,
    created_at: isoAgo(6 * 24 * HOUR),
    updated_at: isoAgo(5 * HOUR),
    last_used_at: null,
  },
  {
    id: "mem_mock_6",
    kind: "decision",
    content: "Chat width capped at 760 px and UI text stays at 14 px.",
    pinned: false,
    confidence: 1,
    created_at: isoAgo(5 * 24 * HOUR),
    updated_at: isoAgo(28 * HOUR),
    last_used_at: isoAgo(28 * HOUR),
  },
  {
    id: "mem_mock_7",
    kind: "project",
    content: "Learning Rust on weekends with the Exercism track.",
    pinned: false,
    confidence: 0.8,
    created_at: isoAgo(4 * 24 * HOUR),
    updated_at: isoAgo(4 * 24 * HOUR),
    last_used_at: null,
  },
  {
    id: "mem_mock_8",
    kind: "person",
    content: "Works with a team lead named Alex on the Kiwibro project.",
    pinned: false,
    confidence: 0.75,
    created_at: isoAgo(3 * 24 * HOUR),
    updated_at: isoAgo(3 * 24 * HOUR),
    last_used_at: null,
  },
  {
    id: "mem_mock_9",
    kind: "fact",
    content: "Workspace is ~/Kiwibro and notes live in notes.md.",
    pinned: false,
    confidence: 1,
    created_at: isoAgo(2 * 24 * HOUR),
    updated_at: isoAgo(20 * HOUR),
    last_used_at: isoAgo(20 * HOUR),
  },
  {
    id: "mem_mock_10",
    kind: "preference",
    content: "Prefers short answers with bullet points and code examples.",
    pinned: false,
    confidence: 0.95,
    created_at: isoAgo(26 * HOUR),
    updated_at: isoAgo(26 * HOUR),
    last_used_at: null,
  },
];

const WORKSPACE_FILES = [
  { path: "notes.md", type: "file", size: 1842, modified: isoAgo(3 * HOUR) },
  { path: "kiwi-notes.md", type: "file", size: 312, modified: isoAgo(40 * MINUTE) },
  { path: "src", type: "dir", size: 0, modified: isoAgo(2 * 24 * HOUR) },
  { path: "src/clean_csv.py", type: "file", size: 942, modified: isoAgo(27 * HOUR) },
  { path: "budgets", type: "dir", size: 0, modified: isoAgo(5 * 24 * HOUR) },
  { path: "budgets/2026.csv", type: "file", size: 12480, modified: isoAgo(5 * 24 * HOUR) },
  { path: "ideas", type: "dir", size: 0, modified: isoAgo(6 * 24 * HOUR) },
  { path: "ideas/kiwibro-next.md", type: "file", size: 2210, modified: isoAgo(6 * 24 * HOUR) },
  { path: "screenshots", type: "dir", size: 0, modified: isoAgo(4 * 24 * HOUR) },
  { path: "screenshots/empty-state.png", type: "file", size: 148211, modified: isoAgo(4 * 24 * HOUR) },
];

const SEED_NEWS = [
  { title: "Kiwi numbers rise on predator-free islands", url: "https://www.doc.govt.nz/kiwi", snippet: "DOC reports a steady increase in kiwi populations on predator-free islands." },
  { title: "Save the Kiwi: community trapping season", url: "https://savethekiwi.nz/", snippet: "Volunteer trappers removed more than 40,000 predators this season." },
  { title: "Kiwi conservation news", url: "https://en.wikipedia.org/wiki/Kiwi_(bird)", snippet: "Overview of kiwi species, habitat and conservation status." },
  { title: "Rowi eggs hatch at Willowbank", url: "https://www.willowbank.co.nz/", snippet: "Two rowi chicks hatched this week as part of the recovery programme." },
];

function chunkText(text, size) {
  const parts = [];
  let index = 0;
  while (index < text.length) {
    const jitter = Math.max(1, size + Math.floor(Math.random() * size));
    parts.push(text.slice(index, index + jitter));
    index += jitter;
  }
  return parts;
}

export class MockBackend {
  constructor() {
    const params = new URLSearchParams(window.location.search);
    this.offline = params.get("offline") === "1";
    this.models = this.offline ? [] : clone(MODELS);
    this.currentModel = this.offline ? null : "qwen3.8-2b";
    this.mindParam = params.get("mind") || "usable"; // Kiwi Mind state shown on the page
    this.mindDl = null; // a simulated download: { state, base, startedAt }
    this.mindFinished = false; // the simulated download has completed
    this.settings = {
      model: "qwen3.8-2b",
      server_url: "", // empty: local mode; an address switches to a llama.cpp-compatible server
      agent_enabled: true,
      memory_enabled: true,
      sense_enabled: true,
      english_only: true,
      max_output_tokens: 2048,
      temperature: 0.6,
      workspace: "~/Kiwibro",
      user_name: "",
      tool_profile: "auto",
      autonomy: "smart",
      sandbox: "auto",
      remote_hosts: [],
      max_tool_rounds: 16,
      permissions: clone(DEFAULT_PERMISSIONS),
      direct_runtime: "",
      direct_models_dir: "",
      direct_threads: 0,
      direct_gpu: "auto",
    };
    this.directServers = [
      {
        model: "qwen3.8-2b",
        pid: 48213,
        port: 8791,
        context_length: 8192,
        parallel: 1,
        threads: 8,
        runtime: DIRECT_RUNTIME.path,
        started_at: isoAgo(12 * MINUTE),
        busy: false,
      },
    ];
    this.memories = clone(SEED_MEMORIES);
    this.workspaceFiles = clone(WORKSPACE_FILES);
    this.dataDir = "~/.local/share/kiwibro (mock)";
    this.approvals = new Map();
    this.sequence = 0;
    this.messageSeq = 0;
    this.conversations = [];
    this._seedConversations();
  }

  /* ------------------------------------------------------------------ seed */

  _seedConversations() {
    const make = (id, title, model, ago, summary = "", compacted = false, messages = []) => ({
      id,
      title,
      model,
      pinned: false,
      created_at: isoAgo(ago + 2 * HOUR),
      updated_at: isoAgo(ago),
      summary,
      compacted,
      messages,
    });

    const conversationA = make("c_mock_kiwi", "Kiwi conservation notes", "qwen3.8-2b", 4 * MINUTE);
    this._seedMessage(conversationA, {
      role: "user",
      content: "What's the latest on kiwi conservation? Keep it short.",
      ago: 9 * MINUTE,
    });
    this._seedMessage(conversationA, {
      role: "assistant",
      reasoning: "A short summary with two or three concrete points is enough here.",
      ago: 8 * MINUTE,
      content:
        "Short version:\n\n- Predator-free islands are seeing **steady population growth**.\n- Community trapping removed tens of thousands of predators this season.\n- Rowi and Haast tokoeka remain the rarest species.\n\nWant the links?",
    });
    this._seedMessage(conversationA, {
      role: "user",
      content: "Save the main points to a file.",
      ago: 5 * MINUTE,
    });
    this._seedMessage(conversationA, {
      role: "assistant",
      ago: 4 * MINUTE,
      content: "",
      tool_calls: [
        {
          id: "call_seed_1",
          type: "function",
          function: {
            name: "write_file",
            arguments: JSON.stringify({
              path: "kiwi-notes.md",
              content: "# Kiwi conservation\n\n- Island populations growing\n- Community trapping working\n- Rowi still rarest\n",
            }),
          },
        },
      ],
    });
    this._seedMessage(conversationA, {
      role: "tool",
      content: "Wrote kiwi-notes.md (312 bytes)",
      tool_call_id: "call_seed_1",
      tool_name: "write_file",
      ago: 4 * MINUTE,
      meta: { ok: true, duration_ms: 96 },
    });
    this._seedMessage(conversationA, {
      role: "assistant",
      ago: 3 * MINUTE,
      content: "Saved it to `kiwi-notes.md`. Say the word if you want a version with sources.",
    });

    const conversationB = make("c_mock_csv", "CSV cleanup script", "qwen3.8-2b", HOUR);
    this._seedMessage(conversationB, {
      role: "user",
      content: "Write a small Python script that cleans up a CSV of expenses.",
      ago: HOUR + 30 * MINUTE,
    });
    this._seedMessage(conversationB, {
      role: "assistant",
      reasoning: "Keep it dependency-free with the csv module and show a short usage example.",
      ago: HOUR + 29 * MINUTE,
      content:
        "Here's a compact one that trims whitespace, drops empty rows, and normalises dates.\n\n```python\nimport csv\nfrom pathlib import Path\n\nsource = Path(\"expenses.csv\")\nwith source.open(newline=\"\") as fh:\n    rows = [row for row in csv.DictReader(fh) if any(row.values())]\n\nfor row in rows:\n    row[\"description\"] = row[\"description\"].strip()\n    row[\"amount\"] = f\"{float(row['amount']):.2f}\"\n\nwith source.with_name(\"expenses-clean.csv\").open(\"w\", newline=\"\") as fh:\n    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())\n    writer.writeheader()\n    writer.writerows(rows)\n```\n\nRun it with `python clean_csv.py`. It writes a sibling file, so the original is never touched.",
    });
    this._seedMessage(conversationB, {
      role: "user",
      content: "Nice — make it skip negative amounts too.",
      ago: HOUR + 2 * MINUTE,
    });
    this._seedMessage(conversationB, {
      role: "assistant",
      ago: HOUR,
      content:
        "Added a guard when filtering:\n\n```python\nrows = [\n    row for row in csv.DictReader(fh)\n    if any(row.values()) and float(row[\"amount\"] or 0) >= 0\n]\n```\n\nNegative refunds are now left out.",
    });

    const conversationC = make(
      "c_mock_ubuntu",
      "Ubuntu theme tweaks",
      "spark-x2.6-4b",
      26 * HOUR,
      "Earlier in this chat we settled on the Ubuntu font stack, 14 px body text, a 264 px sidebar and 12–14 px corner radii. The remaining tweaks were hover states and scrollbar width.",
      true,
    );
    this._seedMessage(conversationC, {
      role: "user",
      content: "What should I tweak first in my GNOME setup?",
      ago: 26 * HOUR + 12 * MINUTE,
    });
    this._seedMessage(conversationC, {
      role: "assistant",
      reasoning: "Keep the advice short and concrete.",
      ago: 26 * HOUR + 11 * MINUTE,
      content: "Three quick wins:\n\n1. Enable **fractional scaling** for a sharper UI.\n2. Set `org.gnome.mutter experimental-features` for VRR.\n3. Drop the window title bars where the header bar already exists.",
    });

    const conversationD = make("c_mock_trip", "Wellington weekend", "qwen3.8-2b", 3 * 24 * HOUR);
    this._seedMessage(conversationD, {
      role: "user",
      content: "Plan a calm two-day Wellington itinerary with good coffee.",
      ago: 3 * 24 * HOUR + 45 * MINUTE,
    });
    this._seedMessage(conversationD, {
      role: "assistant",
      reasoning: "Two days, low intensity, coffee stops.",
      ago: 3 * 24 * HOUR,
      content:
        "**Saturday** — Te Papa in the morning, lunch at the waterfront,Cuban coffee on Cuba Street, sunset at Mount Victoria.\n\n**Sunday** — Zealandia early, brunch in Aro Valley, slow walk through the Botanic Garden.",
    });

    const emptyConversation = make("c_mock_empty", "New chat", null, 0);
    this.conversations = [conversationA, conversationB, conversationC, conversationD, emptyConversation];
    this.conversations[1].pinned = true;
  }

  _seedMessage(conversation, partial) {
    const seq = (conversation.messages.at(-1)?.seq || 0) + 1;
    const message = {
      id: `m_mock_seed_${++this.messageSeq}`,
      seq,
      role: "assistant",
      content: "",
      reasoning: "",
      tool_calls: null,
      tool_name: null,
      tool_call_id: null,
      model: conversation.model,
      created_at: partial.ago ? isoAgo(partial.ago) : nowIso(),
      meta: {},
      ...partial,
    };
    conversation.messages.push(message);
    return message;
  }

  /* --------------------------------------------------------------- helpers */

  _conversation(id) {
    const conversation = this.conversations.find((entry) => entry.id === id);
    if (!conversation) throw new MockApiError(404, "not_found", "That conversation doesn't exist.");
    return conversation;
  }

  _summary(conversation) {
    return {
      id: conversation.id,
      title: conversation.title,
      model: conversation.model,
      pinned: conversation.pinned,
      private: Boolean(conversation.private),
      message_count: conversation.messages.filter((message) => message.role !== "tool").length,
      created_at: conversation.created_at,
      updated_at: conversation.updated_at,
    };
  }

  _list() {
    return this.conversations
      .slice()
      .sort((a, b) => {
        if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
        return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
      })
      .map((conversation) => this._summary(conversation));
  }

  _context(conversation) {
    const characters = conversation.messages.reduce(
      (total, message) => total + (message.content?.length || 0) + (message.reasoning?.length || 0),
      0,
    );
    const used = Math.max(180, Math.round(characters / 3.6) + conversation.messages.length * 6);
    const budget = 30720;
    return { used, budget, percent: Math.round((used / budget) * 100) };
  }

  _addMessage(conversation, partial) {
    const seq = (conversation.messages.at(-1)?.seq || 0) + 1;
    const message = {
      id: `m_mock_${++this.messageSeq}`,
      seq,
      role: "assistant",
      content: "",
      reasoning: "",
      tool_calls: null,
      tool_name: null,
      tool_call_id: null,
      model: this.currentModel,
      created_at: nowIso(),
      meta: {},
      ...partial,
    };
    conversation.messages.push(message);
    conversation.updated_at = nowIso();
    return message;
  }

  _createConversation(title = "New chat", isPrivate = false) {
    const conversation = {
      id: `c_mock_${++this.sequence}`,
      title,
      model: this.currentModel,
      pinned: false,
      private: Boolean(isPrivate),
      created_at: nowIso(),
      updated_at: nowIso(),
      summary: "",
      compacted: false,
      messages: [],
    };
    this.conversations.push(conversation);
    return conversation;
  }

  // "local" (Kiwibro runs models itself) unless a server address is set ("attached").
  _engineMode() {
    return this.settings.server_url ? "attached" : "local";
  }

  _engineError() {
    if (!this.offline) return null;
    return this._engineMode() === "attached"
      ? "The model server isn't reachable"
      : "The engine isn't installed yet";
  }

  /* Kiwi Mind: what the page sees. Bytes on disk come from the download in progress, if any. */
  _mindDone() {
    const dl = this.mindDl;
    if (!dl) return this.mindParam === "downloading" || this.mindParam === "error" ? MIND_PARTIAL : 0;
    if (dl.state !== "running") return dl.base;
    return Math.min(MIND_BYTES, dl.base + ((Date.now() - dl.startedAt) * MIND_BYTES) / MIND_SIM_MS);
  }

  _mindState() {
    if (this.mindDl?.state === "running" && this._mindDone() >= MIND_BYTES) {
      this.mindFinished = true;
      this.mindDl = null;
    }
    const param = this.mindParam;
    const installed = this.mindFinished || ["usable", "nonode", "damaged"].includes(param);
    const usable = installed && !["nonode", "damaged"].includes(param);
    let download = null;
    if (!installed) {
      if (this.mindDl) {
        download = { state: this.mindDl.state, done_bytes: Math.round(this._mindDone()), total_bytes: MIND_BYTES };
      } else if (param === "downloading") {
        download = { state: "running", done_bytes: MIND_PARTIAL, total_bytes: MIND_BYTES };
      } else if (param === "error") {
        download = {
          state: "error",
          done_bytes: MIND_PARTIAL,
          total_bytes: MIND_BYTES,
          error: "The download stopped: no internet connection.",
        };
      }
    }
    const model = this.models.find((entry) => entry.id === this.currentModel);
    const reasons = {
      nonode: "Kiwi Mind needs Node.js, which is not installed.",
      damaged: "Kiwi Mind's files are damaged. Download it again.",
    };
    let reason = "";
    if (!installed) reason = "Kiwi Mind is not installed.";
    else if (!usable) reason = reasons[param];
    return {
      usable,
      installed,
      version: installed ? "1.0.0" : null,
      bytes: installed ? MIND_BYTES : 0,
      node: param === "nonode" ? null : "/usr/bin/node",
      running: usable,
      pid: usable ? 4242 : null,
      rss_bytes: usable ? 1010000000 : 0,
      skills: installed ? [...MIND_SKILLS] : [],
      helper: model ? { id: model.id, name: model.name || model.id } : null,
      reason,
      download: installed ? null : download,
    };
  }

  _mindBrief() {
    const { usable, running, pid } = this._mindState();
    return { usable, running, pid };
  }

  _mindStart() {
    if (this._mindState().installed) return { ok: true, download: { state: "done", done_bytes: MIND_BYTES, total_bytes: MIND_BYTES } };
    if (!this.mindDl || this.mindDl.state !== "running") {
      this.mindDl = { state: "running", base: this._mindDone(), startedAt: Date.now() };
    }
    return { ok: true, download: this._mindState().download };
  }

  _mindCancel() {
    if (this._mindState().installed) return { ok: true, download: null };
    if (!this.mindDl || this.mindDl.state === "running") {
      this.mindDl = { state: "cancelled", base: this._mindDone(), startedAt: 0 };
    }
    return { ok: true, download: this._mindState().download };
  }

  _health() {
    const error = this._engineError();
    return {
      ok: true,
      engine: this._engineMode(),
      engine_ok: !this.offline,
      online: !this.offline, // ?offline=1 also stands for "no internet" here
      ...(error ? { engine_error: error } : {}),
      model: this.currentModel,
      busy: false,
      version: "3.0.0",
      mind: this._mindBrief(),
    };
  }

  _direct() {
    const available = !this.offline;
    return {
      available,
      error: this._engineError(),
      runtime: available ? clone(DIRECT_RUNTIME) : null,
      runtimes: [clone(DIRECT_RUNTIME)],
      servers: clone(this.directServers),
      gpu: this.settings.direct_gpu === "off" ? "" : "AMD Radeon Graphics (RADV RENOIR)",
      gpu_available: true,
      threads: 8,
      idle_sleep_s: 1800,
      model_dirs: ["~/.cache/lm-studio/models"],
      models_count: this.models.length,
      own_models_dir: "~/.local/share/kiwibro/models",
      own_runtime_dir: "~/.local/share/kiwibro/runtime",
      adapters: [],
      adapters_dir: "~/.local/share/kiwibro/adapters",
      sources: [{ source: "kiwibro", label: "Kiwibro", path: "~/.local/share/kiwibro/models", models: this.models.length, bytes: 0 }],
    };
  }

  _titleFromMessage(message) {
    const clean = String(message || "").replace(/\s+/g, " ").trim();
    const words = clean.split(" ").slice(0, 6).join(" ").replace(/[.,!?;:]+$/, "");
    return (words.length > 40 ? words.slice(0, 40) : words) || "New chat";
  }

  /* ------------------------------------------------------------- REST fake */

  async request(path, { method = "GET", body } = {}) {
    await sleep(50 + Math.random() * 90);
    const url = new URL(path, "http://mock.local");
    const route = url.pathname;
    const verb = method.toUpperCase();

    // The model manager and the notes panel keep their fake data in their own files.
    const fail = (status, code, message) => {
      throw new MockApiError(status, code, message);
    };
    for (const part of [mockModels, mockNotes]) {
      const answer = part(this, { route, verb, body, url, fail });
      if (answer !== undefined) return clone(answer);
    }

    if (route === "/health" && verb === "GET") return clone(this._health());

    if (route === "/api/mind" && verb === "GET") return this._mindState();
    if (route === "/api/mind/download" && verb === "POST") return this._mindStart();
    if (route === "/api/mind/download/cancel" && verb === "POST") return this._mindCancel();

    if (route === "/api/bootstrap" && verb === "GET") {
      return {
        health: clone(this._health()),
        settings: clone(this.settings),
        models: clone(this.models),
        current_model: this.currentModel,
        conversations: this._list(),
        memories_count: this.memories.length,
        notes_count: (this.notes || []).length,
        workspace: this.settings.workspace,
      };
    }

    if (route === "/api/models" && verb === "GET") {
      const error = this._engineError();
      return {
        ok: true,
        engine_ok: !this.offline,
        ...(error ? { engine_error: error } : {}),
        current: this.currentModel,
        models: clone(this.models),
      };
    }

    if (route === "/api/models/ready" && verb === "GET") {
      return { ok: true, readiness: { model: this.currentModel, phase: "ready", detail: "", context_length: null } };
    }

    if (route === "/api/models/select" && verb === "POST" && body?.id === "none" && this._mindState().usable) {
      this.currentModel = "none";
      this.settings.model = "none";
      return { ok: true, current: "none" };
    }
    if (route === "/api/models/select" && verb === "POST") {
      const model = this.models.find((entry) => entry.id === body?.id);
      if (!model) throw new MockApiError(400, "unknown_model", "That model isn't available.");
      this.currentModel = model.id;
      this.settings.model = model.id;
      return { ok: true, current: model.id };
    }

    if (route === "/api/conversations" && verb === "GET") {
      return { ok: true, conversations: this._list() };
    }

    if (route === "/api/conversations/search" && verb === "GET") {
      const q = (url.searchParams.get("q") || "").trim().toLowerCase();
      const matches = [];
      for (const conversation of this.conversations) {
        const hit = q.length >= 2 && conversation.messages.find((m) => (m.content || "").toLowerCase().includes(q));
        if (hit) matches.push({ id: conversation.id, snippet: hit.content.slice(0, 90) });
      }
      return { ok: true, matches };
    }

    if (route === "/api/conversations" && verb === "POST") {
      const conversation = this._createConversation(body?.title || "New chat");
      return {
        id: conversation.id,
        title: conversation.title,
        model: conversation.model,
        created_at: conversation.created_at,
        updated_at: conversation.updated_at,
        summary: null,
        messages: [],
      };
    }

    const conversationMatch = route.match(/^\/api\/conversations\/([^/]+)$/);
    if (conversationMatch) {
      const conversation = this._conversation(decodeURIComponent(conversationMatch[1]));
      if (verb === "GET") {
        return {
          ok: true,
          conversation: {
            id: conversation.id,
            title: conversation.title,
            model: conversation.model,
            created_at: conversation.created_at,
            updated_at: conversation.updated_at,
            summary: conversation.summary || null,
            compacted: Boolean(conversation.compacted),
            private: Boolean(conversation.private),
          },
          messages: clone(conversation.messages),
          context: this._context(conversation),
        };
      }
      if (verb === "PATCH") {
        if (typeof body?.title === "string" && body.title.trim()) conversation.title = body.title.trim();
        if (typeof body?.pinned === "boolean") conversation.pinned = body.pinned;
        if (typeof body?.private === "boolean") conversation.private = body.private;
        conversation.updated_at = nowIso();
        return {
          ok: true,
          conversation: {
            id: conversation.id,
            title: conversation.title,
            model: conversation.model,
            created_at: conversation.created_at,
            updated_at: conversation.updated_at,
            summary: conversation.summary || null,
            compacted: Boolean(conversation.compacted),
            private: Boolean(conversation.private),
          },
        };
      }
      if (verb === "DELETE") {
        this.conversations = this.conversations.filter((entry) => entry.id !== conversation.id);
        return { ok: true };
      }
    }

    const clearMatch = route.match(/^\/api\/conversations\/([^/]+)\/clear$/);
    if (clearMatch && verb === "POST") {
      const conversation = this._conversation(decodeURIComponent(clearMatch[1]));
      conversation.messages = [];
      conversation.summary = "";
      conversation.compacted = false;
      conversation.updated_at = nowIso();
      return {
        ok: true,
        conversation: {
          id: conversation.id,
          title: conversation.title,
          model: conversation.model,
          created_at: conversation.created_at,
          updated_at: conversation.updated_at,
          summary: null,
          compacted: false,
        },
      };
    }

    if (route === "/api/memories" && verb === "GET") {
      const query = (url.searchParams.get("q") || "").toLowerCase();
      const kind = url.searchParams.get("kind") || "";
      let memories = this.memories.slice();
      if (kind) memories = memories.filter((memory) => memory.kind === kind);
      if (query) {
        memories = memories.filter((memory) => memory.content.toLowerCase().includes(query));
      }
      memories.sort((a, b) => {
        if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
        return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
      });
      return { ok: true, memories: clone(memories), total: memories.length };
    }

    if (route === "/api/memories" && verb === "POST") {
      const content = String(body?.content || "").trim();
      if (content.length < 3) throw new MockApiError(400, "invalid_memory", "Memory is too short.");
      const memory = {
        id: `mem_mock_${++this.sequence}`,
        kind: body?.kind || "fact",
        content: content.slice(0, 300),
        pinned: false,
        confidence: 1,
        created_at: nowIso(),
        updated_at: nowIso(),
        last_used_at: null,
      };
      this.memories.unshift(memory);
      return { ok: true, memory: clone(memory) };
    }

    if (route === "/api/memories" && verb === "DELETE") {
      const deleted = this.memories.length;
      this.memories = [];
      return { ok: true, deleted };
    }

    const memoryMatch = route.match(/^\/api\/memories\/([^/]+)$/);
    if (memoryMatch) {
      const memory = this.memories.find((entry) => entry.id === decodeURIComponent(memoryMatch[1]));
      if (!memory) throw new MockApiError(404, "not_found", "That memory doesn't exist.");
      if (verb === "PATCH") {
        if (typeof body?.content === "string" && body.content.trim()) memory.content = body.content.trim().slice(0, 300);
        if (typeof body?.kind === "string" && ["fact", "preference", "project", "person", "decision"].includes(body.kind)) {
          memory.kind = body.kind;
        }
        if (typeof body?.pinned === "boolean") memory.pinned = body.pinned;
        memory.updated_at = nowIso();
        return { ok: true, memory: clone(memory) };
      }
      if (verb === "DELETE") {
        this.memories = this.memories.filter((entry) => entry.id !== memory.id);
        return { ok: true };
      }
    }

    if (route === "/api/engine" && verb === "GET") {
      return { ok: true, engine: this._engineMode(), direct: clone(this._direct()) };
    }

    if (route === "/api/engine/stop" && verb === "POST") {
      this.directServers = [];
      return { ok: true, direct: clone(this._direct()) };
    }

    if (route === "/api/settings" && verb === "GET") {
      return { ok: true, settings: clone(this.settings), data_dir: this.dataDir };
    }

    if (route === "/api/settings" && verb === "PATCH") {
      const next = body || {};
      if (next.permissions && typeof next.permissions === "object") {
        this.settings.permissions = { ...this.settings.permissions, ...next.permissions };
      }
      for (const [key, value] of Object.entries(next)) {
        if (key === "permissions") continue;
        this.settings[key] = value;
      }
      if (next.workspace && typeof next.workspace === "string") {
        this.settings.workspace = next.workspace;
      }
      return { ok: true, settings: clone(this.settings) };
    }

    // Safe copy is not part of the demo: every chat reads as having no copy open.
    if (route === "/api/safecopy/status" && verb === "GET") {
      return { ok: true, active: false, real: null, work: null, changed_files: 0, disk_bytes: 0, left_out: [], applied: false, can_undo: false };
    }
    if (route.startsWith("/api/safecopy/")) fail(404, "not_found", "Safe copy is not available in the demo.");

    if (route === "/api/workspace" && verb === "GET") {
      return { ok: true, root: this.settings.workspace, files: clone(this.workspaceFiles) };
    }

    if (route === "/api/chat/stop" && verb === "POST") {
      return { ok: true };
    }

    const approvalMatch = route.match(/^\/api\/approvals\/([^/]+)$/);
    if (approvalMatch && verb === "POST") {
      const id = decodeURIComponent(approvalMatch[1]);
      const resolver = this.approvals.get(id);
      if (!resolver) throw new MockApiError(404, "not_found", "That approval has expired.");
      this.approvals.delete(id);
      const decision = body?.decision;
      if (!["allow", "always", "deny"].includes(decision)) {
        throw new MockApiError(400, "invalid_decision", "Unknown approval decision.");
      }
      if (decision === "always" && resolver.tool) {
        this.settings.permissions[resolver.tool] = "auto";
      }
      resolver.resolve(decision);
      return { ok: true, decision };
    }

    throw new MockApiError(404, "not_found", `No mock route for ${verb} ${route}`);
  }

  /* ------------------------------------------------------------- streaming */

  chat({ conversationId, message, model, private: isPrivate = false }, { signal } = {}) {
    let conversation;
    try {
      conversation = conversationId ? this._conversation(conversationId) : this._createConversation("New chat", isPrivate);
    } catch (error) {
      return Promise.resolve({
        ok: false,
        status: 404,
        json: async () => ({ error: { code: "not_found", message: error.message } }),
      });
    }

    const encoder = new TextEncoder();
    const turn = {
      conversation,
      message,
      model: model || this.currentModel,
      aborted: Boolean(signal?.aborted),
      toolCallSeq: 0,
      approvalSeq: 0,
      content: "",
      reasoning: "",
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };

    const body = new ReadableStream({
      start: (controller) => {
        let closed = false;
        const send = (event, data) => {
          if (closed || turn.aborted) return;
          try {
            controller.enqueue(
              encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
            );
          } catch {
            closed = true;
          }
        };
        const finish = () => {
          if (closed) return;
          closed = true;
          try {
            controller.close();
          } catch {
            /* already closed */
          }
        };
        const fail = () => {
          if (closed) return;
          closed = true;
          try {
            controller.error(new DOMException("The operation was aborted.", "AbortError"));
          } catch {
            /* already errored */
          }
        };
        const onAbort = () => {
          turn.aborted = true;
          for (const pending of this.approvals.values()) pending.resolve("abort");
          this.approvals.clear();
          fail();
        };
        if (signal?.aborted) {
          fail();
          return;
        }
        signal?.addEventListener("abort", onAbort, { once: true });

        const wait = (ms) =>
          new Promise((resolve) => {
            if (turn.aborted) return resolve();
            const timer = window.setTimeout(() => {
              signal?.removeEventListener("abort", onAbort);
              resolve();
            }, ms);
            signal?.addEventListener(
              "abort",
              () => {
                window.clearTimeout(timer);
                resolve();
              },
              { once: true },
            );
          });

        this._runTurn({ turn, send, wait, signal })
          .catch((error) => {
            if (turn.aborted) return;
            send("error", {
              message: "The mock agent hit a snag.",
              code: "mock_error",
              recoverable: true,
            });
            void error;
          })
          .finally(() => {
            signal?.removeEventListener("abort", onAbort);
            if (!turn.aborted) finish();
          });
      },
    });

    return Promise.resolve({
      ok: true,
      status: 200,
      body,
      json: async () => ({}),
    });
  }

  async _streamText(turn, send, wait, event, text, options = {}) {
    const chunks = chunkText(text, options.size || 5);
    for (const chunk of chunks) {
      if (turn.aborted) return;
      if (event === "delta") turn.content += chunk;
      else turn.reasoning += chunk;
      send(event, { text: chunk });
      await wait(options.delay || 24);
    }
  }

  async _waitForApproval(turn, send, wait, approval) {
    let resolveDecision;
    const decisionPromise = new Promise((resolve) => {
      resolveDecision = resolve;
    });
    const entry = {
      resolve: (decision) => resolveDecision(decision),
      tool: approval.tool,
    };
    this.approvals.set(approval.approval_id, entry);
    send("approval", approval);
    const timeout = window.setTimeout(() => {
      if (this.approvals.get(approval.approval_id) === entry) {
        this.approvals.delete(approval.approval_id);
        resolveDecision("deny");
      }
    }, 120000);
    const decision = await decisionPromise;
    window.clearTimeout(timeout);
    for (const pending of this.approvals.values()) {
      if (pending === entry) this.approvals.delete(approval.approval_id);
    }
    if (turn.aborted) return "abort";
    return decision;
  }

  async _runTurn({ turn, send, wait, signal }) {
    const conversation = turn.conversation;
    const userMessage = this._addMessage(conversation, { role: "user", content: turn.message });
    send("meta", {
      conversation_id: conversation.id,
      user_message_id: userMessage.id,
      model: turn.model,
      title: conversation.title,
      context: this._context(conversation),
    });

    // Who writes the reply: Kiwi Mind when usable; a message that says "helper" also shows the helper.
    const mind = this._mindState();
    if (mind.usable) {
      send("brain", { brain: "kiwimind", skill: "chat", hard: false });
      if (mind.helper && /\bhelper\b/i.test(turn.message)) {
        send("brain", { brain: "helper", name: mind.helper.name, skill: null, hard: true });
      }
    }

    const wantsTool = /\b(search|web|save|file|command|shell|run|notes|news|script)\b/i.test(turn.message);
    try {
      if (/\bcomponents?\b/i.test(turn.message)) await this._componentFlow({ turn, send, wait });
      else if (wantsTool) await this._toolFlow({ turn, send, wait, signal });
      else await this._answerFlow({ turn, send, wait, signal });
    } catch (error) {
      if (!turn.aborted) throw error;
    }

    if (turn.aborted) return;

    if (conversation.title === "New chat") {
      conversation.title = conversation.id === "c_mock_kiwi" ? conversation.title : this._titleFromMessage(turn.message);
      conversation.updated_at = nowIso();
      send("title", { id: conversation.id, title: conversation.title });
    }

    const finalMessage = conversation.messages.filter((entry) => entry.role === "assistant").at(-1);
    send("done", {
      message_id: finalMessage?.id || null,
      aborted: false,
      model: turn.model,
      usage: turn.usage,
    });
  }

  async _answerFlow({ turn, send, wait }) {
    send("status", { phase: "thinking", label: "Thinking" });
    await wait(420);
    const reasoning =
      "The user is asking for a compact, useful overview. A short answer with a couple of bullets and one example should be enough.";
    await this._streamText(turn, send, wait, "reasoning", reasoning, { size: 8, delay: 18 });
    await wait(200);

    const answer = `I'm Kiwibro, an AI agent that runs on this computer. Your chats stay here.

Here's what I can do:

- **Chat** with any GGUF model on this computer
- **Use tools**: search the web, read and write files, run shell commands
- **Remember** durable facts between conversations
- **Compact** long chats automatically, so you never hit a context wall

A tiny example:

\`\`\`python
from pathlib import Path

notes = sorted(Path(".").glob("*.md"))
for note in notes:
    print(note.name)
\`\`\`

Ask me to *find something online*, *edit a file* or *run a command*. I ask before I change anything.`;

    await this._streamText(turn, send, wait, "delta", answer, { size: 6, delay: 26 });
    this._addMessage(turn.conversation, {
      role: "assistant",
      content: answer,
      reasoning,
      model: turn.model,
    });

    const compacted = /summar/i.test(turn.message);
    send("context", {
      used: compacted ? 26800 : 1840,
      budget: 30720,
      percent: compacted ? 87 : 6,
      compacted,
    });
    if (/remember|prefer|my name/i.test(turn.message)) {
      const memory = {
        id: `mem_mock_${++this.sequence}`,
        kind: "preference",
        content: "Prefers short answers with bullets and code examples.",
        pinned: false,
        confidence: 1,
        created_at: nowIso(),
        updated_at: nowIso(),
        last_used_at: null,
      };
      this.memories.unshift(memory);
      send("memory", { added: 1, updated: 0, removed: 0 });
    }
  }

  // A reply that looks through the component library (mock data only).
  async _componentFlow({ turn, send, wait }) {
    const conversation = turn.conversation;
    send("status", { phase: "tool", label: "Using a tool" });
    await wait(300);
    const callId = `call_mock_${++turn.toolCallSeq}`;
    const args = { action: "search", query: "web page starter" };
    send("tool_call", { id: callId, name: "component", args, risk: "write", requires_approval: false });
    await wait(700);
    const preview = "mock/web-page — a starter for one static page\nmock/api-json — a small JSON API with tests";
    send("tool_result", { id: callId, name: "component", ok: true, duration_ms: 690, preview });
    this._addMessage(conversation, {
      role: "assistant",
      content: "",
      model: turn.model,
      tool_calls: [{ id: callId, type: "function", function: { name: "component", arguments: JSON.stringify(args) } }],
    });
    this._addMessage(conversation, {
      role: "tool",
      content: preview,
      tool_call_id: callId,
      tool_name: "component",
      meta: { ok: true, duration_ms: 690 },
    });
    const answer = "The component library has a starter for a single web page. Start from `mock/web-page`.";
    await this._streamText(turn, send, wait, "delta", answer, { size: 6, delay: 22 });
    this._addMessage(conversation, { role: "assistant", content: answer, model: turn.model });
  }

  async _toolFlow({ turn, send, wait, signal }) {
    const conversation = turn.conversation;
    // Mirrors plan.TurnPlan.statuses() for "search…, summarise…, then save a note".
    const steps = [
      { n: 1, text: "look up the latest kiwi conservation news", kind: "web", done: false },
      { n: 2, text: "summarise it in a few lines", kind: "answer", done: false },
      { n: 3, text: "save a short note in the workspace", kind: "write", done: false, target: "kiwi-notes.md" },
    ];
    const plan = (...done) => {
      for (const n of done) steps[n - 1].done = true;
      send("plan", { steps: steps.map((step) => ({ ...step })) });
    };
    plan();
    send("status", { phase: "thinking", label: "Thinking" });
    await wait(360);
    const reasoning =
      "I should look up current kiwi conservation news, summarise it in a few lines, then save a short note in the workspace. Writing a file needs approval first.";
    await this._streamText(turn, send, wait, "reasoning", reasoning, { size: 7, delay: 18 });

    const intro = "I'll check the latest conservation news and save a short note.";
    await this._streamText(turn, send, wait, "delta", intro, { size: 6, delay: 22 });
    this._addMessage(conversation, { role: "assistant", content: intro, model: turn.model });

    send("status", { phase: "web", label: "Searching the web" });
    await wait(260);
    const callId = `call_mock_${++turn.toolCallSeq}`;
    send("tool_call", {
      id: callId,
      name: "web_search",
      args: { query: "kiwi conservation news", max_results: 4 },
      risk: "low",
      requires_approval: false,
    });
    await wait(880);
    send("tool_result", {
      id: callId,
      name: "web_search",
      ok: true,
      duration_ms: 842,
      preview: SEED_NEWS.map((item) => `${item.title} — ${item.url}`).join("\n"),
    });
    plan(1);
    this._addMessage(conversation, {
      role: "assistant",
      content: "",
      model: turn.model,
      tool_calls: [
        {
          id: callId,
          type: "function",
          function: {
            name: "web_search",
            arguments: JSON.stringify({ query: "kiwi conservation news", max_results: 4 }),
          },
        },
      ],
    });
    this._addMessage(conversation, {
      role: "tool",
      content: SEED_NEWS.map((item) => `${item.title} — ${item.url}`).join("\n"),
      tool_call_id: callId,
      tool_name: "web_search",
      meta: { ok: true, duration_ms: 842 },
    });

    const summary = `Here's the short version:

- **Island sanctuaries** are reporting steady kiwi population growth.
- **Community trapping** removed tens of thousands of predators this season.
- **Rowi** and **Haast tokoeka** are still the rarest species.

I'll save these three points for you.`;
    await this._streamText(turn, send, wait, "delta", summary, { size: 6, delay: 24 });
    this._addMessage(conversation, { role: "assistant", content: summary, model: turn.model });

    send("status", { phase: "tool", label: "Preparing a file" });
    await wait(240);
    const fileContent =
      "# Kiwi conservation — notes\n\n- Island populations are growing\n- Community trapping is working\n- Rowi and Haast tokoeka remain rarest\n\nSources: doc.govt.nz, savethekiwi.nz\n";
    const writeCallId = `call_mock_${++turn.toolCallSeq}`;
    send("tool_call", {
      id: writeCallId,
      name: "write_file",
      args: { path: "kiwi-notes.md", content: fileContent },
      risk: "medium",
      requires_approval: true,
    });
    this._addMessage(conversation, {
      role: "assistant",
      content: "",
      model: turn.model,
      tool_calls: [
        {
          id: writeCallId,
          type: "function",
          function: {
            name: "write_file",
            arguments: JSON.stringify({ path: "kiwi-notes.md", content: fileContent }),
          },
        },
      ],
    });

    const approval = {
      approval_id: `ap_mock_${++turn.approvalSeq}`,
      tool: "write_file",
      args: { path: "kiwi-notes.md", content: fileContent },
      summary: "Save a 6-line summary to kiwi-notes.md",
      risk: "write",
      timeout_s: 120,
      // Mirrors safety.decide for this call under the supervised autonomy mode.
      level: "low",
      reasons: ["writes new file kiwi-notes.md"],
      source: "permission",
    };
    const decision = await this._waitForApproval(turn, send, wait, approval);
    if (decision === "abort" || turn.aborted) return;

    if (decision === "deny") {
      send("tool_result", {
        id: writeCallId,
        name: "write_file",
        ok: false,
        duration_ms: 12,
        preview: "Denied by user — the file was not written.",
      });
      this._addMessage(conversation, {
        role: "tool",
        content: "Denied by user",
        tool_call_id: writeCallId,
        tool_name: "write_file",
        meta: { ok: false, duration_ms: 12 },
      });
      const denied = "Understood — I didn't write anything. The summary is above if you want to copy it.";
      await this._streamText(turn, send, wait, "delta", denied, { size: 7, delay: 24 });
      this._addMessage(conversation, { role: "assistant", content: denied, model: turn.model });
      send("context", { used: 4320, budget: 30720, percent: 14, compacted: false });
      return;
    }

    await wait(420);
    send("tool_result", {
      id: writeCallId,
      name: "write_file",
      ok: true,
      duration_ms: 96,
      preview: "Wrote kiwi-notes.md (312 bytes)",
    });
    plan(3);
    this._addMessage(conversation, {
      role: "tool",
      content: "Wrote kiwi-notes.md (312 bytes)",
      tool_call_id: writeCallId,
      tool_name: "write_file",
      meta: { ok: true, duration_ms: 96 },
    });

    const finale =
      "Saved to `kiwi-notes.md` in your workspace. Want me to draft a short update for the team too?";
    await this._streamText(turn, send, wait, "delta", finale, { size: 5, delay: 22 });
    this._addMessage(conversation, { role: "assistant", content: finale, model: turn.model });
    plan(2);

    send("context", { used: 5840, budget: 30720, percent: 19, compacted: false });
    this.memories.unshift({
      id: `mem_mock_${++this.sequence}`,
      kind: "project",
      content: "Keeps kiwi conservation notes in kiwi-notes.md.",
      pinned: false,
      confidence: 0.9,
      created_at: nowIso(),
      updated_at: nowIso(),
      last_used_at: null,
    });
    send("memory", { added: 1, updated: 0, removed: 0 });
    this.workspaceFiles = [
      { path: "kiwi-notes.md", type: "file", size: 312, modified: nowIso() },
      ...this.workspaceFiles.filter((file) => file.path !== "kiwi-notes.md"),
    ];
    void signal;
  }
}
