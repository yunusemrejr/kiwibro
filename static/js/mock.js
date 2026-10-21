/**
 * Development-only fake backend, activated with ?mock=1.
 * It mirrors docs/API.md (REST shapes and SSE events) so the whole UI can be
 * exercised without Flask or LM Studio. It has no effect in normal mode.
 */

const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
const clone = (value) => JSON.parse(JSON.stringify(value));
const nowIso = () => new Date().toISOString();
const isoAgo = (ms) => new Date(Date.now() - ms).toISOString();

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
  undo_changes: "auto",
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
    content: "Runs LM Studio on 127.0.0.1:1234 with qwen3.8-2b loaded.",
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
    this.lmstudio = !this.offline;
    this.settings = {
      model: "qwen3.8-2b",
      lmstudio_url: "http://127.0.0.1:1234",
      agent_enabled: true,
      memory_enabled: true,
      max_output_tokens: 2048,
      temperature: 0.6,
      workspace: "/home/yemre/Kiwibro",
      user_name: "",
      tool_profile: "auto",
      max_tool_rounds: 16,
      permissions: clone(DEFAULT_PERMISSIONS),
    };
    this.memories = clone(SEED_MEMORIES);
    this.workspaceFiles = clone(WORKSPACE_FILES);
    this.dataDir = "/home/yemre/.local/share/kiwibro (mock)";
    this.approvals = new Map();
    this.sequence = 0;
    this.messageSeq = 0;
    this.conversations = [];
    this._seedConversations();
    this.health = {
      ok: true,
      version: "1.0.0",
      lmstudio: this.lmstudio,
      lmstudio_url: this.settings.lmstudio_url,
      model: this.currentModel,
    };
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

  _createConversation(title = "New chat") {
    const conversation = {
      id: `c_mock_${++this.sequence}`,
      title,
      model: this.currentModel,
      pinned: false,
      created_at: nowIso(),
      updated_at: nowIso(),
      summary: "",
      compacted: false,
      messages: [],
    };
    this.conversations.push(conversation);
    return conversation;
  }

  _health() {
    return {
      ok: true,
      version: "1.0.0",
      lmstudio: this.lmstudio,
      lmstudio_url: this.settings.lmstudio_url,
      model: this.currentModel,
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

    if (route === "/health" && verb === "GET") return clone(this._health());

    if (route === "/api/bootstrap" && verb === "GET") {
      return {
        health: clone(this._health()),
        settings: clone(this.settings),
        models: clone(this.models),
        current_model: this.currentModel,
        conversations: this._list(),
        memories_count: this.memories.length,
        workspace: this.settings.workspace,
      };
    }

    if (route === "/api/models" && verb === "GET") {
      return { ok: true, lmstudio: this.lmstudio, current: this.currentModel, models: clone(this.models) };
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
          },
          messages: clone(conversation.messages),
          context: this._context(conversation),
        };
      }
      if (verb === "PATCH") {
        if (typeof body?.title === "string" && body.title.trim()) conversation.title = body.title.trim();
        if (typeof body?.pinned === "boolean") conversation.pinned = body.pinned;
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

  chat({ conversationId, message, model }, { signal } = {}) {
    let conversation;
    try {
      conversation = conversationId ? this._conversation(conversationId) : this._createConversation();
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

    const wantsTool = /\b(search|web|save|file|command|shell|run|notes|news|script)\b/i.test(turn.message);
    try {
      if (wantsTool) await this._toolFlow({ turn, send, wait, signal });
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

    const answer = `I'm your local agent — nothing here leaves the machine.

Here's what I can do:

- **Chat** with any model you've loaded in LM Studio
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

Ask me to *find something online*, *edit a file*, or *run a command* — I'll ask before touching anything.`;

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

  async _toolFlow({ turn, send, wait, signal }) {
    const conversation = turn.conversation;
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
