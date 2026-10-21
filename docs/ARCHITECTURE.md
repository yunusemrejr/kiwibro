# Kiwibro — Architecture & Build Contract

Kiwibro is a local-first AI agent for Linux: a **Python Flask** backend serving a
**browser GUI** designed to feel like a native Ubuntu/GNOME application, talking to
**LM Studio** on `127.0.0.1:1234` (OpenAI-compatible API). No cloud, no accounts,
no external CDNs, no bundled JS frameworks.

Version 1.1.0 · MIT · Python 3.10+ · Flask 3

---

## 1. Product principles

1. **Local.** Everything runs on the machine. UI assets are local. Web access only
   through the explicit web tools.
2. **Low text.** The interface is quiet: icons, short labels, no explanatory walls,
   no setup wizard. One composer, one place to talk.
3. **Never lose context.** Automatic token budgeting + rolling conversation
   summaries + hard fallbacks mean the user never sees a "context length exceeded"
   error. Long chats keep working; old detail becomes summary.
4. **Memory without chaos.** Durable facts are extracted, deduplicated, categorised,
   searchable, editable and capped. Switching models or starting a new chat never
   changes what is remembered or rewrites past transcripts.
5. **Safe autonomy.** Tool calls are reviewed. Read-only tools are auto-allowed;
   writes and shell commands ask first, with an "always" opt-in. File access is
   sandboxed to the workspace; shell commands are blocklisted, time-limited and
   run in the workspace.
6. **Native feel.** A GNOME-style window: header bar, model pill, sidebar, sheets,
   subtle motion, Ubuntu typography, dark theme, a hand-drawn kiwi mark. Installed
   with an app icon and `.desktop` launcher that opens a Chrome app window
   (browser GUI, no Electron).

---

## 2. Repository layout

```
kiwibro/
├── app.py                  # Flask entry point: create_app(), python3 app.py
├── kiwibro/                # Python package (backend)
│   ├── __init__.py         # create_app, extension wiring
│   ├── config.py           # paths, env, defaults, settings helpers
│   ├── db.py               # SQLite schema, connections, CRUD helpers
│   ├── llm.py              # LM Studio client: models, streaming, tools, JSON calls
│   ├── context.py          # token estimation, budget, rolling compaction
│   ├── memory.py           # extraction, dedupe, retrieval, rendering
│   ├── prompts.py          # system / summary / memory / title prompts
│   ├── tools.py            # tool catalog, dispatch, permission evaluation
│   ├── search.py           # web search providers + page fetching/SSRF guard
│   ├── sandbox.py          # workspace path + shell safety
│   ├── agent.py            # agent loop, SSE event queue, approvals, abort
│   └── api.py              # HTTP API + SSE endpoints
├── templates/index.html    # single page shell
├── static/
│   ├── css/app.css
│   ├── js/*.js             # ES modules, no build step
│   └── img/                # kiwi logo SVG + PNG icons
├── scripts/                # run.sh, kiwibro.sh, install.sh, kiwibro.desktop
├── tests/                  # pytest unit + API tests
├── docs/                   # architecture, API contract, screenshots, QA
├── requirements.txt
└── README.md
```

Data lives **outside** the repo:

- Database: `~/.local/share/kiwibro/kiwibro.db` (override `KIWIBRO_DATA_DIR`).
- Workspace: `~/Kiwibro` (override in settings).
- Logs: `~/.local/share/kiwibro/kiwibro.log` (rotating optional).

---

## 3. Runtime

- `python3 app.py` starts Flask on `127.0.0.1:8777` (override `KIWIBRO_HOST`,
  `KIWIBRO_PORT`). Threaded server, no debug/reloader by default.
- Single process. In-memory state: active turns, pending approvals, abort flags,
  search cache. Everything durable is in SQLite (WAL mode).
- One turn per conversation at a time (`409` if already running).
- Bind loopback only. Local app, no auth. Add `X-Kiwibro` response header maybe not
  needed. Reject requests whose `Host` is not localhost/127.0.0.1 to avoid DNS
  rebinding.
- `/health` exposes `started_at` (Unix seconds). The desktop launcher
  (`scripts/kiwibro.sh`) compares it with the newest source mtime and restarts a
  running server when the repository changed, so the app always serves current code.

### Prompt identity

Every chat request builds a system prompt (`kiwibro/prompts.py` +
`Agent._system_prompt`) that tells the model exactly what it is and what it can do:

- identity: Kiwibro, a local AI agent on the user's Linux machine, running through
  LM Studio (never a cloud service);
- the current LM Studio model id, the workspace path, and the current local time;
- the tools available this turn (and their auto/ask permission split), or an explicit
  "no tools" notice when agent mode is off or the model lacks tool support;
- memory and conversation summary blocks added by the memory/context engines.

Utility prompts (titles, rolling summaries, memory extraction) also identify
themselves as Kiwibro's, so even background calls stay on-brand.

---

## 4. LM Studio integration

Base: `http://127.0.0.1:1234` (settings key `lmstudio_url`, no trailing slash).

### Models. `GET /api/v0/models`

Returns richer metadata than `/v1/models`. Observed fields:

```json
{"data": [{
  "id": "qwen3.8-2b", "object": "model", "type": "llm",
  "publisher": "empero-ai", "arch": "qwen35", "quantization": "Q4_K_M",
  "state": "loaded", "max_context_length": 262144, "loaded_context_length": 32768,
  "capabilities": ["tool_use"]
}]}
```

`type` is `llm`, `vlm` or `embeddings`. Only `llm`/`vlm` are selectable chat
models. `state` is `loaded` or `not-loaded`; LM Studio just-in-time loads models on
the first request, which can take 30–120 s. `capabilities` may be absent.

List endpoint falls back to `/v1/models` (`{data:[{id}]}`) if `/api/v0/models` is
missing. Always expose `lmstudio_ok` and a human-readable error to the UI.

### Chat. `POST /v1/chat/completions`

OpenAI-compatible; supports `stream: true`, `tools`, `tool_choice`,
`temperature`, `max_tokens`, and `stream_options: {"include_usage": true}`.

Streaming chunks observed from LM Studio:

```json
{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
{"choices":[{"delta":{"reasoning_content":"Let me think"}}]}
{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function",
  "function":{"name":"web_search","arguments":""}}]}}]}
{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q"}}]}}]}
{"choices":[{"finish_reason":"tool_calls"}]}
{"usage":{"prompt_tokens":278,"completion_tokens":58,"total_tokens":336},...}
```

Delta fragments for `content`, `reasoning_content`, tool-call `arguments` must be
accumulated in arrival order, grouped by `tool_calls[].index`. Parse SSE lines that
start with `data: `; stop at `[DONE]`. Some models stream only
`reasoning_content` for a while — keep waiting, show it in the UI as a collapsed
"thinking" line.

Tool turn format:

- assistant: `{"role":"assistant","content":"","tool_calls":[{"id":"...","type":"function","function":{"name":"...","arguments":"{...json...}"}}]}`
- tool result: `{"role":"tool","tool_call_id":"...","content":"<text>"}`

Requests: connect timeout 5 s, read timeout 600 s for streaming (JIT load), 300 s
for non-streaming. Use `requests`, `stream=True`, iterate `response.iter_lines()`.

### Non-streaming calls

Used for summarisation, memory extraction, auto-title, with `stream: false`,
`temperature: 0`, `max_tokens` ≤ 900. Parse content defensively (strip code fences,
extract the first JSON array/object if asked for JSON). Models without tool support
are handled: if `capabilities` lacks `tool_use`, do not send tools; if LM Studio
returns a 400 mentioning tools, retry once without tools.

### Errors

- Connection refused → UI state "LM Studio not running", status card in settings,
  chat input disabled with a calm hint. No tracebacks in the UI.
- HTTP 400 with body containing context/n_ctx/token/too long → **auto-recover**:
  aggressive compaction, retry up to 2 times, then hard-trim and retry once.
- 5xx / timeouts → one retry for non-critical generation; stream errors emit
  `event: error` with `recoverable: true` and a short message.
- Never surface raw HTML or Python tracebacks; log details to the log file.

---

## 5. Context management

Token estimation (no tokenizer dependency): `estimate_tokens(text)`, roughly
`ascii_chars / 3.6 + non_ascii_chars * 0.9` (C-speed counting: `str.isascii`
fast path / one `encode('ascii','ignore')` pass), plus 4 tokens per message
overhead and a 1.1 safety factor on totals. Server-reported `prompt_tokens`
calibrate the estimator per model (EMA, clamped 0.75–1.4), so budgets track the
real tokenizer instead of drifting with the heuristic.

Budget per request:

```
models = fetch model metadata
context = loaded_context_length if state == loaded else min(max_context_length, 32768)
budget  = context - reserve(output) - 512
reserve(output) = min(settings.max_output_tokens, max(512, context // 8))
```

History sent: system prompt + retrieval memory block + messages with
`seq > conversation.summary_through_seq` + newest user message. Tool transcripts are
included (they are part of the past) with age-decayed caps: the 2 newest tool
results keep ~4000 chars, the next 4 get ~1200, older ones ~300 (head+tail, so
conclusions survive). Stale tool output no longer dominates long prompts.
Inside one turn, tool rounds after the first reuse the prepared context via
`ContextEngine.extend` (append-only, full `prepare` fallback past the trigger)
instead of rebuilding history from SQLite every round.

Automatic compaction (`ContextEngine.prepare`):

1. Compute estimate at ~75 % of budget before every LLM call. If over, summarise
   only the oldest slice needed to land at ~92 % of the trigger (keeping at least
   the last 4 messages verbatim, and more whenever that suffices) into
   `conversations.summary`, merged with the previous summary, advancing
   `summary_through_seq`. The summariser's transcript represents the whole
   segment (per-message caps shrink adaptively) so old detail is condensed, not
   dropped unseen. One summariser call, non-streaming, bounded (summary ≤ 600
   tokens). Repeat up to 3 passes if needed.
2. If still over: drop old tool results first, then hard-trim oldest messages, and
   set a `trimmed` flag. The user sees only a subtle "context compacted" chip.
3. Keep the newest user message and the last assistant reply always.
4. On LM Studio context-limit HTTP 400: summarise half of the remaining history and
   retry; after 2 failures truncate to system prompt + last 6 messages and retry.

The UI receives `event: context {used, budget, percent, compacted}` after context
work and on `meta`, rendered as a minimal ring/bar near the composer.

---

## 6. Memory

Durable, global (not per conversation), SQLite-backed.

Schema: `memories(id, kind, content, normalized, tags, source_conversation_id,
confidence, pinned, created_at, updated_at, last_used_at)` with FTS5 index.
Kinds: `fact`, `preference`, `project`, `person`, `decision`.

- **Extraction**: after each completed assistant turn (after `done` is emitted), if
  enabled, a background daemon worker asks the selected model for strict JSON operations
  (`{"ops":[{"op":"add|update|delete","id?":"...","kind":"...","content":"..."}]}`)
  given the last few turns plus a compact list of existing memories (top ~40).
  Validate: content 3–300 chars, kind whitelist, no duplicates, max 3 ops applied
  per turn; cap 400 memories total (prune oldest low-confidence non-pinned).
- **Dedupe**: normalise (lowercase, collapse whitespace, strip punctuation) and
  compare with `difflib.SequenceMatcher` ≥ 0.86 or token Jaccard ≥ 0.72; merge by
  keeping the clearer/longer content and bumping `updated_at`.
- **Retrieval for prompt**: pinned memories first, then FTS5 matches for the newest
  user text, then most recently used; render compact bullets under
  `## Memory` capped at ~800 tokens. Mark retrieved memories' `last_used_at`.
- **Explicit remember tool**: `remember(content, kind)` writes immediately.
- **Titles**: conversation titles are derived deterministically from the first user
  message (fast, no extra model round-trip); a model-written title is accepted only
  when it validates, so `done` never waits on it.
- **User control**: memory sheet with search, add, edit, pin, delete, and a
  "forget all" action; an SSE `memory` event tells the UI to refresh counts.
- Model swap or a new conversation never touches memories or past messages; the
  transcript stores the model used per message but is otherwise immutable.

---

## 7. Agent loop & tools

`POST /api/chat` starts a worker thread per turn that pushes SSE events into a
queue; the HTTP response streams that queue. Events and approval flow are in
`docs/API.md`.

Tool rounds per turn are configurable (`max_tool_rounds` setting, default 16, range 1–32). After the final step, append a short note if truncated.
Every assistant tool-call message is stored; every tool result is stored as a
`tool` message (content capped). Reasoning text is stored on assistant messages but
not re-sent as history.

Tool catalog (OpenAI function schemas):

| tool | args | permission default | notes |
|---|---|---|---|
| `web_search` | `query`, `max_results?` (1–8) | auto | Brave HTML → Bing HTML → DuckDuckGo lite → Wikipedia API; 10 min memory cache; 8 s per provider |
| `fetch_page` | `url`, `max_chars?` (≤8000) | auto | http(s) only; SSRF guard (reject localhost/private/link-local/IPv6 unique-local, re-resolve redirects); bs4 readable text |
| `read_file` | `path`, `start?`, `end?` | auto | workspace-sandboxed, ≤200 KB |
| `list_files` | `path?` | auto | depth ≤ 2, ≤ 200 entries, hidden-heavy dirs pruned |
| `write_file` | `path`, `content` | ask | workspace-sandboxed, mkdir parents, ≤256 KB, preview in approval card |
| `replace_in_file` | `path`, `old`, `new` | ask | exactly-one-match replace, workspace-sandboxed |
| `run_command` | `command`, `cwd?` | ask | cwd inside workspace, 60 s timeout, 6 KB output cap, blocklist |
| `remember` | `content`, `kind?` | auto | writes memory immediately |

### Tool layer 2.0 (35 built-ins + extensions)

Tools live in `kiwibro/tools.py` (`ToolSpec` catalog) with implementations split by
domain: `kiwibro/research.py` (parallel `multi_search`/`fetch_pages`/`web_research`
evidence packs with `[n]` citations, URL dedupe + term-overlap rerank, fetch cache),
`kiwibro/codebase.py` (`glob_files`, `file_outline`, atomic `apply_patch` with diff,
`file_manage`, `multi_command` pipelines, read-only `git`, sandboxed
`run_python`/`run_javascript` snippets in `.kiwibro-sandbox/` scratch dirs),
`kiwibro/desktop.py` (`system_info`, `open_path`, `clipboard`, `notify`,
`screenshot` — all degrade with a plain message when the desktop binary is absent),
`kiwibro/delegate.py` + `kiwibro/routing.py` (capability registry over
`/api/v0/models`; `model_info`, `delegate`, `describe_image` and map-reduce
`study_file` keep the main model in charge while the *smallest adequate* installed
model answers bounded side-questions, JIT-loaded and best-effort unloaded).
Persistent terminals (`terminal_session`), grep (`grep_files`) and batched reads
(`read_files`) ride on `kiwibro/sandbox.py`; the persistent `task_board` and the
multi-tab `browser` ride on `kiwibro/tasks.py` / `kiwibro/browser.py`.
Optional plug-ins register through `kiwibro/extensions.py` into the same catalog.

Efficiency rules of the loop:

- **Tool profiles** (`tool_profile` setting: auto/full/minimal/coding/research/
  desktop) send a task-shaped schema subset per turn — fixed for the turn so the
  prompt prefix stays cache-stable — minus user-denied tools.
- **Parallel batches**: approvals are sequential, then independent read/web calls
  run concurrently (`MAX_PARALLEL_TOOLS`); writes/commands stay serial. Results
  are recorded in original call order.
- **Loop guard**: an identical (tool, args) call is refused on its third repeat in
  a turn; bursts of writes/deletes are graded by `safety.decide` against
  `MAX_WRITES_PER_TURN` (escalating to ask, or deny in autonomous mode).
- **Result shaping**: long outputs are head+tail truncated before storage; the
  context engine additionally decays stale tool results with age.
- **Steering**: `POST /api/chat/steer` queues a user note mid-turn; the worker
  persists it as a user message at the next round boundary (SSE `steering`).
- **Safeguards**: `apply_patch` is all-or-nothing; every write-risk call is
  checkpointed first (see below); recursive deletes refuse
  >100 entries at once; snippets run without a shell under timeout/output caps.
- **Turn stats**: `done` carries `stats {tool_calls, tool_rounds, duration_ms}`.

### Tool-call pipeline for small models

Small (1–4B, heavily quantised) models are the design target, so every call passes a
pipeline that turns near-misses into valid calls and keeps actions reversible
(`agent._prepare_tool_call` → `run_one`):

1. **Rescue** (`kiwibro/toolcall.py`). Arguments go through a lenient JSON parser
   (single quotes, bare keys, missing/trailing commas, comments, Python literals,
   unescaped inner quotes, missing closers). Tool names resolve through aliases and
   fuzzy matching (`websearch`, `bash`, `functions.web_search`); argument names map
   onto the schema (`q`→`query`, `file`→`path`, `start_line`→`start`); values are
   coerced and clamped to it (`"20"`→`8` for `max_results`); `{"arguments": {…}}`
   wrappers are peeled. A validation error returns the exact signature plus an
   example, and the result notes what was fixed so the model learns the right form.
2. **Text-written calls**. With no structured `tool_calls`, `extract_calls` recovers
   calls written as text: Hermes `<tool_call>`, Qwen `<function=…>`, Llama
   `<|python_tag|>`, Mistral `[TOOL_CALLS]`, harmony `to=functions.x`, DeepSeek
   markers and `tool_code` fences always count; bare JSON, ReAct and `name(…)` lines
   count only for known tools, outside "for example" prose, when the call dominates
   the reply. `StreamFilter` holds call markup back while streaming. Models LM Studio
   does not flag for tool use get the **text tool protocol**: a byte-stable manual in
   message 0 (settings-only tool set) and tool history rewritten as plain text.
3. **Decide**. `safety.decide` (policy, command risk, autonomy, breakers) merges
   `guard.inspect_call` (content); the stricter verdict wins. The guard refuses
   web-bound arguments that carry secrets, blocks overwrites containing elision
   placeholders (`# ... rest unchanged`) or clobbering binary files, and escalates
   large shrinks, big folder deletes and overwriting moves.
4. **Checkpoint** (`kiwibro/checkpoints.py`). Before a write-risk call runs, every
   file it may touch is snapshotted into `<data dir>/checkpoints` (zlib,
   content-addressed, journaled per conversation; ≤512 MB / 20k rows, pruned oldest
   first). `undo_changes` restores the newest change(s), optionally for one path; a
   restore snapshots the current state first, so undoing never loses data.
5. **Execute** under `safety.applied`; `guard.reveal` fills `[SECRET_n]`
   placeholders back in for local write/exec tools only.
6. **Postprocess** (`guard.postprocess`). Secrets in output (provider keys, tokens,
   private keys, URL passwords, high-entropy credential assignments) become stable
   `[SECRET_n]` placeholders held in memory only, so they never reach chat history.
   Output scoring as prompt injection (override phrases, chat-template tokens,
   exfiltration or "don't tell the user" requests, Unicode tag characters) is fenced
   with a warning and taints the conversation for three turns, which makes
   `safety.decide` escalate exec/write calls. Web and document output also has
   chat-template tokens broken up (`< |im_start| >`) and invisible characters removed.
7. **Page** (`library.page_output`). Output longer than ~15% of the context window
   (1.5–8k chars) is stored in the document library as `D<n>`; the context keeps
   head + tail plus `doc_read`/`doc_search` hints, so nothing is lost and small
   windows are not flooded.

An empty reply after tool results gets one transient nudge that is never stored.

**Documents** (`kiwibro/library.py`, separate `library.db`). FTS5/BM25 index with
streaming extraction (PDF via `pdftotext` with a pure-Python fallback, DOCX, ODT,
EPUB, PPTX, XLSX, HTML, Markdown, text, code, CSV, JSON) into ~1k-char chunks that
keep headings and locations (page, lines, slide, sheet). `doc_search` first indexes
new or changed workspace documents (time-budgeted; files over 8 MB via `doc_add`);
`doc_read` pages through a document with explicit `Next:` hints; `doc_list` shows
documents or one outline. Memory stays flat: a 28 MB log indexes with ~3 MB RSS growth.
`convert_document` (`kiwibro/convert.py`) routes conversions to installed tools — pandoc
(text formats), pandoc + WeasyPrint (PDF output), LibreOffice (Office formats), pdftotext
(PDF input) — as fixed argv without a shell, never overwriting (`name-2.ext`). Extension
tools declare the files they create or change (`ExtensionTool.touches`), so checkpoints and
`undo_changes` cover them without name lists. `study_file` extracts through the same
readers and, for long documents, map-reduces over the chunks most relevant to the question.

**Offline compute** (`kiwibro/compute.py`). `calculate` (a core tool) evaluates math
through an allow-listed AST walker (exact big integers, power/factorial guards),
converts units across 12 categories and does date arithmetic; `analyze_table`
profiles, filters, groups and aggregates CSV/TSV/JSON/JSONL by streaming rows with
bounded heaps and reservoir samples.

Extension tools (`calculate`, `analyze_table`, `doc_*`, `undo_changes`) are
`ExtensionTool` records in their modules, registered into the `tools.py` catalog by
`kiwibro/extensions.py`; they carry their own permission default and profile category.

Permission evaluation per tool: explicit `deny` → deny; `auto` → allow; `ask` →
create approval, emit `event: approval`, block the loop until the user resolves it
(`allow`, `always`, `deny`) or 120 s timeout (deny). `always` upgrades that tool to
`auto` persistently. Approvals are keyed by UUID and live in memory.

Safety:

- Workspace root resolved once; every path is `resolve()`d and must be inside the
  root (`is_relative_to`). Symlinks escape checks; reject `..` after resolution.
- `run_command`: `shell=True`, `cwd` inside workspace, timeout, limited env, blocks
  patterns such as `sudo`, `su `, `doas`, `rm -rf /`, `mkfs`, `shutdown`, `reboot`,
  `dd of=/dev/`, fork bombs, `:()`; output truncated. Honest limit: this is
  best-effort safety, not a container. It always asks first by default.
- Never auto-run commands or writes without an approval unless the user set
  `always` for that tool.

Web search: requests with a desktop Chrome UA, `Accept-Language: en`. Parse Brave
`div.snippet` (title/url/description) → Bing `li.b_algo h2 a` (+ decode `u=` base64
redirect) → DDG lite links → Wikipedia search API as last resort. Return ≤ 8
results as `[{title,url,snippet}]`. `fetch_page` returns `{title,url,text}` and the
agent cites URLs in prose.

---

## 8. Database & settings

SQLite at `data_dir/kiwibro.db`, `PRAGMA journal_mode=WAL`, `foreign_keys=ON`,
`busy_timeout=5000`. Schema (migrations forward-only, version table):

```sql
conversations(id TEXT PK, title TEXT, title_auto INTEGER DEFAULT 1, model TEXT,
  summary TEXT DEFAULT '', summary_through_seq INTEGER DEFAULT 0,
  created_at TEXT, updated_at TEXT)
messages(id TEXT PK, conversation_id TEXT FK ON DELETE CASCADE, seq INTEGER,
  role TEXT, content TEXT, reasoning TEXT, tool_calls TEXT, tool_call_id TEXT,
  tool_name TEXT, model TEXT, tokens INTEGER, meta TEXT, created_at TEXT,
  UNIQUE(conversation_id, seq))
memories(id TEXT PK, kind TEXT, content TEXT, normalized TEXT, tags TEXT,
  source_conversation_id TEXT, confidence REAL DEFAULT 1, pinned INTEGER DEFAULT 0,
  created_at TEXT, updated_at TEXT, last_used_at TEXT)
permissions(tool TEXT PK, mode TEXT)         -- auto | ask | deny
settings(key TEXT PK, value TEXT)            -- JSON values
usage(id INTEGER PK AUTOINCREMENT, conversation_id TEXT, model TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, created_at TEXT)
```

Default settings: model `null` (auto = first loaded chat model, else first chat
model), agent_enabled `true`, memory_enabled `true`, max_output_tokens `2048`,
temperature `0.6`, workspace `~/Kiwibro`, lmstudio_url `http://127.0.0.1:1234`,
permissions as per table above, user_name `""`. Settings API returns merged values.

---

## 9. Frontend contract

Single page, no build step, ES modules, no external resources. Files:
`templates/index.html`, `static/css/app.css`, `static/js/app.js` (+ modules),
`static/img/kiwibro.svg` + generated PNG icons + favicon.

Design language:

- GNOME/Ubuntu dark: base `#1b1b1f`, surfaces `#242429`/`#2c2c32`, text `#f4f2f0`,
  muted `#9a9aa3`, hairline `rgba(255,255,255,.07)`, radius 12–14 px. Accent: kiwi
  green `#9BCE6B` with a warmer secondary `#E95420` used sparingly (recording dot,
  danger, focus). Font stack `Ubuntu, 'Segoe UI', system-ui, sans-serif`; code
  `'Ubuntu Mono', monospace`.
- Layout: 56 px header bar (kiwi icon + wordmark left, centered model pill, right
  icon buttons: memory, settings). 264 px sidebar (New chat button, conversation
  list, tiny footer stats) collapsible. Chat column max-width ~760 px. Composer as
  a floating rounded field with agent toggle, send/stop, context ring.
- Empty state: large kiwi mark, "Kiwibro" wordmark, one-line subtitle, 3 quiet
  suggestion chips. Very low text overall.
- Messages: user bubbles right-aligned with accent tint; assistant plain text with
  markdown, code blocks (language label + copy button), tool activity as thin
  collapsible rows with icons and status dots, reasoning collapsed by default.
- Approval card inline in the transcript: tool icon + name, one-line summary,
  expandable details (command / file path + preview), risk badge, buttons
  `Allow`, `Always`, `Deny`. Keyboard: `a`/`d` maybe; Esc cancels approval.
- Sheets: memory sheet (search, kind chips, rows with pin/delete, add field),
  settings sheet (LM Studio status + model list, workspace path, permissions
  toggles, data location, "forget all"). Model popover with state dots, context
  size and tool-capable marker; selecting swaps instantly (toast).
- Streaming: `fetch` POST to `/api/chat`, read `response.body` via
  `ReadableStream` + `TextDecoder`, parse SSE blocks (`event:`/`data:`), render
  deltas through a markdown renderer that is safe (escape HTML first; no remote
  images; links `rel="noreferrer noopener"`). Abort via `AbortController` and
  `/api/chat/stop`.
- Persistence: active conversation id in `localStorage`; sidebar refresh after
  title events; no reload while streaming. Reduced-motion respected.
- Responsive: < 860 px sidebar becomes a slide-over drawer; header remains.
- No external fonts/CDNs/analytics. Everything must work offline except tools.

---

## 10. Packaging ("native app")

- `static/img/kiwibro.svg` — hand-crafted geometric kiwi bird mark (dark-friendly),
  plus PNGs 512/256/128/64, `favicon.ico`, `favicon.svg`.
- `scripts/run.sh` — check deps (Flask/requests/bs4), start `python3 app.py`,
  optionally open the browser. `--open` flag.
- `scripts/kiwibro.sh` — desktop entry point: start server in background if not
  healthy (nohup + log), wait for `/health`, then open Chrome `--app=` window with
  `--class=Kiwibro`; fall back to `xdg-open`.
- `scripts/kiwibro.desktop` — Name=Kiwibro, Comment=Local AI agent, Icon (absolute
  path substituted by installer), Exec=kiwibro.sh, Categories=Utility;Chat;AI.
- `scripts/install.sh` — installs the `.desktop` file and icons into
  `~/.local/share/{applications,icons}`, refreshed with `update-desktop-database`;
  prints one line on completion. Must not require root.
- `README.md` — hero screenshot, features, install, run, LM Studio setup, privacy,
  architecture summary, license.

---

## 11. Acceptance checklist (QA)

- [ ] `python3 -m pytest -q` green (all unit/API tests, no network, no LM Studio).
- [ ] Server boots clean, `/health` OK, UI loads with zero console errors.
- [ ] Live chat against LM Studio streams tokens; no raw error surfaces.
- [ ] Model switch mid-conversation works; past chats and memories unchanged.
- [ ] Tool flow: web search auto-runs; `write_file`/`run_command` ask permission,
      allow/always/deny all behave; denied tool does not execute.
- [ ] Memory survives restart, dedupes near-identical facts, appears in prompt,
      editable/deletable in UI.
- [ ] Context: long conversation compacts automatically (summary chip), no
      context-length error surfaced, works after compaction.
- [ ] Mobile/responsive pass; dark theme consistent; screenshots captured.
- [ ] README accurate; repo clean; `.desktop` launcher works; icon renders.
