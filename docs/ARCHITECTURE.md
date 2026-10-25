# Kiwibro — Architecture & Build Contract

Kiwibro is a local-first AI agent for Linux: a **Python Flask** backend serving a
**browser GUI** designed to feel like a native Ubuntu/GNOME application, talking to
**LM Studio** on `127.0.0.1:1234` (OpenAI-compatible API). No cloud, no accounts,
no external CDNs, no bundled JS frameworks.

Version 2.2.0 · MIT · Python 3.10+ · Flask 3

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
5. **Safe autonomy.** Every tool call is graded (safe → critical) and the
   `autonomy` setting decides what runs without a human: *smart* (default) runs
   safe and undoable work and asks before risky steps, *supervised* follows the
   per-tool permission list, *autonomous* never waits and refuses risky steps with
   a reason the model can act on. Critical actions are always refused. Commands
   and code run confined (bubblewrap + a systemd scope): read-only system,
   writable workspace, credentials hidden, memory/process caps.
6. **Native feel.** A GNOME-style window: header bar, model pill, sidebar, sheets,
   subtle motion, Ubuntu typography, dark theme, a hand-drawn kiwi mark. Installed
   with an app icon and `.desktop` launcher that opens a Chrome app window
   (browser GUI, no Electron).

---

## 2. Repository layout

```
kiwibro/
├── app.py                  # Flask entry point: create_app(), python3 app.py
├── kiwibro/                # Python package (backend), grouped by concern
│   ├── __init__.py         # application factory and service wiring
│   ├── api.py              # HTTP API + SSE endpoints (docs/API.md)
│   ├── config.py           # paths, defaults, settings values + validation
│   ├── db.py               # SQLite schema, forward migrations, CRUD
│   │   # model & inference
│   ├── llm.py              # LM Studio client: discovery, streaming, JSON helpers
│   ├── inference.py        # KV-stable prompt layout, thinking control, telemetry
│   ├── readiness.py        # load the selected model + prefill its prompt ahead of time
│   ├── capabilities.py     # model capability inference, smallest-capable pick
│   ├── routing.py          # temporary helper models (load → use → unload)
│   ├── media.py            # image/audio/video handling, no-model fallbacks
│   ├── delegate.py         # agent-facing delegation to other local models
│   │   # context & memory
│   ├── context.py          # token estimation, budgets, compaction
│   ├── memory.py           # durable memory: extraction, dedupe, retrieval
│   ├── terms.py            # key terms: stop words, stems, topic tags for matching
│   ├── sessions.py         # continuity across chats: earlier-chats note, gists
│   ├── directives.py       # standing-instruction ledger + end-of-turn self-check
│   ├── prompts.py          # prompt templates
│   │   # agent & tools
│   ├── agent.py            # agent loop, SSE queue, approvals, abort
│   ├── tools.py            # tool catalog, validation, dispatch
│   ├── extensions.py       # plug-in surface for tool modules outside tools.py
│   ├── toolcall.py         # tool-call rescue for small models
│   ├── codebase.py         # grep/glob/outline/patch/file management, snippets
│   ├── research.py         # parallel multi-search, batch fetch, evidence packs
│   ├── search.py           # web search providers + SSRF-guarded fetching
│   ├── browser.py          # headless browser and page inspection
│   ├── desktop.py          # system info, open, clipboard, notify, screenshot
│   ├── library.py          # on-disk document library (FTS5), output paging
│   ├── convert.py          # document conversion with installed converters
│   ├── compute.py          # exact math, units, dates, table analysis
│   ├── tasks.py            # task board for long-running workflows
│   ├── remote.py           # SSH remote hosts: run, upload, download
│   ├── workspaces.py       # per-conversation working folders
│   │   # safety
│   ├── safety.py           # risk grading, auto/ask/deny policy, confinement
│   ├── sandbox.py          # workspace paths, file tools, process runner
│   ├── guard.py            # content safety: secrets, injections, bad writes
│   └── checkpoints.py      # automatic file checkpoints + undo
├── templates/index.html    # single page shell
├── static/
│   ├── css/*.css
│   ├── js/*.js             # ES modules, no build step
│   └── img/                # kiwi logo SVG + PNG icons
├── scripts/                # run.sh, kiwibro.sh, install.sh, kiwibro.desktop
├── tests/                  # pytest unit + API tests, fake-LM e2e suite (tests/e2e)
├── docs/                   # architecture, API contract, memory, QA
├── requirements.txt        # Flask, requests, beautifulsoup4 (nothing else)
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
- the current LM Studio model id and the workspace path;
- the tools of the current profile (and their auto/ask permission split) plus one line
  naming the rest of the catalog, or an explicit "no tools" notice when agent mode is
  off or the model lacks tool support;
- the rolling conversation summary added by the context engine.

Everything in that system prompt is **stable across turns** (see §4.1): the local time,
retrieved memories, recalled episodes and the directive ledger change per turn, so they
travel in a context note on the newest user message instead
(`Agent._turn_context` → `ContextEngine.prepare(turn_context=…)`).

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

### 4.1 Local inference engine (`inference.py`, `routing.py`, `media.py`)

On a CPU-only laptop, prompt prefill dominates latency. Measured on a Ryzen 5 7530U
(LM Studio 0.4.26, `spark-x2.6-4b`, 4,052-token prompt): **124 s cold, 1.3 s** when
LM Studio reuses its KV cache for the prefix, and 135 s again when one changing line
(the clock) moved to the front. LM Studio keeps one cache per parallel slot (4 by
default); a background request in between did not evict a chat's warm prefix.

- **Prefix-stable prompts.** Message 0 (identity, profile tool set, summary) is
  byte-identical across turns *and* conversations; per-turn text rides on the newest
  user message. Tool results decay by user-turn age so old messages never re-render.
  Live through Kiwibro: a new chat's first token came after 2.4 s instead of 17.3 s.
- **Stable tool list.** Chat templates render the tool schemas ahead of all messages, so
  `auto` is one fixed set (`tools.AUTO_TOOLS`, about 20 tools sized for 1–4B models), never
  re-selected per message. Other catalog tools run when called by exact name.
- **Reasoning control** (setting `reasoning`: `auto`/`on`/`off`). LM Studio honours
  `reasoning_effort: "none"` for models whose `/api/v1/models` entry lists a reasoning
  toggle ("say hi" took 27.7 s with thinking and 2.1 s without). Templates render the
  switch near the top, so a turn keeps one decision for all its rounds and `auto` only
  changes it between turns while the prompt is ≤ 1,500 tokens. Non-streaming utility
  calls (summaries, memory, titles, helpers) never think.
- **Telemetry.** `done.inference` reports `ttft_ms`, `tokens_per_s`, a message-level
  estimate of reused prompt tokens (`PrefixTracker`) and whether the turn thought; the
  UI shows it as one quiet line under the reply. Time to first token includes waiting
  for a free LM Studio slot.
- **Model metadata.** `/api/v1/models` adds `key`, `display_name`, `arch`, `params`,
  `size_bytes`, `quant_bits`, reasoning options and the loaded instance's
  `loaded_config` (`context_length`, `parallel`, `flash_attention`) to every model record
  (`llm.merge_native_metadata`); `inference.model_profile()` turns any record into a
  size/tier/capability profile.
- **Capability routing.** `routing.run_helper(llm, need, messages, current_model=…)` keeps
  the main model when it can do the job, else leases the smallest capable installed model:
  free-RAM check (never evicts the main model), `POST /api/v1/models/load` with an 8K
  window, reference-counted sharing between concurrent calls, `POST /api/v1/models/unload`
  after the last one, JIT `ttl` as a crash backstop. Live: `spark-x2.6-4b` (no vision) →
  `qwen3.8-27b` loaded, answered, unloaded in 33 s. Images are downscaled to ≤1024 px
  first (`media.image_data_url`); videos become keyframes (`media.video_frames`); with no
  capable model, `media.media_facts` gives deterministic code-derived facts.
- **Model readiness** (`readiness.py`). Opening Kiwibro, picking a model, opening a chat
  or changing a prompt-shaping setting queues one background job. **Load:** a model that
  is not loaded is loaded explicitly with a 32K window and 4 parallel slots (16K, then 8K
  when LM Studio's memory guardrails refuse; no load API → JIT), and a loaded model
  below 16K is reloaded larger while no reply runs. A JIT load uses the model's own
  defaults, which can be a single slot where every background call evicts the chat's
  cache. **Warm:** `AgentManager.warm_prefix` renders exactly the next request's leading
  messages and tools (a chat's history as if a new message followed it, tool results
  already compact) and `LMStudioClient.prefill` sends them with `max_tokens: 1`. Measured
  (`lfm2.5-1.2b`, 4.7k-token prefix): first request 47 s cold vs 6 s warm; through
  Kiwibro a new chat's first token came after 7.3 s with 97 % prefix reuse. A turn that
  starts mid-job waits for it when the prefix is its own (status *Getting the model
  ready*) and cancels it otherwise: `prefill` shuts the socket down, because LM Studio
  sends nothing until the first token. Chats estimated above 24k tokens are not
  prefilled ahead. **Switching models** unloads the model Kiwibro loaded for the
  previous selection before loading the new one (two resident 4B models made this
  laptop swap); models the user loaded in LM Studio and helper leases are never
  touched. The header's model dot pulses amber while a job runs
  (`GET /api/models/ready`).
- **LM Link.** Models on linked devices appear in the local server's lists like local
  ones (no device field; a key present on two devices is listed twice in
  `/api/v1/models`, and the first entry's metadata is kept). They are selected, loaded and
  warmed through the same endpoints. Settings can point Kiwibro at another LM Studio
  server and hold an API token for servers with authentication on.
- **Bigger windows for long chats.** `POST /api/models/context {id, context_length}`
  reloads a loaded model with a larger context (`/api/models` lists `context_choices`);
  if LM Studio refuses for lack of memory the previous window is restored.
- **Attachments.** Files the user attaches (button, paste, drop) are stored under
  `<workspace>/attachments/<day>/` so every tool reaches them by path. Before the main
  model runs, `Agent._see_attachments` → `delegate.see_attachment` describes images and
  video keyframes with the main model if it can see, else with the smallest vision model
  (leased as above); audio and SVG get code-measured facts, and so do images when no
  vision model is installed. The result is stored in the user message's meta and rendered
  as one fixed note *before* the user's words (`context.to_llm_message`), so it is
  KV-stable and the user's question is what a small model reads last. Live: a test image
  sent to `spark-x2.6-4b` was described by `qwen3.8-27b` ("green triangle … ORDER 7781")
  and answered correctly.
- **Stop during prefill.** LM Studio sends response headers at once and nothing else until
  the first token. With an abort flag, `stream_chat` (and `prefill`, and `complete(abort=…)`,
  which streams) starts a watcher that shuts the socket down when the flag is set: Kiwibro
  returns at once and LM Studio drops the request at its next 2048-token batch (measured:
  a 75 s prefill stopped ~25 s after the abort).
- **Leaked reasoning.** With thinking off some small models think anyway and close a
  `</think>` LM Studio never saw opened. `LMStudioClient` moves that text into the
  reasoning channel mid-stream and the agent sends an SSE `rewrite` so the UI replaces it.
- **Models that cannot stop thinking.** `qwen3.8-27b` (the only installed vision model)
  reasons on every request whatever `reasoning_effort` or a `/no_think` prompt says, so
  helper calls budget tokens for reasoning plus the answer.

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
budget  = context - reserve(output) - 512 - tool_schema_tokens
reserve(output) = min(settings.max_output_tokens, max(512, context // 8))
```

The tool schemas render outside the messages (~3.5k tokens for the auto set), so
`prepare(..., reserve_tokens=…)` keeps them out of the message budget and compaction
starts before the window overflows. When the auto set would take more than 35 % of the
budget (a small window that cannot be enlarged), the minimal set is sent instead
(`AgentManager._tool_list`); other tools still run when called by name.

**Messages longer than the window.** A user message over ~40 % of the budget is stored
in the document library (`library.stash_text`); the user message's meta gets a fixed
`prompt_text` (head + tail + the handle to `doc_read`/`doc_search` the rest) that
`to_llm_message` sends instead, so the prompt stays KV-stable. The transcript, the
episodic index and memory extraction keep the full text. Messages up to 400,000
characters are accepted.

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
3. Keep the newest user message and the last assistant reply always. Compaction never
   summarises the request being worked on (a long tool turn would otherwise fold the
   user's own message into the summary), hard-trimming keeps it, and a mid-turn
   compaction (`extend` past the trigger) keeps the turn's note.
4. On LM Studio context-limit HTTP 400: summarise half of the remaining history and
   retry; after 2 failures truncate to system prompt + last 6 messages and retry.

The UI receives `event: context {used, budget, percent, compacted}` after context
work and on `meta`, rendered as a minimal ring/bar near the composer.

---

## 6. Memory

Durable, global (not per conversation), SQLite-backed.

Schema: `memories(id, kind, content, normalized, tags, source_conversation_id,
confidence, pinned, created_at, updated_at, last_used_at, use_count, origin)` with FTS5
index. Kinds: `fact`, `preference`, `project`, `person`, `decision`. `origin` (schema v4):
`you` (added or edited in the Memory sheet), `chat` (the `remember` tool), `learned`
(background extraction); NULL for older rows. The sheet labels each memory with its origin
and links back to the chat it came from.

- **Extraction**: after each completed assistant turn (after `done` is emitted), if
  enabled, the chat is queued for one background worker. It runs once no reply has
  streamed for `MEMORY_IDLE_S` (20 s) and no warm-up runs, one chat at a time, and a reply
  starting cancels it mid-request (the chat is re-queued; the watermark is untouched).
  On a CPU-only laptop an extraction beside the next reply slowed its first token from
  2.8 s to 8.4 s and stored a fact that reply had just changed. The worker asks the
  selected model for strict JSON operations
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
- **Explicit remember tool**: `remember(content, kind, action, old)` writes immediately.
  `action: save` (default) adds or merges; `update` replaces the memory `old` refers to
  (words from it or its id; ≥ 50 % of its key words must match) — "I moved to Ankara"
  rewrites "lives in Istanbul" instead of adding a contradiction; `forget` deletes the
  memory the content refers to (pinned memories are kept and the model is told so).
  Saves and updates must be grounded in what the user said in the chat
  (`memory.grounded`), and facts about the assistant itself are refused — small models
  otherwise store "Kiwibro is a local AI agent…".
- **The user's memories are protected**: extraction never deletes a `you` or pinned
  memory, and pruning at the cap never evicts them.
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

Permission evaluation per call goes through `safety.decide` (below): the result is
`auto` (run), `deny` (refused with reasons the model can act on) or `ask` → create
an approval, emit `event: approval` (with `level`/`reasons`), block the loop until
the user resolves it (`allow`, `always`, `deny`) or 120 s timeout (deny). `always`
upgrades that tool to `auto` persistently (critical actions stay refused).
Approvals are keyed by UUID and live in memory.

Staying on task (`directives.py`): a ledger derived from the user's messages keeps
the goal, the current request, standing constraints and choices ("never push
without asking", "use TypeScript") and what the user changed their mind about.
It rides in the per-turn note when it adds something, so small models keep
following instructions after compaction and topic switches; course changes
("actually…", "instead…") supersede what they overlap. After a turn with real
work (3+ write/exec actions, smart/autonomous) the model gets one transient
checklist from the ledger to verify its work before finishing.

Doing a whole request (`plan.py`): "tell me what X is, then search the web for Y,
and finally write a doc at ~/Desktop/notes/y.md" is split by code into ordered steps
(sequencing words, sentences, list items; what the user says before the first
instruction is context) and each step is classified: *answer*, *web*, *write*, *run*,
*read* or *other* (a constraint code cannot check). A write or read step gets its
target from the message: a path, a place ("on my desktop", from `user-dirs.dirs`) plus
a file name, or a place alone. No model call is involved.

- **Steps in view.** With several steps, or a place outside the workspace, the steps
  and the exact path to use ride in the per-turn note. The UI shows them as a
  checklist (SSE `plan`), ticked off as they finish.
- **Places the user named.** A file tool whose path points at a file the user named
  in this message runs in a workspace rooted at that file's folder, so the call is
  confined and graded there (a new file is *low*: no approval in smart mode). A named
  file may be written or edited; a named folder ("on my desktop") allows reading and
  new files only, never its existing files. A model that drops the folder
  (`raspberry.md`) is routed to the named file. Home itself, credentials, start-up
  files, system folders and Kiwibro's data are never targets. A path outside the
  workspace that the user did not name is refused at once, without an approval card
  for a call that could not run.
- **Progress on the tool result.** When a call completes a step (a search that found
  something, a write to the target), one line is appended to that round's last tool
  result: what is next, or that every tool step is done with the files checked on
  disk and only the parts of the reply not yet given. It is stored, so the next
  request's prefix matches what LM Studio has cached.
- **Verified finish.** When the model stops, code checks the plan: the target file
  exists and is not (almost) empty, web steps searched, each explain-step's topic
  appears in the reply. What is left becomes one transient nudge (at most two per
  turn); a plan that checks out ends the turn without the generic self-check round,
  which only runs for steps code cannot check. A turn that still ends with open steps
  says so in one honest line.
- **Research pacing.** A search nearly repeating one made earlier in the turn (Jaccard
  ≥ 0.6 on its words) is refused with a pointer to the results above. After three
  rounds of searching the result says to answer; after five, or one round after the
  plan's tool steps are done, searches are refused. Seen live: a 4B model searched
  six rounds of near-identical queries ("let me get the details, then produce the
  bulletin") for 30 minutes and never wrote the bulletin. A page that an earlier search
  of the turn already showed keeps its title and URL but not its snippet
  (`research.drop_seen_snippets`): every 500 prompt tokens cost ~15 s of prefill here.
- **Whole files in one call.** While a planned file is unwritten the reply cap rises
  to 6,144 tokens (within the window's free room), so thinking plus the file fit. A
  call cut off by the cap (`finish_reason: length`) is refused instead of run, since
  repaired JSON would save a truncated file.
- **Thinking budget.** A request that should act (a planned tool step is open, or tool
  results are in) may stream 4,000 characters of reasoning before it does; past that it
  is cut and retried once with thinking off and `prompts.ACT_NOW` naming the next step.
  Later rounds think again. Seen live: `spark-x2.6-4b` drafted the same file in its
  thinking for 20 minutes (29,500 characters, the call written out inside the
  reasoning) and never called `write_file`. 30 acting rounds measured a median of 440
  characters (p90 1,640, max 3,540). Switching thinking off re-prefills the prompt once for
  this model (47 s for 1.6k tokens, 190 s for a 6k-token chat; a `/no_think` soft switch is
  ignored), still far cheaper than an unbounded think. Live: cut after ~5 min, file
  written 6 min later, instead of 20+ min and no file.
- **Greetings need no tools.** For "hi" or "thanks" tool calls are refused with a
  one-line reason (small models otherwise read files or the task board to say hello).

Safety (`safety.py`, `sandbox.py`, `guard.py`, `checkpoints.py`):

- **Grading.** `safety.classify_command` reads a shell command the way bash does:
  quotes, comments, heredocs, `$(…)`/backticks/process substitution, pipelines,
  wrappers (`env`, `xargs`, `timeout`, `find -exec`, `bash -c`, `eval`), variables
  and symlinks set earlier on the same line, brace/`$IFS`/ANSI-C tricks, the
  scripts it runs (by `#!` line) and statically decodable payloads
  (`echo … | base64 -d | sh`). It knows where paths point: workspace, regenerable
  build output, the rest of `$HOME`, credentials, start-up files, Kiwibro's own
  data, system files, and grades `safe < low < medium < high < critical`.
  File, code, desktop and remote tools are graded by `safety.assess`.
- **Policy.** `safety.decide` combines the grade with the per-tool permission
  (`deny` always wins), the `autonomy` setting, graded per-turn circuit breakers
  (mass deletes/edits escalate one level per multiple of the limit), conversation
  taint from prompt injection, and `guard.inspect_call` (stricter wins). Critical
  is always denied; autonomous mode turns every "ask" into a deny with reasons.
  Denials reach the model as `Decision.denial_text()`.
- **Confinement.** `safety.confine` wraps commands, terminals and
  `run_python`/`run_javascript` in a transient systemd user scope
  (`MemoryMax`, `TasksMax`) and bubblewrap: read-only root, writable workspace
  (snippets: writable scratch only, no network), private `/tmp`, own PID
  namespace, `~/.ssh`/credential dirs and the session bus hidden. Every layer is
  feature-detected and self-tested; the `sandbox` setting picks auto/strict
  (no network)/off. A high-risk call the user explicitly approved (e.g.
  `git push`, which needs `~/.ssh`) runs unconfined.
- **Processes.** `sandbox.run_process` puts each command in its own process group
  and kills the whole group on timeout; stdin is EOF; one env builder drops
  code-injection variables (`LD_PRELOAD`, `GIT_EXTERNAL_DIFF`, …) and sets HOME to
  `safety.tool_home()` (`~/.cache/kiwibro/home`, writable in the sandbox) with
  `GIT_CONFIG_GLOBAL` pointing at the user's `~/.gitconfig`; a user-approved
  high-risk command gets the real HOME.
- **Failure loops.** After 3 failed tool calls in a row the result carries a
  coaching line; after 6 the agent refuses further calls this turn
  (`prompts.TOOLS_PAUSED`) so the model answers instead of guessing until the
  round limit. Refusing keeps the tool list, and so the KV-cached prefix, intact.
- **Degenerate output.** `inference.is_degenerate` watches the streamed tail of
  content and reasoning (near content-free filler, a unit repeated end to end, the
  same sentence line after line, and `loop_start`: a block of up to three lines
  repeated, as in "**The fix**" + the same bullet a dozen times, which it cuts at the
  third copy; markdown markers and case are ignored, fenced code is skipped, lists,
  FAQs and tables never match), and every 250 characters `inference.is_looping`
  checks for circling deliberation (≥0.15 five-word-phrase recurrence with ≥4 hedges
  per 100 words; code blocks ignored). The agent raises out of the stream (closing it
  stops LM Studio generating). A reply that repeated itself keeps its good start (up
  to the block's first copy) and is *continued*: the start goes back as an assistant
  message with `prompts.DEGENERATE_CONTINUE` and anti-repetition sampling
  (`repeat_penalty` 1.15, `frequency_penalty` 0.3) for that request only; the stored
  reply is the start plus the continuation, as the user saw it. A circling reply is
  cleared and redone with `prompts.DEGENERATE_NUDGE`. A second collapse ends the round
  with what is good (or `DEGENERATE_GAVE_UP`). Nothing degenerate is stored.
- **Paths.** Every file-tool path is resolved and must stay inside the workspace;
  a conversation can bind its own folder (`workspaces.py`, schema v3), and the
  model may switch folders on its own only to a folder the user's message names.
- **Remote.** `remote.py` uses the system ssh/scp in BatchMode; hosts from
  `~/.ssh/config` or `remote_hosts` are trusted, others need approval; commands are
  graded for a server (`sudo` high, wiping `/` or `~` critical).

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
  extracted_through_seq INTEGER DEFAULT 0 /* v2 */,
  created_at TEXT, updated_at TEXT, workspace TEXT /* v3 */,
  private INTEGER DEFAULT 0 /* v5: never remembered, never shown to other chats */)
messages(id TEXT PK, conversation_id TEXT FK ON DELETE CASCADE, seq INTEGER,
  role TEXT, content TEXT, reasoning TEXT, tool_calls TEXT, tool_call_id TEXT,
  tool_name TEXT, model TEXT, tokens INTEGER, meta TEXT, created_at TEXT,
  UNIQUE(conversation_id, seq))
memories(id TEXT PK, kind TEXT, content TEXT, normalized TEXT, tags TEXT,
  source_conversation_id TEXT, confidence REAL DEFAULT 1, pinned INTEGER DEFAULT 0,
  created_at TEXT, updated_at TEXT, last_used_at TEXT,
  use_count INTEGER DEFAULT 0 /* v2 */, origin TEXT /* v4: you | chat | learned */)
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
  `Allow`, `Always` (Manual mode) or `Allow in this chat` (automatic modes), `Deny`.
- Sheets: memory sheet (search, kind chips, rows with pin/edit/delete and an origin
  label — *Added by you*, *Saved in a chat*, *Learned* — that links to its chat, add
  field), settings sheet (LM Studio status, model list, server address and API token,
  permissions, behaviour, workspace path, data location, "forget all"). Permissions is
  one Automatic/Manual switch: Automatic shows only what to do with risky steps (ask
  me / refuse them); the per-tool Auto/Ask/Deny list appears in Manual only. Model
  popover with state dots, context size and tool-capable marker; selecting swaps
  instantly (toast) and the header dot pulses amber while the model loads and warms.
- Rich replies (`markdown.js` + `richblocks.js`): code is highlighted by a small
  escape-first tokenizer; `diff` blocks colour added/removed lines; `csv`/`tsv` blocks
  render as tables with the source folded below; complete `html` and `svg` blocks
  render as live previews (final render only, never while streaming) with Preview/Code
  tabs. SVG goes through an `<img>` data URI; HTML through an iframe with an opaque
  origin and a CSP that blocks every network request, with refresh tags removed and
  link clicks inert. Scripts stay paused until the user presses *Run scripts*: a page
  can still navigate its own frame, and a prompt-injected page could ask the model for
  one that carries chat text out in a URL.
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
