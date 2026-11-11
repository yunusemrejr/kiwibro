# Kiwibro — Architecture & Build Contract

Kiwibro is a local-first AI agent for Linux: a **Python Flask** backend serving a
**browser GUI** designed to feel like a native Ubuntu/GNOME application. The models run on
the same machine through **llama.cpp** (`llama-server`, one process per loaded model), or
through a llama.cpp-compatible server that the user points Kiwibro at. No cloud, no
accounts, no external CDNs, no bundled JS frameworks.

Version 3.0.0 · MIT · Python 3.10+ · Flask 3

---

## 1. Product principles

1. **Local.** Everything runs on the machine, the models included. UI assets are local.
   Web access only through the explicit web tools.
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
├── app.py                    # entry point: starts the Flask server (python3 app.py)
├── kiwibro/                  # Python package (backend), grouped by concern
│   ├── __init__.py           # Kiwibro application factory and service wiring
│   ├── api.py                # HTTP API and SSE endpoints, implementing docs/API.md
│   ├── api_models.py         # routes for the model manager: library, downloads, builds
│   ├── config.py             # paths, constants, settings defaults and validation
│   ├── db.py                 # SQLite persistence: schema, forward migrations, CRUD helpers
│   ├── selfupdate.py         # keeps a running Kiwibro on the newest code
│   │   # engine & models
│   ├── engine.py             # the engine: one llama-server child per loaded model
│   ├── runtime.py            # Kiwibro's own llama.cpp build, from the GitHub releases
│   ├── hub.py                # Hugging Face search, resumable downloads, disk cleanup
│   ├── gguf.py               # reads the metadata header of a GGUF file
│   ├── llm.py                # model client: discovery, streaming chat, JSON helpers
│   ├── inference.py          # prompt layout for the KV cache, thinking control, telemetry
│   ├── readiness.py          # loads the selected model and prefills its prompt ahead of time
│   ├── routing.py            # temporary helper models, leased for one job at a time
│   ├── capabilities.py       # model capability inference, smallest-capable pick
│   ├── adapters.py           # task adapters: LoRA files switched per request by task
│   ├── media.py              # image, audio and video handling, no-model fallbacks
│   ├── delegate.py           # agent-facing delegation to other installed local models
│   │   # Kiwi Sense & steering
│   ├── sense/                # Kiwi Sense: a small reading model that watches the model
│   │   ├── __init__.py       # the Sense front door: embed, similarity, probe, repeat, heads
│   │   ├── helpers.py        # the five helper heads: language, message, words, shift, relate
│   │   ├── blob.py           # the data file: named sections behind a table of contents
│   │   ├── pyengine.py       # the Sense core in plain Python (twin of core.c)
│   │   ├── wasmengine.py     # the Sense core run as WebAssembly on the server (wasmtime)
│   │   ├── core.c            # the Sense core, compiled to WebAssembly
│   │   └── build.sh          # compiles core.c to static/sense/sense.wasm
│   ├── steer.py              # steering: finds a wrong reply and writes the correction
│   ├── readings.py           # the helpers' readings put to work: English only, search, memory, dates
│   │   # context & memory
│   ├── context.py            # token estimation, budgets, compaction, recall of earlier messages
│   ├── memory.py             # durable memory: extraction, dedupe, retrieval, user CRUD
│   ├── notes.py              # notes: longer pages (title, markdown body, tags)
│   ├── terms.py              # key terms: what a message is about, for matching
│   ├── sessions.py           # continuity across chats: what other conversations covered
│   ├── directives.py         # standing-instruction ledger and end-of-turn self-check
│   ├── autofacts.py          # deterministic facts about the user from one message
│   ├── guidance.py           # situational guidance for this turn, chosen by code
│   ├── prompts.py            # prompt templates
│   │   # agent & tools
│   ├── agent.py              # agent loop: one worker per turn, SSE queue, approvals, abort
│   ├── tools.py              # tool catalog, argument validation, permissions, dispatch
│   ├── extensions.py         # plug-in surface for tool modules outside tools.py
│   ├── toolcall.py           # tool-call rescue for small local models
│   ├── plan.py               # turn plan: splits a multi-step request, verifies the finish
│   ├── codebase.py           # glob, outlines, atomic multi-edits, file management, git
│   ├── research.py           # parallel web research: multi-search, batch fetch, evidence
│   ├── search.py             # web search providers and SSRF-guarded page fetching
│   ├── browser.py            # headless browser and page inspection
│   ├── desktop.py            # system info, open, clipboard, notify, screenshot
│   ├── library.py            # on-disk document library (FTS5), output paging
│   ├── convert.py            # document conversion with the converters on this computer
│   ├── compute.py            # exact math, units, dates, table analysis
│   ├── tasks.py              # task board for long-running workflows
│   ├── remote.py             # SSH remote hosts: run, upload, download
│   ├── workspaces.py         # per-conversation working folders
│   │   # safety
│   ├── safety.py             # risk grading, auto/ask/deny policy, confinement
│   ├── sandbox.py            # workspace paths, file tools, process runner, terminals
│   ├── guard.py              # content safety: secrets, injections, bad writes
│   └── checkpoints.py        # automatic file checkpoints and undo
├── templates/index.html      # single page shell
├── static/
│   ├── css/                  # app.css and one stylesheet per panel or feature
│   │   ├── app.css           # base layout, tokens, header, sidebar, composer
│   │   ├── attachments.css   # composer attachments
│   │   ├── how.css           # How it works panel
│   │   ├── inference.css     # thinking and speed line, context ring
│   │   ├── markdown.css      # rendered replies, code, tables, maths
│   │   ├── models.css        # Models panel
│   │   ├── notes.css         # Notes panel
│   │   └── shell.css         # sheets, popovers, toasts, responsive shell
│   ├── js/                   # ES modules, no build step
│   │   ├── api.js            # API client (real server, or the mock in ?mock=1)
│   │   ├── app.js            # controller: bootstrap, conversations, stream, sidebar, shortcuts
│   │   ├── attachments.js    # composer attachments: pick, paste, drop, upload
│   │   ├── chat.js           # transcript: messages, streaming, tool activity, approvals
│   │   ├── export.js         # chat export as Markdown (pure functions)
│   │   ├── how.js            # How it works: eight small animated figures
│   │   ├── inference.js      # thinking control, reply speed, context window size
│   │   ├── markdown.js       # escape-first Markdown renderer for chat replies
│   │   ├── mathml.js         # LaTeX maths to MathML, without dependencies
│   │   ├── mock.js           # development fake backend (?mock=1)
│   │   ├── mock_models.js    # fake model library and downloads for ?mock=1
│   │   ├── mock_notes.js     # fake notes for ?mock=1
│   │   ├── models.js         # Models panel: installed models, engine, Hugging Face downloads
│   │   ├── notes.js          # Notes panel: list and editor, autosave
│   │   ├── richblocks.js     # live previews of model-written HTML and SVG
│   │   ├── sense.js          # Kiwi Sense in the browser (WebAssembly core)
│   │   ├── sheets.js         # Memory and Settings sheets
│   │   └── ui.js             # DOM helpers, icons, popovers, sheets, toasts
│   └── img/                  # kiwi logo SVG, favicon and PNG icons
│   └── sense/                # sense.bin (data file) and sense.wasm (committed build)
├── scripts/                  # run.sh, kiwibro.sh, install.sh, python.sh, desktop entry
│   └── sense/                # builds the Kiwi Sense data file and the behaviour probe
├── tests/                    # pytest unit and API tests, fake-server e2e suite (tests/e2e)
├── docs/                     # ARCHITECTURE.md, API.md, MEMORY.md, QA.md, screenshots/
├── requirements.txt          # Flask, requests, beautifulsoup4
├── CHANGELOG.md              # release notes
├── COLLAB.md                 # notes for people working on the tree together
├── LICENSE                   # MIT
├── .github/                  # CI workflows
└── README.md
```

Data lives **outside** the repo, in `~/.local/share/kiwibro` (override `KIWIBRO_DATA_DIR`):

- `kiwibro.db`: the SQLite database.
- `models/`: models Kiwibro downloads (`models/author/name/file.gguf`).
- `runtime/`: llama.cpp builds Kiwibro installs.
- `engine/`: pid files and one log per model server (`<model>.log`).
- `adapters/`: task adapters Kiwibro keeps, and `adapter_health.json`.
- `checkpoints/`: file snapshots for undo. `library.db`: the document library.

The workspace is `~/Kiwibro` (override in settings). Logs: `kiwibro.log` in the data folder.

---

## 3. Runtime

- `python3 app.py` starts Flask on `127.0.0.1:8777` (override `KIWIBRO_HOST`,
  `KIWIBRO_PORT`). Threaded server, no debug/reloader by default.
- Single process. In-memory state: active turns, pending approvals, abort flags,
  search cache, the running llama.cpp servers. Everything durable is in SQLite (WAL mode).
- One turn per conversation at a time (`409` if already running).
- Bind loopback only. Local app, no login. Every request whose `Host` header is not
  `localhost`, `127.0.0.1` or `::1` gets `403 forbidden_host`, which keeps pages on other
  hosts (DNS rebinding) away from the API.
- `/health` reports `started_at` (Unix seconds) and `code_version`. The desktop launcher
  (`scripts/kiwibro.sh`) compares the running `code_version` with the repository's and, when
  they differ, restarts the server unless a reply is running (then it notifies and waits).
- Self-update (`kiwibro/selfupdate.py`, `KIWIBRO_AUTO_UPDATE`, on unless set to `0`): a running
  server re-executes itself on changed code once that code has been unchanged for 20 s and
  imports cleanly, and no reply is running. Open windows reload when `code_version` changes.

### Prompt identity

Every chat request builds a system prompt (`kiwibro/prompts.py` +
`Agent._system_prompt`) that tells the model exactly what it is and what it can do:

- identity: Kiwibro, a local AI agent on the user's Linux machine, running its language
  model on this computer (never a cloud service);
- an engine block: the id of the model in use and that it runs on this computer;
- a workspace block: the workspace folder and how paths are resolved;
- the tools of the current profile (and their auto/ask permission split) plus one line
  naming the rest of the catalog, or an explicit "no tools" notice when agent mode is
  off or the model lacks tool support;
- the rolling conversation summary added by the context engine.

Everything in that system prompt is **stable across turns** (see §4.1): the local time,
retrieved memories, recalled episodes, notes and the directive ledger change per turn, so
they travel in a context note on the newest user message instead
(`Agent._turn_context` → `ContextEngine.prepare(turn_context=…)`).

Utility prompts (titles, rolling summaries, memory extraction) also identify
themselves as Kiwibro's, so even background calls stay on-brand.

---

## 4. Engine

Kiwibro runs models with llama.cpp. Every model request goes through one client,
`llm.LLMClient`, which speaks the OpenAI-compatible routes of llama.cpp's server
(`/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/props`). The client does not
know which server answers; it asks `engine_mode()`, which is one of two modes:

- **local** (the default): Kiwibro finds a llama.cpp build, starts one `llama-server` for
  each model it needs, and stops it again. The servers listen on `127.0.0.1` on a free port
  with a random API key. Settings: `direct_runtime`, `direct_models_dir`, `direct_threads`,
  `direct_gpu`.
- **attached**: `KIWIBRO_SERVER_URL` names a llama.cpp-compatible server that is already
  running (on this machine or another one), and `KIWIBRO_SERVER_KEY` is its bearer key if it
  needs one. Kiwibro does not start, stop or load anything; the model list is the server's
  `/v1/models`. Model management (library, downloads, builds, context changes) is not
  available in this mode and answers `409 attached`.

Failures are reported to the UI as a state, never as a traceback: when no build is
installed, the Models panel says that Kiwibro's engine is not installed yet and offers
*Install engine*; when an attached server does not answer, the header shows it offline.

### 4.1 Models

Model files are GGUF files on disk. `engine.model_sources()` lists the folders to scan, in
priority order (the first folder wins when two hold the same id):

| source | folder | deletable from Kiwibro |
|---|---|---|
| `folder` | the folder in Settings (`direct_models_dir`) or `KIWIBRO_MODELS_DIR` | no |
| `kiwibro` | Kiwibro's own `<data>/models` (where downloads go) | yes |
| `lmstudio` | LM Studio's downloads and `models/` folders (`~/.lmstudio`, and the Flatpak install's), read-only | no |
| `shared` | `~/.cache/lm-studio/models` and `~/.cache/llama.cpp` (`-hf` downloads), read-only | no |

`KIWIBRO_OWN_MODELS_ONLY` leaves out the last two sources. A model's `deletable` flag is
true only for `kiwibro`. Other folders are read, never written.

- **Ids** follow the publisher and repository: `publisher/repo` lower-cased, without `-GGUF`,
  with `@quant` appended when one folder holds several quantisations. Split files
  (`-00001-of-00003`) are listed once; an `mmproj` file in a folder makes its model a vision
  model; embedding models are marked and kept out of the chat model list.
- **Reading a file** (`gguf.py`) takes only the header's key/values and skips the tokenizer
  arrays by seeking: architecture, trained context length, size label, file type (which gives
  the quantisation) and the chat template. A template that mentions tools declares tool use;
  one with a `<think>` block thinks, and one with `enable_thinking` or `reasoning_effort`
  can switch thinking off.
- **Records.** `DirectEngine.raw_entries()` and `native_entries()` describe each file in the
  shape `llm.normalize_model` and `llm.merge_native_metadata` read, so the model list has
  one normaliser for both modes (see §4.1 for the fields).
- **Runtime discovery** (`engine.find_runtimes`), best first:
  1. a custom build: Settings `direct_runtime`, or `KIWIBRO_LLAMA_SERVER`;
  2. Kiwibro's own installs under `<data>/runtime`, the one `current.json` names first;
  3. `llama-server` on `PATH`;
  4. builds another app keeps on this computer: LM Studio's bundled llama.cpp backends
     (`extensions/backends/llama.cpp-*`), found by the file layout and not used when
     `KIWIBRO_OWN_RUNTIME_ONLY` is set.

  Among the candidates, the GPU order follows `direct_gpu`. With `auto` (the default) builds
  whose `llama-server --list-devices` shows a hardware GPU come first (Vulkan, ROCm, Metal;
  CUDA only with an NVIDIA device present; software renderers such as llvmpipe are ignored).
  With `off`, CPU builds come first. Each build is started with its own folder on
  `LD_LIBRARY_PATH`, and its `--help` is read once so flags an older build lacks are left
  out. A runtime that exits while loading is retried without speculative decoding, then the
  next runtime is tried (four at most).

- **Catalog.** The list is rescanned at most every 30 s, or on `?refresh=1`
  (`GET /api/models`).

### 4.2 Engine builds (`runtime.py`)

Kiwibro installs its own build when asked (`POST /api/runtime/install`, or *Install engine*
in Models). `kind` is `vulkan`, `cpu` or `auto`; `auto` takes the Vulkan build when a Vulkan
loader and a render node are present (`runtime.wants_vulkan`), otherwise the CPU build.

- The release comes from `ggml-org/llama.cpp` on GitHub (`KIWIBRO_RUNTIME_API` overrides the
  API address). An asset is used only when GitHub publishes a `sha256` digest for it.
- The archive is downloaded to a new folder and checked against that digest. Its members are
  vetted before anything is unpacked (no links out of the folder, no unexpected paths), the
  unpacked size is capped at 4 GiB, and free disk space must be four times the archive size.
- The result is `<data>/runtime/<tag>-<kind>/`, and `current.json` names the build in use.
  `POST /api/runtime/use` switches the build, `DELETE /api/runtime/<name>` deletes one unless a
  model runs on it.
- The installer states are `idle`, `checking`, `downloading`, `verifying`, `unpacking`,
  `done`, `error` and `cancelled`. A download restarts when it is interrupted; only model
  downloads resume (§4.3). `POST /api/runtime/cancel` stops an install at its next chunk.
- Unsupported machines (the architecture has no build) get the `unsupported` error.

### 4.3 Downloads (`hub.py`)

- **Search** (`GET /api/hub/search`) asks Hugging Face for GGUF repositories tagged
  `conversational`, most downloaded first, and leaves out private and gated ones.
  `KIWIBRO_HF_URL` overrides the address.
- **Files** (`GET /api/hub/files`) come with each file's size and sha256 from the repository
  tree; the client never supplies them. Each file gets a `fit` value (below).
- **Download.** A job writes `<name>.part` in the model's folder
  (`models/author/name/file.gguf`), continues an interrupted file with an HTTP `Range`
  request, and checks the size and sha256 before renaming the file into place. A partial file
  is never listed as a model. Cancelled jobs keep their partial bytes.
- **Disk.** A download needs its size plus 512 MiB (`DISK_RESERVE`) free; `disk_usage` reports
  the models, the partial files and the free space.
- **Fit.** `hub.fit(size, ram)` answers `good` when the file plus overhead (1.25 × size + 1 GiB)
  fits in 60 % of RAM, `tight` up to 85 %, and `too_big` otherwise.

### 4.4 Model servers (`engine.py`)

- **Lifecycle.** A request acquires its model's server, starting it on first use, and
  releases it when done. Eviction and unloading stop only servers with no request in
  flight. At most `MAX_RESIDENT` (2) servers run: the chat model and one helper; a third
  stops the least recently used idle one, or one that no longer fits in free memory.
  A start waits up to `START_TIMEOUT_S` (300 s) for the server to answer.
- **Process ownership.** Children are spawned from one long-lived thread with
  `PR_SET_PDEATHSIG`, so they die with Kiwibro even through its self-update `execv`. Pid files
  in `<data>/engine/` let the next start stop any left behind. Logs: `<data>/engine/<model>.log`.
- **Idle sleep.** A server frees its memory after `IDLE_SLEEP_S` (1800 s) without a request
  (`--sleep-idle-seconds`); `KIWIBRO_IDLE_SLEEP_S` changes it, `0` keeps servers resident.
  `POST /api/engine/stop` stops them all at once.
- **Launch.** `server_args()` builds the command line:
  - `--ctx-size` (32K by default, capped by the trained context), `--parallel 4 --kv-unified`
    (one KV pool for the chat slot and background slots);
  - `--threads` = `--threads-batch` = the physical cores (`direct_threads` overrides);
    measured: 12 batch threads on this 6-core SMT CPU prefilled at 27 tok/s vs 37 with 6;
  - `--flash-attn auto`, `--jinja`, `--reasoning-format deepseek`, `--cache-reuse 256`,
    `--cache-ram` (a sixth of free memory, 1–4 GiB), `--spec-type ngram-mod` (no draft model;
    measured 4.7 → 6.8 tok/s on a reply that copies its input, unchanged on free text),
    `--no-webui`, `--metrics`, `--api-key`, `127.0.0.1` on a free port;
  - `--ctx-checkpoints 8`: sliding-window and hybrid models (spark-x2.6, qwen3.8, lfm2) rewind
    only to saved checkpoints; llama.cpp's default 32 per slot grew host memory ~110 MB per turn
    up to ~1.7 GB per slot (54 MB per checkpoint for spark-x2.6); with 4 a turn after six tool
    rounds re-read 445 more tokens (~5 s on the iGPU) and memory stayed flat;
  - `--no-cache-idle-slots` (more than one slot): by default llama.cpp copies idle slots out to
    the RAM prompt cache and clears them on every new request, so a memory or title call between
    two chat turns moved the chat's state out and back (with a small cache it was lost: a full
    3.5k-token re-read). Kept in place, when the KV pool is full llama.cpp purges the least
    recently used idle slot, which was the background one in testing.
  - Adapters (§4.5) add `--lora <file>` and `--lora-init-without-apply`.
- **GPU offload.** On a GPU build with acceleration on, `--gpu-layers auto` puts every layer that
  fits on the GPU (`999` on builds without `--fit`). With acceleration off (`direct_gpu: off`),
  a GPU build gets `--device none`.
- **Measurements on the Ryzen 5 7530U with its Radeon iGPU** (Vulkan, spark-x2.6-4b IQ2_M):
  prompt processing ~100 tok/s vs 54 on six CPU cores, decode ~10 vs 7.4 tok/s, 0.04 vs
  0.81 CPU-seconds per generated token; qwen3.8-2b 249 vs 112 tok/s prefill. KV quantisation
  (q8_0) and MTP drafting were measured and left off (slower on this GPU / no gain).

### 4.5 Task adapters (`adapters.py`)

A LoRA adapter is a small GGUF (`general.type = adapter`) trained on top of one base model.
The engine loads the adapters that fit a model when it starts its server (same architecture and
base, at most `MAX_ADAPTERS` = 6, held at scale 0). Each request then names the scale of every
adapter (llama-server's per-request `lora` field), so the server switches the task on and off
without reloading.

- **Which task.** The task comes from the message: `code`, `math`, `writing`, `summarize`,
  `translate` or `research` (`adapters.task_of`). A follow-up without a signal of its own keeps
  the task of the message before it, so the adapter, and with it the server's KV cache, only
  changes when the task really does.
- **What an adapter is for.** A sidecar `<file>.json` (`{"tasks": ["code"], "scale": 0.8}`;
  `"*"` means always on) or, without one, words in its file name and `general.name`. An adapter
  whose purpose is unknown is loaded but never switched on.
- **Where adapters are found** (`adapter_dirs`): `KIWIBRO_ADAPTERS_DIR`, Kiwibro's own
  `<data>/adapters`, then the model folders.
- **Self-adjustment.** When a reply goes wrong with an adapter on and comes out right with it
  off, that counts against the adapter (`adapter_health.json`). After `TROUBLE_HALF` (2) such
  replies it runs at half its scale, after `TROUBLE_OFF` (4) it stays off, and the count ages
  out after `TROUBLE_DAYS` (30). The turn's `done.inference.adapters_turned_down` lists the
  adapters judged this way. In the Models panel an adapter can be reset to full strength or
  deleted from Kiwibro's own folder (`/api/library/adapters`).

### 4.6 Requests and thinking

- **Payload** (`engine.adapt_payload`): `cache_prompt` is set; the keys llama.cpp has no use
  for (`ttl`, `draft_model`) are dropped; `reasoning_effort: "none"` turns thinking off
  (`enable_thinking: false`) and any other effort word turns it on (`enable_thinking: true`),
  because templates disagree on effort vocabulary (spark-x2.6 raises on "medium");
  `parallel_tool_calls: true` is set when tools are sent (llama.cpp otherwise allows one call
  per reply, and switches this off itself for templates that cannot express several).
- **Thinking budgets** (direct engine, thinking model): the agent keeps the template in thinking
  mode on every turn (`chat_template_kwargs.enable_thinking: true`, `TEMPLATE_THINKING`) and
  expresses each decision as `thinking_budget_tokens`: 0 for no thinking, 1000 for rounds that
  must act, 2500 for a turn's first round. A `reasoning_budget_message` (a short sentence in the
  model's voice, written into the thought before llama.cpp closes it) goes with a non-zero
  budget only, because with 0 it leaked a stray `</think>`. Flipping the template switch re-reads
  the whole prompt (measured: 466 of 466 tokens re-read, against 1 with a budget of 0), so
  thinking decisions never change the prompt and `"auto"` may switch per turn. If a runtime
  ignores budgets, a thought longer than 8 characters per budget token (250-token minimum) is cut
  off and retried once with thinking off in the template. Without budgets (an attached server
  that does not take them), an acting round is cut off at a flat 4,000 characters instead.
- **Telemetry**: llama.cpp's per-request `timings` (`cache_n`, `prompt_n`, `prompt_ms`,
  `predicted_n`, `predicted_per_second`) are kept on `ChatResult.timings`; with them the reply's
  reuse percentage, prefill speed and decode speed are exact instead of estimated
  (`inference.server_reuse`).

### 4.7 Local inference (`inference.py`, `routing.py`, `media.py`, `readiness.py`)

On a CPU-only laptop, prompt prefill dominates latency. Measured through the model server
(measured through LM Studio 0.4.26, which runs the same llama.cpp; read the figures as orders of magnitude) on a Ryzen 5 7530U with a 4,052-token prompt
(`spark-x2.6-4b`): **124 s cold, 1.3 s** when the server reuses its KV cache for the prefix, and
135 s again when one changing line (the clock) moved to the front. The server keeps one cache per
parallel slot (4 by default); a background request in between did not evict a chat's warm prefix.

- **Prefix-stable prompts.** Message 0 (identity, profile tool set, summary) is byte-identical
  across turns *and* conversations; per-turn text rides on the newest user message. Tool results
  decay by user-turn age so old messages never re-render. Measured through the model server: a
  new chat's first token came after 2.4 s instead of 17.3 s.
- **Stable tool list.** Chat templates render the tool schemas ahead of all messages, so `auto`
  is one fixed set (`tools.AUTO_TOOLS`, 22 tools sized for 1–4B models), never re-selected per
  message. Other catalog tools run when called by exact name.
- **Reasoning control** (setting `reasoning`: `auto`/`on`/`off`). For a model whose template has
  a thinking switch, `reasoning_effort: "none"` turns thinking off (§4.6). Templates render the
  switch near the top, so a turn keeps one decision for all its rounds and `auto` only changes it
  between turns while the prompt is ≤ 1,500 tokens. Non-streaming utility calls (summaries,
  memory, titles, helpers) never think.
- **Telemetry.** `done.inference` reports `ttft_ms`, `tokens_per_s`, the reused prompt tokens
  (`PrefixTracker`, an estimate; the exact figures come from llama.cpp's timings where present)
  and whether the turn thought; the UI shows it as one quiet line under the reply. Time to first
  token includes waiting for a free slot.
- **Model metadata.** Every model record can carry `key`, `display_name`, `arch`, `params`,
  `size_bytes`, `quant_bits`, the reasoning options and the loaded instance's `loaded_config`
  (`context_length`, `parallel`, `flash_attention`) (`llm.merge_native_metadata`).
  `inference.model_profile()` turns any record into a size/tier/capability profile.
- **Capability routing** (`routing.py`). `routing.run_helper(llm, need, messages, current_model=…)`
  keeps the main model when it can do the job, else leases the smallest capable installed model.
  A lease checks free RAM against the model's size (never evicts the main model), loads it with
  an 8K window (`HELPER_CONTEXT`), shares it between concurrent calls by reference count, and
  unloads it after the last one. Images are downscaled to ≤1024 px first (`media.image_data_url`);
  videos become keyframes (`media.video_frames`); with no capable model, `media.media_facts` gives
  deterministic code-derived facts. Measured through the model server: `spark-x2.6-4b` (no
  vision) → `qwen3.8-27b` loaded, answered, unloaded in 33 s.
- **Model readiness** (`readiness.py`). Opening Kiwibro, picking a model, opening a chat or
  changing a prompt-shaping setting queues one background job. **Load:** the selected model is
  loaded explicitly with a 32K window and 4 parallel slots; when the server's memory check refuses
  it, 16K and then 8K are tried, and a loaded model below 16K is reloaded larger while no reply
  runs. **Warm:** `AgentManager.warm_prefix` renders exactly the next request's leading messages
  and tools (a chat's history as if a new message followed it, tool results already compact) and
  `LLMClient.prefill` sends them with `max_tokens: 1`. Measured through the model server
  (`lfm2.5-1.2b`, 4.7k-token prefix): first request 47 s cold vs 6 s warm; through Kiwibro a new
  chat's first token came after 7.3 s with 97 % prefix reuse. A turn that starts mid-job waits
  for it when the prefix is its own (status *Getting the model ready*) and cancels it otherwise:
  `prefill` shuts the socket down, because the server sends nothing until the first token. A
  chat's history is warmed ahead of time only while that takes at most 180 s at the model's
  measured prefill speed (`readiness.warm_limit`: between 6,000 and 24,000 tokens), because
  warming an opened chat is wasted when the chat is only read. **Switching models** unloads the
  model Kiwibro loaded for the previous selection before loading the new one (two resident 4B
  models made this laptop swap); models loaded outside Kiwibro and helper leases are never
  touched. The header's model dot pulses amber while a job runs (`GET /api/models/ready`).
- **Bigger windows for long chats.** `POST /api/models/context {id, context_length}` reloads a
  loaded model with a larger window (`/api/models` lists `context_choices`); if the load is
  refused for lack of memory, the previous window is restored.
- **Attachments.** Files the user attaches (button, paste, drop) are stored under
  `<workspace>/attachments/<day>/` so every tool reaches them by path. Before the main model
  runs, `Agent._see_attachments` → `delegate.see_attachment` describes images and video keyframes
  with the main model if it can see, else with the smallest vision model (leased as above); audio
  and SVG get code-measured facts, and so do images when no vision model is installed. The result
  is stored in the user message's meta and rendered as one fixed note *before* the user's words
  (`context.to_llm_message`), so it is KV-stable and the user's question is what a small model
  reads last. Measured: a test image sent to `spark-x2.6-4b` was described by `qwen3.8-27b`
  ("green triangle … ORDER 7781") and answered correctly.
- **Stop during prefill.** The server may send nothing until the first token. With an abort
  flag, `stream_chat` (and `prefill`, and `complete(abort=…)`, which streams) starts a watcher
  that shuts the socket down when the flag is set: Kiwibro returns at once and the server drops
  the request at its next batch. Measured through the model server: a 75 s prefill stopped ~25 s
  after the abort.
- **Leaked reasoning.** With thinking off some small models think anyway and close a `</think>`
  that was never opened. `LLMClient` moves that text into the reasoning channel mid-stream and the
  agent sends an SSE `rewrite` so the UI replaces it.
- **Models that cannot stop thinking.** Some vision models reason on every request whatever
  `reasoning_effort` or a `/no_think` prompt says, so helper calls budget tokens for reasoning
  plus the answer.

---

## 4a. Kiwi Sense and steering

Kiwi Sense is a small reading model that runs beside the language model. It never asks that model
anything, so it works the same for every model. It is used in three places: to read each user
message before the model sees it (the helpers, below), to judge the replies (steering) and to
match meaning when no word matches (memory and message recall, search ranking, notes).

### The Sense core (`kiwibro/sense/`)

- **Questions it answers**: how close in meaning two texts are (static word-piece embeddings);
  which behaviour class a sentence falls in (a small trained probe, `probe`); whether a text is
  going round in circles (the share of repeated four-word runs, `repeat`); whether two texts share
  wording (byte trigrams, any language, `lexical`).
- **Methods** of `kiwibro.sense.Sense`: `embed`, `coverage`, `similarity`, `lexical`, `closeness`,
  `nearest(query, texts, limit, floor)`, `rank`, `probe`, `repeat`. Texts are cut at 4,000
  characters; a small cache keeps 1,024 embeddings.
- **One core, two engines.** The core is written in freestanding C (`core.c`) and compiled to
  WebAssembly (`build.sh` → `static/sense/sense.wasm`, committed so that nobody needs a compiler).
  The data (vocabulary, embeddings and the probe) is `static/sense/sense.bin`, built by
  `scripts/sense/` (`build_blob.py` from a model2vec model, `train.py` for the probe). On the
  server, `KIWIBRO_SENSE_ENGINE` (`auto`, `wasm` or `python`) picks the WebAssembly core when the
  `wasmtime` package is installed and otherwise `pyengine.py`, a Python twin that gives the same
  results. The browser always runs the WebAssembly file (`static/js/sense.js`).
- **Meaning matches.** `memory` (`MEANING_MATCH` 0.41) and `context` recall (§6) use Sense to
  find the one memory or earlier message that answers the question in other words. Search ranking
  adds Sense closeness to the word overlap (§7).

### Helpers (`sense/helpers.py`, `readings.py`)

Five tiny trained deciders ride in `sense.bin` as extra sections (tags starting with `H`) and
run in the same core, on the server and in the browser. They never write text and never call
the language model, so they read every message the same way whatever model is loaded, in a few
milliseconds on the CPU. Each is a hashed feature table (int8) feeding one small hidden layer.

| helper | reads | decides |
|---|---|---|
| `Hlng` | one text (word and character pieces, accents kept) | which of 55 languages it is in |
| `Hmsg` | the message | intent (10 kinds), web need (`none`, `own`, `asked`, `live`, `lookup`), time (`none`, `clock`, `now`, `recent`, `past`, `future`), topic (18), and whether it says something about the user worth keeping (`fact`, `preference`, `passing`, `unsure`) |
| `Htag` | each word of the message | time phrases, demands on the answer (format, length, audience, tools) and the words that name its subject |
| `Hshf` | the exchange before and the message | `followup`, `same` topic or `new` topic |
| `Hrel` | a saved memory or note and the message | whether the item bears on the message |

`readings.py` turns their readings into decisions made in code:

- **English only.** `Helpers.language` judges only what the user wrote: code, links, paths,
  file names and quotations are taken out, short messages in the Latin script ("ok", "ciao")
  pass, and English around foreign words is English. A message that is not English gets one
  fixed reply asking for English (`readings.ENGLISH_ONLY`, naming the language when the helper is
  sure), written by code before any model is asked; the reply's meta carries
  `english_only` and the language code. The `english_only` setting (on by default) turns it off.
- **Search first.** A search the word rules want is dropped when `Hmsg` is at least 0.9 sure the
  message needs no web (`WEB_NOT_SURE`) and the user did not say to search; a message the word
  rules miss is searched when `Hmsg` is at least 0.6 sure it asks for a search or for live data
  (`WEB_SURE`) and it is not a follow-up.
- **Memory.** Facts are not captured from a message `Hmsg` is at least 0.7 sure is hypothetical,
  about someone else or a passing mood (`KEEP_SKIP`). A memory or note that only shares a word
  with the message is left out when `Hrel` is at least 98% sure it does not bear on it
  (`RELATE_FLOOR` 0.02: dropping a true memory costs more than letting a lookalike in); pinned
  items and the user's profile always stay.
- **Follow-ups.** When `Hshf` reads a follow-up ("and in Rust?"), memories and notes are looked
  up with the exchange before it as well, and the helpers start no search from the bare
  follow-up.
- **The turn's note** gains up to three lines: the demands the answer must honour, the dates the
  message means (`resolve_time` works "last week", "since March", "3 days ago" out from today's
  date), and, when `Hshf` is at least 0.85 sure the topic changed (`NEW_TOPIC_SURE`), that the
  message stands on its own. `Hshf` is the least certain helper (it agrees with the teacher on
  62% of held-out pairs, and short elliptical follow-ups such as "and in Rust?" come out close to
  a tie), so a new topic is only marked when it is plain.

The whole reading is in `done.inference.reading`. Without the helpers in `sense.bin`, or with
Kiwi Sense off, every function gives a neutral answer and Kiwibro behaves as before.

Training (`scripts/helpers/`, run on Google Colab): `label.py` collects real messages from open
chat datasets and has a teacher model label them (intents, web need, time, topic, keep, phrases,
topic shifts, which memories bear on a question); `train.py` trains each head with the feature
table quantised while training, packs the heads into `sense.bin` and reports how often each
agrees with held-out labels. The language helper learns from labelled sentences in 55
languages, with English full of names, code words and quoted foreign phrases so that it does not
turn English back.

### Steering (`steer.py`, `agent._steer_reply`)

Steering notices a reply that goes wrong and writes the correction for the model. A sentence is
a problem only when the facts of the turn say so:

| verdict | the reply | when it is a problem |
|---|---|---|
| `other_identity` | speaks as another assistant ("I am ChatGPT") | always |
| `adopted_user` | speaks as the user ("I am <the user's name>") | always |
| `denial` | says it cannot do what a tool it has can do | when that tool is available this turn and no tool has run |
| `unbacked_claim` | claims an action that did not happen ("I've saved the file") | when no tool has run this turn |
| `unkept_promise` | announces a step and stops ("Let me search for that.") | when the reply ends there without a tool call |
| `meta_leak` | talks about its instructions ("The user is asking about …") or copies the turn's note | when it opens the reply, or when it repeats the note |
| `off_topic` | answers the previous question again | when the new question is about something else |
| `wrong_result` | gives a different number than the sum works out to | when Kiwibro calculated the sum itself (below) |

Each verdict carries a short note for the model. The agent reads the opening of a streamed reply
(the first 500 characters, `EARLY_CHARS`), checks each finished sentence there, and stops the
stream at a wrong one (`Agent._read_opening`). A reply that is wrong at the end is retried once
with the note at a temperature of 0.3 (`RETRY_TEMPERATURE`); if the retry fails the same way,
`steer.repair` removes the offending sentences. The verdict kinds appear in
`done.inference.steered` and in the chat as a *Kiwi Sense caught it* line.

Some mistakes need no second attempt, because code knows the right text. Such a verdict carries a
`rewrite` and the sentence is replaced where it stands: a reply that repeats the user's own words
back as its own (“My dog is named Biscuit”) is turned round (“Your dog is named Biscuit”), and a
reply that ignores facts the user just stated becomes a one-line acknowledgement of them.

### Kiwibro does it itself (`agent._act_instead`, `compute.worked`, `plan.search_request`)

What code can do for certain is not left to the model:

| the message | what Kiwibro does before or after the model replies |
|---|---|
| asks for a sum or unit conversion (“12% of 480”, “26.2 miles in km”) | `compute.worked` calculates up to three of them (`MAX_WORKED`) and puts the exact results in the turn's note; `compute.gives` then checks the reply states them |
| asks to search, or asks about the world as it is now, and the model called no web tool | makes the search and hands the results to the model. Never for the user's own things (notes, files, memories) |
| dictates a note, or says “save that as a note”, and the model called no `note` tool | saves the note (origin `kiwibro`) and tells the model it is saved |
| states facts about the user | saved before the reply (§6); the model is told what was just learned |
| holds several questions | they are listed in the turn's note (up to five), and a reply that skips one is asked for it once |
| asks who Kiwibro is or what it can do | a tool call is refused: the answer is in the prompt |

A stand-in action shows in `done.inference.acted_instead` and as a chip under the reply. When a
final answer repeats text the model already wrote earlier in the turn, the earlier piece is
emptied and a `retract` event removes it from the page. A reply wrapped in `<answer>`, `<output>`
or a similar tag is unwrapped (`toolcall.unwrap`), in the stream as well.

Per-model habits are kept in the `model_traits` table (schema v8: model, kind, count, last
time). A model that was corrected often is reminded of its habit at the start of later turns
(`Agent._standing_reminders`, in a volatile note, never in message 0). Circling is a habit too:
`LOOP` counts the replies a model went round in circles in, and from `LOOP_HABIT` (2) the
model's requests are sampled against repetition. `LOOP` is not a verdict: the reply itself is
judged by the degenerate-output check (§7), and the trait only changes how that model is sampled.
Traits are read over the last `TRAIT_DAYS` (30) days.

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
model   = the model's record: loaded_context_length if loaded, else its trained context
context = loaded context (or min(max_context_length, 32768) when not loaded)
reserve = min(settings.max_output_tokens, max(512, context // 8))
budget  = max(1024, context - reserve - 512)                      # context.compute_budget
```

The tool schemas render outside the messages (~3.5k tokens for the auto set), so
`prepare(..., reserve_tokens=…)` keeps them out of the message budget and compaction
starts before the window overflows. When the auto set would take more than 35 % of the
budget (a small window that cannot be enlarged), the minimal set is sent instead
(`AgentManager._tool_list`); other tools still run when called by name.

**Messages longer than the window.** A user message over ~40 % of the budget (3.6 characters
per token, at least 4,000 characters) is stored in the document library (`library.stash_text`);
the user message's meta gets a fixed `prompt_text` (head + tail + the handle to
`doc_read`/`doc_search` the rest) that `to_llm_message` sends instead, so the prompt stays
KV-stable. The transcript, the episodic index and memory extraction keep the full text.
Messages up to 400,000 characters are accepted.

History sent: system prompt + retrieval memory block + notes block + messages with
`seq > conversation.summary_through_seq` + newest user message. Tool transcripts are
included (they are part of the past) with age-decayed caps: the 2 newest tool
results keep ~4000 chars, the next 4 get ~1200, older ones ~300 (head+tail, so
conclusions survive), so stale tool output does not dominate a long prompt.
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
4. On a context-limit HTTP 400 from the model server (`ContextLimitError`): summarise half of
   the remaining history and retry; after 2 failures truncate to system prompt + last 6 messages
   and retry.

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
  It runs only when the pending window holds a memorable user statement or is full
  (`memory.worth_extracting`; see `docs/MEMORY.md`); otherwise the watermark stays put.
  Measured through the model server on a CPU-only laptop: an extraction beside the next reply
  slowed its first token from 2.8 s to 8.4 s and stored a fact that reply had just changed. The
  worker asks the selected model for strict JSON operations
  (`{"ops":[{"op":"add|update|delete","id?":"...","kind":"...","content":"..."}]}`)
  given the last few turns plus a compact list of existing memories (top ~40).
  Validate: content 3–300 chars, kind whitelist, no duplicates, max 5 ops applied
  per turn; cap 400 memories total (prune oldest low-confidence non-pinned).
- **Dedupe**: normalise (lowercase, collapse whitespace, strip punctuation) and
  compare with `difflib.SequenceMatcher` ≥ 0.86 or token Jaccard ≥ 0.72; merge by
  keeping the clearer/longer content and bumping `updated_at`.
- **Retrieval for prompt**: pinned memories first, then FTS5 matches for the newest
  user text, then most recently used; render compact bullets under
  `## Memory` capped at ~800 tokens. Mark retrieved memories' `last_used_at`.
- **Recall by meaning**: when no memory shares a key term with the message, the one memory
  that answers it in other words is added (`memory._meaning_matches`: the closest memory by Kiwi
  Sense, only at `MEANING_MATCH` 0.41; "my wife" finds "married to Selin"). One-line memories give
  Sense little to read, so only the closest one rides.
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

### Earlier messages and recall by meaning

When a question refers to an earlier exchange, `context.py` recalls it. The order is: the chat's
newest messages that share words with the question; then, when none does, the messages that
answer it in other words (`context._recall_by_meaning`: the newest 240 messages of the chat are
compared with Kiwi Sense, and a match needs `MEANING_RECALL_MATCH` 0.44); then the words of the
two newest messages. Recalled messages are put in the turn note, not in message 0.

### Notes

A note is a longer page: a title (at most 120 characters, otherwise its first line), a markdown
body (at most 20,000 characters) and up to eight lower-case tags. Notes live in the `notes` table
with a full-text index (`notes_fts`, schema v6); the notes list is ordered pinned first, then by
last change. The user writes notes in the Notes panel; the model writes them with the `note` tool
(`note(action, title, content)`), which records the chat as `source_conversation_id`. The
notes block of a turn holds the two notes that match the message best, at most 600 characters
(`Notebook.relevant`, `Notebook.render_block`).

---

## 7. Agent loop & tools

`POST /api/chat` starts a worker thread per turn that pushes SSE events into a
queue; the HTTP response streams that queue. Events and approval flow are in
`docs/API.md`.

Tool rounds per turn are configurable (`max_tool_rounds` setting, default 16, range 1–32). After the final step, append a short note if truncated.
Every assistant tool-call message is stored; every tool result is stored as a
`tool` message (content capped). Reasoning text is stored on assistant messages but
not re-sent as history.

Tool catalog (OpenAI function schemas): `tools.TOOL_SPECS` holds 46 tools; `tools.AUTO_TOOLS`
is the set of 22 sent by default. The main entries:

| tool | args | permission default | notes |
|---|---|---|---|
| `web_search` | `query`, `max_results?` (1–8) | auto | providers in tiers (below); results ranked by words and meaning; 10 min cache |
| `fetch_page` | `url`, `max_chars?` (≤8000) | auto | http(s) only; SSRF guard (reject localhost/private/link-local/IPv6 unique-local, re-resolve redirects); readable text |
| `read_file` | `path`, `start?`, `end?` | auto | workspace-sandboxed, ≤200 KB |
| `list_files` | `path?` | auto | depth ≤ 2, ≤ 200 entries, hidden-heavy dirs pruned |
| `write_file` | `path`, `content` | ask | workspace-sandboxed, mkdir parents, ≤256 KB, preview in approval card |
| `replace_in_file` | `path`, `old`, `new` | ask | exactly-one-match replace, workspace-sandboxed |
| `run_command` | `command`, `cwd?` | ask | cwd inside workspace, 60 s timeout, 6 KB output cap, blocklist |
| `remember` | `content`, `kind?` | auto | writes memory immediately |
| `note` | `action`, `title?`, `content?` | auto | saves, appends, reads and searches notes |

### Tool layer

Tools live in `kiwibro/tools.py` (`ToolSpec` catalog) with implementations split by
domain: `kiwibro/research.py` (parallel `multi_search`/`fetch_pages`/`web_research`
evidence packs with `[n]` citations, URL dedupe + term-overlap rerank, fetch cache),
`kiwibro/codebase.py` (`glob_files`, `file_outline`, atomic `apply_patch` with diff,
`file_manage`, `multi_command` pipelines, read-only `git`, sandboxed
`run_python`/`run_javascript` snippets in `.kiwibro-sandbox/` scratch dirs),
`kiwibro/desktop.py` (`system_info`, `open_path`, `clipboard`, `notify`,
`screenshot` — all degrade with a plain message when the desktop binary is absent),
`kiwibro/delegate.py` + `kiwibro/routing.py` (capability registry over the engine's model
records; `model_info`, `delegate`, `describe_image` and map-reduce `study_file` keep the main
model in charge while the *smallest adequate* installed model answers bounded side-questions,
loaded for the call and unloaded after it). Persistent terminals (`terminal_session`), grep
(`grep_files`) and batched reads (`read_files`) ride on `kiwibro/sandbox.py`; the persistent
`task_board` and the multi-tab `browser` ride on `kiwibro/tasks.py` / `kiwibro/browser.py`.
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

Web search (`search.py`): six providers, in three tiers. Tier 0 is Yahoo and Brave; tier 1 is
Bing (HTML) and DuckDuckGo; tier 2 is Wikipedia and Bing RSS, the strict providers. Each tier is
queried in parallel, and the next tier runs only when the results so far are too few. A provider
that is blocked goes into a cooldown of 120 s. Each provider call has an 8 s timeout. The merged
results are deduplicated and ranked by the words they share with the query and by Kiwi Sense
closeness (`search.best_first`), so the result that answers the query is first. Requests carry a
desktop Chrome user agent and `Accept-Language: en-US,en;q=0.9`. `fetch_page` returns the page's
text.

Follow-up searches are completed before they run (`research.anchor_query`): a query that only
points at something ("how old is he") gets the subject the conversation is about added, when the
subject is named in earlier messages.

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
   the reply. `StreamFilter` holds call markup back while streaming. A model whose
   record does not declare tool support (`model_info["tools"]` is false) gets the **text
   tool protocol**: a byte-stable manual in message 0 (settings-only tool set) and tool
   history rewritten as plain text.
3. **Decide**. `safety.decide` (policy, command risk, autonomy, breakers) merges
   `guard.inspect_call` (content); the stricter verdict wins. The guard refuses
   web-bound arguments that carry secrets, blocks overwrites containing elision
   placeholders (`# ... rest unchanged`) or clobbering binary files, and escalates
   large shrinks, big folder deletes and overwriting moves.
4. **Checkpoint** (`kiwibro/checkpoints.py`). Before a write-risk call runs, every
   file it may touch is snapshotted, and before a command runs, the workspace files
   it was graded to change or remove (`Decision.targets`: globs expanded, folders
   walked without `.git` and build output, newest 500 first; `git reset --hard` and
   friends copy the working tree). Snapshots go into `<data dir>/checkpoints` (zlib,
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
  request's prefix matches what the server has cached.
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
  code and data (`safety.APP_DIR`, the data dir), system files, and grades
  `safe < low < medium < high < critical`. Credentials, start-up files and
  Kiwibro's own places are recognised before the workspace, so a working folder
  that contains them (`~/.config`, a `~/Desktop` holding the checkout) does not
  unlock them. Removals and in-place edits (`sed -i`, `perl -pi`, `awk -i inplace`)
  of files only known at run time (`xargs`, `find -exec`, `$(…)`) are high; globs
  count every file they match and brace ranges expand; one command that changes or
  removes more than 20 existing files in total is high whatever the route (several
  globs, a loop, `truncate`, `tee`, `ln -sf`, Ruby or inline code that lists folders,
  `zip -m`, `7z -sdel`, scripted `vim`/`ex`/`ed`, `git rm -r`); a symlink is graded
  where it is made, before it is followed; `mv` into `/tmp` is a removal
  (commands get a private `/tmp`); removing a file larger than an undo copy
  (`config.UNDO_FILE_BYTES`) is high. Each graded part carries flags: `loss` (can
  destroy data Kiwibro cannot put back), `outside` (a high part that has to leave
  the sandbox to work), and `Risk.paths`, the workspace files to copy before it runs.
  File, code, desktop and remote tools are graded by `safety.assess`.
- **Policy.** `safety.decide` combines the grade with the per-tool permission
  (`deny` always wins), the `autonomy` setting, graded per-turn circuit breakers
  (mass deletes/edits escalate one level per multiple of the limit), a
  conversation-wide breaker over the last hour (`SESSION_DELETE_LIMIT` = 24
  removals, `SESSION_WRITE_LIMIT` = 120 file changes, so a long unattended session
  cannot spread a spree over many turns), conversation taint from prompt
  injection, and `guard.inspect_call` (stricter wins). Critical is always denied;
  autonomous mode turns every "ask" into a deny with reasons. Denials reach the
  model as `Decision.denial_text()`. `Decision.allow_key` is what "allow in this
  chat" remembers: the tool and the kind of step, without its operands;
  `Decision.can_always` is false for `loss` steps and for anything a breaker,
  taint or the guard escalated, so those are approved one at a time.
- **Confinement.** `safety.confine` wraps commands, terminals and
  `run_python`/`run_javascript` in a transient systemd user scope
  (`MemoryMax`, `TasksMax`) and bubblewrap: read-only root, writable workspace
  (snippets: writable scratch only, no network), private `/tmp`, own PID
  namespace, `~/.ssh`/credential dirs and the session bus hidden. Every layer is
  feature-detected and self-tested; the `sandbox` setting picks auto/strict
  (no network)/off. Start-up files and Kiwibro's code and data that lie inside a
  writable mount are mounted read-only again on top of it, and credential dirs are
  hidden on top (bubblewrap applies mounts in order). A high-risk call the user
  explicitly approved runs unconfined only when it has to leave the box
  (`Decision.escape`: `git push` needs `~/.ssh`, a change outside the workspace);
  an approved removal or edit inside the workspace stays confined.
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
  per 100 words; code blocks ignored). The agent raises out of the stream (closing the
  stream stops the server generating). A reply that repeated itself keeps its good start (up
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

---

## 8. Database & settings

SQLite at `data_dir/kiwibro.db`, `PRAGMA journal_mode=WAL`, `foreign_keys=ON`,
`busy_timeout=5000`. Schema version 8 (migrations forward-only, version table):

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
summary_segments(...)                        -- v2: one bounded summary per seq range
memories(id TEXT PK, kind TEXT, content TEXT, normalized TEXT, tags TEXT,
  source_conversation_id TEXT, confidence REAL DEFAULT 1, pinned INTEGER DEFAULT 0,
  created_at TEXT, updated_at TEXT, last_used_at TEXT,
  use_count INTEGER DEFAULT 0 /* v2 */, origin TEXT /* v4: you | chat | learned */)
notes(id TEXT PK, title TEXT, body TEXT, tags TEXT /* JSON list */, pinned INTEGER,
  origin TEXT /* you | kiwibro */, source_conversation_id TEXT,
  created_at TEXT, updated_at TEXT)          -- v6; notes_fts (FTS5) indexes title, body, tags
model_traits(model TEXT, kind TEXT, count INTEGER, last_at TEXT,
  PRIMARY KEY (model, kind))                 -- v8: per-model steering habits (§4a)
permissions(tool TEXT PK, mode TEXT)         -- auto | ask | deny
settings(key TEXT PK, value TEXT)            -- JSON values
usage(id INTEGER PK AUTOINCREMENT, conversation_id TEXT, model TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, created_at TEXT)
```

v7 drops the settings of the old external-server engine (its address, its token and the
engine choice); they are neither read nor returned.

Default settings (`config.default_settings`): model `null` (auto = the loaded chat model, else
the first chat model), server_url `""` (from `KIWIBRO_SERVER_URL`, attached mode only),
direct_runtime `""`, direct_models_dir `""`, direct_threads `0` (every physical core),
direct_gpu `auto`, agent_enabled `true`, memory_enabled `true`, sense_enabled `true`,
max_output_tokens `2048`, temperature `0.6`, workspace `~/Kiwibro`, user_name `""`,
tool_profile `auto`, reasoning `auto`, autonomy `smart`, sandbox `auto`, remote_hosts `[]`,
max_tool_rounds `16`, and the permissions table above. The Settings API returns the merged values.

---

## 9. Frontend contract

Single page, no build step, ES modules, no external resources. Files:
`templates/index.html`, `static/css/*.css`, `static/js/*.js` (listed in §2),
`static/img/kiwibro.svg` + generated PNG icons + favicon, and `static/sense/` (the Kiwi Sense
data file and WebAssembly core, §4a).

Design language:

- GNOME/Ubuntu dark: base `#1b1b1f`, surfaces `#242429`/`#2c2c32`, text `#f4f2f0`,
  muted `#9a9aa3`, hairline `rgba(255,255,255,.07)`, radius 12–14 px. Accent: kiwi
  green `#9BCE6B` with a warmer secondary `#E95420` used sparingly (recording dot,
  danger, focus). Font stack `Ubuntu, 'Segoe UI', system-ui, sans-serif`; code
  `'Ubuntu Mono', monospace`.
- Layout: 56 px header bar (kiwi icon + wordmark left, centered model pill, right
  icon buttons: settings). 264 px sidebar (New chat button, conversation list, the
  places Notes, Memory, Models and How it works, tiny footer stats) collapsible. Chat column
  max-width ~760 px. Composer as a floating rounded field with agent toggle, send/stop, context ring.
- Empty state: large kiwi mark, "Kiwibro" wordmark, one-line subtitle, 3 quiet
  suggestion chips. Very low text overall.
- Messages: user bubbles right-aligned with accent tint; assistant plain text with
  markdown, code blocks (language label + copy button), tool activity as thin
  collapsible rows with icons and status dots, reasoning collapsed by default.
  A reply Kiwi Sense corrected carries a *Kiwi Sense caught it* line.
- Approval card inline in the transcript: tool icon + name, one-line summary,
  expandable details (command / file path + preview), risk badge, buttons
  `Allow`, `Always` (Manual mode) or `Allow in this chat` (automatic modes), `Deny`;
  a step that can destroy data (`can_always: false`) offers `Allow` and `Deny` only.
- Sheets:
  - memory sheet (search, kind chips, rows with pin/edit/delete and an origin
    label — *Added by you*, *Saved in a chat*, *Learned* — that links to its chat, add field);
  - Notes sheet (`notes.js`): a list on the left and an editor on the right, one pane at a time
    on phones, edits saved as one types, tags and pinning, and the notes Kiwibro saved in chats;
  - Models sheet (`models.js`): the models on this computer with their source, fit and size,
    the engine (the llama.cpp build in use, the installed builds, *Install engine*), and the
    Hugging Face search and downloads;
  - How it works (`how.js`): eight small animated figures, one per idea of the design, one of
    which runs Kiwi Sense in the browser;
  - settings sheet (`sheets.js`): llama.cpp runtime (empty = the engine Kiwibro installed or
    found), Models folder (an extra folder; Kiwibro's own and LM Studio's read-only folder are
    always listed), CPU threads, model server address (empty = models on this computer), the
    running servers, permissions, behaviour, workspace path, data location and "forget all".
    Permissions is one Automatic/Manual switch: Automatic shows only what to do with risky steps
    (ask me / refuse them); the per-tool Auto/Ask/Deny list appears in Manual only.
- Model popover with state dots, context size and tool-capable marker; selecting swaps
  instantly (toast) and the header dot pulses amber while the model loads and warms.
- Rich replies (`markdown.js` + `richblocks.js` + `mathml.js`): code is highlighted by a small
  escape-first tokenizer; `diff` blocks colour added/removed lines; `csv`/`tsv` blocks
  render as tables with the source folded below; complete `html` and `svg` blocks
  render as live previews (final render only, never while streaming) with Preview/Code
  tabs. SVG goes through an `<img>` data URI; HTML through an iframe with an opaque
  origin and a CSP that blocks every network request, with refresh tags removed and
  link clicks inert. Scripts stay paused until the user presses *Run scripts*: a page
  can still navigate its own frame, and a prompt-injected page could ask the model for
  one that carries chat text out in a URL. LaTeX maths in replies becomes MathML
  (`mathml.js`); anything it does not know stays as written.
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
  optionally open the browser. `--open` flag. `scripts/python.sh` finds an interpreter with
  Kiwibro's dependencies, creating a local virtualenv when needed.
- `scripts/kiwibro.sh` — desktop entry point: start server in background if not
  healthy (nohup + log), wait for `/health`, then open Chrome `--app=` window with
  `--class=Kiwibro`; fall back to `xdg-open`.
- `scripts/kiwibro.desktop.in` — template for the desktop entry (Name=Kiwibro,
  Comment=Local AI agent, Exec=kiwibro.sh, Categories=Utility;Chat;AI); the installer writes
  `kiwibro.desktop` with the repository path in place of `@ROOT@`.
- `scripts/install.sh` — installs the `.desktop` file and icons into
  `~/.local/share/{applications,icons}`, refreshed with `update-desktop-database`;
  prints one line on completion. Must not require root.
- `scripts/sense/` — builds the Kiwi Sense data file (§4a); `kiwibro/sense/build.sh` compiles the
  WebAssembly core. Both outputs are committed, so running Kiwibro needs neither.
- `scripts/helpers/` — labels data with a teacher model and trains the Kiwi Sense helpers into
  the same data file (§4a), on Google Colab.
- Optional: the `wasmtime` package runs the Kiwi Sense WebAssembly core on the server;
  without it the Python twin runs.
- `README.md` — hero screenshot, features, install, run, engine setup (*Install engine* or an
  existing llama.cpp build), privacy, architecture summary, license.

---

## 11. Acceptance checklist (QA)

- [ ] `python3 -m pytest -q` green (all unit/API tests, no network, no model server: the
      engine tests use a fake llama.cpp-compatible server and a fake Hugging Face / GitHub).
- [ ] Server boots clean, `/health` OK, UI loads with zero console errors.
- [ ] Live chat against the llama.cpp engine (local mode) and against an attached server
      streams tokens; no raw error surfaces.
- [ ] Model switch mid-conversation works; past chats and memories unchanged.
- [ ] Tool flow: web search auto-runs; `write_file`/`run_command` ask permission,
      allow/always/deny all behave; denied tool does not execute.
- [ ] Memory survives restart, dedupes near-identical facts, appears in prompt,
      editable/deletable in UI.
- [ ] Context: long conversation compacts automatically (summary chip), no
      context-length error surfaced, works after compaction.
- [ ] Mobile/responsive pass; dark theme consistent; screenshots captured.
- [ ] README accurate; repo clean; `.desktop` launcher works; icon renders.
