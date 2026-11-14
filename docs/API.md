# Kiwibro HTTP API Contract (v1)

All endpoints return JSON unless noted. Errors use `{"error": {"code": "...",
"message": "..."}}` with HTTP 4xx/5xx. Base URL `http://127.0.0.1:8777`.
Request/response bodies are UTF-8 JSON. Timestamps are ISO-8601 UTC strings.

Static: `GET /` → `templates/index.html`; `/static/*` from disk; `GET /favicon.ico`.
Security: only loopback; reject `Host` other than `127.0.0.1`/`localhost` (incl.
port). No auth. `Cache-Control: no-store` on API responses.

---

## Health & bootstrap

### `GET /health`
```json
{"ok": true, "version": "3.0.0", "started_at": 1761000000, "pid": 4242, "busy": false,
 "code_version": "a8a52b7b0cad72fd", "ui_version": "3f0c9e2a71d4b6e8", "update_pending": false,
 "engine_ok": true, "engine": "local", "online": true, "model": "qwen3.8-2b",
 "mind": {"usable": true, "running": false, "pid": null}}
```
`engine` is `local` (Kiwibro starts and stops llama.cpp servers itself) or `attached`
(`KIWIBRO_SERVER_URL` is set, see Settings). `engine_ok: false` adds `engine_error` with the
reason, for example when no llama.cpp build is installed in local mode. `/health` also carries
the runtime fields of `/api/version` below; `started_at` is when the current server image
started (Unix seconds, renewed by a self-update). `online` is whether the internet is
reachable right now: a TCP probe to public addresses, remembered for a short while
(`kiwibro/connectivity.py`; `KIWIBRO_OFFLINE=1` or `0` in the environment forces the answer).
`mind` is Kiwi Mind's state: `usable`, `running` and `pid`; a build without Kiwi Mind
answers `{"usable": false, "running": false, "pid": null}`.

### `GET /api/version`
Cheap poll (no model call): which code this process serves.
```json
{"ok": true, "version": "3.0.0", "started_at": 1761000000, "pid": 4242, "busy": false,
 "code_version": "a8a52b7b0cad72fd", "ui_version": "3f0c9e2a71d4b6e8", "update_pending": false,
 "update_blocked": false}
```
`code_version` is the content fingerprint of the served source (`kiwibro/selfupdate.py`;
`null` when not started through `app.py`). `update_pending`: the code on disk differs;
the server restarts onto it once `busy` (a reply or memory extraction running) is false.
`update_blocked`: the new code failed to compile/import, so the server stays on the
current version. `ui_version` fingerprints only the page assets (templates, static): open
windows reload when it changes, not for backend-only updates. The launcher replaces a
stale idle server.

### `GET /api/bootstrap`
Aggregate for first paint.
```json
{
  "health": {"ok": true, "version": "3.0.0", "engine_ok": true, "engine": "local"},
  "settings": {...},                       // see settings
  "models": [{"id": "qwen3.8-2b", "name": "qwen3.8-2b", "type": "llm",
              "state": "loaded", "context": 32768, "max_context": 262144,
              "tools": true, "vision": false, "quantization": "Q4_K_M"}],
  "current_model": "qwen3.8-2b",
  "readiness": {"model": "qwen3.8-2b", "phase": "warming", "detail": "Getting ready", "context_length": 32768},
  "conversations": [{"id": "...", "title": "...", "model": "...", "pinned": false, "private": false,
                     "message_count": 4, "created_at": "...", "updated_at": "..."}],
  "memories_count": 12,
  "notes_count": 3,
  "workspace": "/home/you/Kiwibro"
}
```
If the engine cannot list models (`health.engine_ok` is `false`, with `health.engine_error`),
`models` is `[]` and the response is still 200. With models listed, bootstrap also queues
the selected model's load + warm-up (see `GET /api/models/ready`); the response never waits
for it. `conversations` has the same items as `GET /api/conversations`.

---

## Models

### `GET /api/models`
Add `?refresh=1` to rescan the model folders first.
```json
{"ok": true, "engine_ok": true, "current": "qwen3.8-2b",
 "models": [{"id": "...", "name": "...", "type": "llm|vlm", "state": "loaded|not-loaded",
             "context": 32768, "max_context": 262144, "tools": true, "vision": false,
             "capabilities": ["tool_use"], "publisher": "...", "quantization": "...",
             "key": "...", "display_name": "...", "arch": "...", "params": "...",
             "size_bytes": 2400000000, "quant_bits": null, "reasoning": ["off", "on"],
             "reasoning_default": "on", "source": "kiwibro", "deletable": true,
             "loaded_config": {"context_length": 32768, "parallel": 4, "flash_attention": true},
             "context_choices": [65536]}],
 "readiness": {...}}
```
`context` = `loaded_context_length` if loaded else `min(max_context_length, 32768)`.
Embedding models are excluded. `tools` = capabilities include `tool_use`. `vision` is true for
a model with a vision projector file. `key`, `display_name`, `arch`, `params`, `size_bytes`,
`quant_bits`, `reasoning`, `reasoning_default`, `source` and `deletable` describe the model
file; they appear for the models of the local engine (`source` and `deletable` as in
*Library and downloads*). `reasoning` lists the allowed thinking switch options when the model
has one. `loaded_config` is present for loaded local models, and `context_choices` (larger
windows the model supports, at most three) for loaded models. `engine_error` is added when
`engine_ok` is false. With an attached server only the first set of fields is present.

### `GET /api/engine`
Which engine serves models and what the local engine found:
```json
{"ok": true, "engine": "local|attached", "direct": {
  "available": true, "error": null, "models_count": 7, "threads": 6,
  "gpu": "AMD Radeon Graphics (RADV RENOIR)", "gpu_available": true,
  "runtime": {"path": ".../llama-server", "label": "...", "kind": "vulkan",
              "source": "kiwibro", "version": "...", "own": true},
  "runtimes": [{"path": "...", "label": "...", "source": "kiwibro"}],
  "model_dirs": ["/home/you/.local/share/kiwibro/models"],
  "sources": [{"path": "...", "source": "kiwibro", "label": "Kiwibro", "models": 2, "bytes": 4800000000}],
  "servers": [{"model": "bartowski/spark-x2.6-4b", "pid": 123, "port": 45885,
               "context_length": 32768, "parallel": 4, "threads": 6, "runtime": "...",
               "started_at": 1793018131, "busy": false, "gpu": "", "adapters": []}],
  "adapters": [{"name": "...", "path": "...", "arch": "...", "base": "...", "tasks": ["code"],
                "scale": 0.8, "trouble": 0, "strength": 1.0, "size_bytes": 1000000}],
  "adapters_dir": "...", "own_models_dir": "...", "own_runtime_dir": "...",
  "idle_sleep_s": 1800, "error": null}}
```
`runtime` is the llama.cpp build in use (`null` when none is installed). `runtime.source` is
`kiwibro` (installed by Kiwibro), `custom` (Settings `direct_runtime` or `KIWIBRO_LLAMA_SERVER`),
`path` (`llama-server` on PATH) or `found` (a build another app keeps on this computer). `gpu`
is the GPU the next model would run on (`""` = CPU only, or graphics acceleration off);
`gpu_available` says whether any build found here sees a hardware GPU. A server's `gpu` is
`""` when it runs on the CPU. `idle_sleep_s` is the time without a request after which a server
frees its memory (0 = never).

The `direct` object is the local engine's state. In attached mode it is reported but not
used for requests.

In local mode every model endpoint above keeps its shape: models come from the GGUF files
on disk (`state: loaded` while a server runs), and `/api/models/context` reloads the model
with the new window. `/health` and `/api/bootstrap` include `engine`.

### `POST /api/engine/stop`
Stops every llama.cpp server Kiwibro runs (frees its memory; the next message starts one
again). `{"ok": true, "direct": {...}}`; `409 busy` while a reply runs.

### `GET /api/models/ready`
`{"ok": true, "readiness": {"model", "phase", "detail", "context_length"}}`. `phase` is
`idle | loading | warming | ready | error`. Opening Kiwibro, selecting a model, opening a
chat and changing a prompt-shaping setting queue a background job: the model is loaded
explicitly with a working window (32K, or 16K/8K when the load is refused for memory; a
loaded model below 16K is reloaded larger while no reply runs) and the exact leading
messages + tools of the next request are prefilled with `max_tokens: 1`, so the first
reply only processes the new message. A reply that starts during the job waits for it
(status `Getting the model ready`) when it is the same prefix, and cancels it otherwise.
`/api/bootstrap`, `/api/models` and `/api/models/select` include the same `readiness`
object. Disabled with `KIWIBRO_WARMUP=0` (the e2e harness does this).

### `POST /api/models/context`
Body `{"id": "spark-x2.6-4b", "context_length": 65536}` (one of the model's
`context_choices`). Reloads the loaded model with that window (a running server cannot be
resized in place). `200 {"ok": true, "current": "<instance id>", "context_length": 65536,
"readiness": {...}}`; `409 no_memory` when the larger window does not fit in memory — the
previous window is loaded again; `409 busy` while a reply runs; `400 unknown_model` when the
model is not loaded; `400 bad_request` when the window is not one of its `context_choices`.

### `POST /api/models/select`
Body `{"id": "qwen3.8-2b"}`. Persists the model in settings; does **not** touch
conversations or memories. Response `{"ok": true, "current": "...", "readiness": {...}}`
(the load + warm-up runs in the background). `400 unknown_model` if the id is not in the
model list (checked only while the engine answers). Selection applies to new turns in any chat.
`{"id": "none"}` selects no helper model: Kiwi Mind answers alone. It is accepted only while
Kiwi Mind is usable (`GET /api/mind`); otherwise `none` is an unknown id.

---

## Kiwi Mind

Kiwibro's own model (`kiwibro/mind/`): a pack downloaded once and a host process that
serves it. Everything here answers `404 not_found` in a build without Kiwi Mind.

### `GET /api/mind`
What Kiwi Mind is and whether it can be used. The answer has no `ok` field.
```json
{"usable": true, "installed": true, "version": "...", "bytes": 486110000,
 "node": "/usr/bin/node", "running": true, "pid": 4242, "rss_bytes": 512000000,
 "skills": ["chat", "web", "code", "terminal", "computer", "tools", "write"], "helper": {"id": "qwen3.8-2b", "name": "qwen3.8-2b"},
 "reason": "", "download": null}
```
`usable` is true when `reason` is empty, that is when the pack is installed, every file
matches its checksum and a new enough Node.js was found. `installed` only says a manifest is
there; `version` and `bytes` come from it (`null` and `0` without one). `node` is the Node.js
binary the host runs on (`null` when none was found). `running`, `pid` and `rss_bytes` are the
host process (`rss_bytes` `null` when it is not running). `skills` lists the skill names the
pack carries. `helper` is the model that answers hard cases, `{id, name}`, or `null`.
`download` is `null` or the download state below.

### `POST /api/mind/download`
Starts the pack download into `<data>/mind/`, or resumes one that was interrupted.
`{"ok": true, "download": {"state": "running", "done_bytes": 120000000,
"total_bytes": 486110000}}`. A download already running is returned as it is.

Download state: `{"state", "done_bytes", "total_bytes"}` plus `error` when there is one.
`state` is `running`, `done`, `cancelled` or `error`. Each file is written as `<name>.part`
and continues from it after a dropped connection; it takes its final name only when its size
and SHA-256 match the manifest.

### `POST /api/mind/download/cancel`
`{"ok": true, "download": {...}}`; `ok` is false when no download is running. The download
stops at its next block and its state becomes `cancelled`; the partial files are kept for the
next attempt.

---

## Library and downloads

### `GET /api/library`
Installed models with where they live, plus disk, RAM, the engine build and downloads.
```json
{"ok": true, "mode": "local", "models": [...], "sources": [...],
 "disk": {"models_bytes": 0, "partial_bytes": 0, "free_bytes": 0}, "ram_bytes": 16000000000,
 "runtime": {...}, "adapters": [...], "adapters_dir": "...",
 "installer": {...}, "downloads": [...]}
```
`mode` is `local` or `attached`; `models`, `sources` and `adapters` are empty in attached
mode. Each model: `id`, `name`, `publisher`, `params`, `quantization`, `size_bytes`,
`context_length`, `kind` (`llm`, `vlm` or `embeddings`; embeddings models are left out of
`models`), `source`, `source_label`, `deletable`, `loaded` and `fit` (`good`, `tight` or
`too_big`: whether the file fits in RAM with room left for the system, from its size and
the RAM the computer reports). `source` is one of:

| source | where | deletable |
|---|---|---|
| `kiwibro` | Kiwibro's own folder (`<data>/models`), where downloads go | yes |
| `folder` | the folder in Settings (`direct_models_dir`) or in `KIWIBRO_MODELS_DIR` | no |
| `lmstudio` | LM Studio's folders, read-only (remove the file in LM Studio) | no |
| `shared` | model caches of other tools on this computer (llama.cpp, LM Studio), read-only | no |

`sources` lists the folders scanned, with `path`, `source`, `label`, `models` and `bytes`.

### `DELETE /api/library/models`
Body `{"id": "..."}`. Deletes a model Kiwibro downloaded itself and stops its server first.
`{"ok": true, "freed_bytes": 2400000000}`. `409 attached` in attached mode, `409 not_deletable`
for a model in another folder (the error names the folder), `409 busy` while it is answering,
`404 not_found` for an unknown id.

### `DELETE /api/library/adapters` and `POST /api/library/adapters/reset`
Both take the adapter's `path` in the body. The delete removes an adapter from Kiwibro's own
adapters folder (a model that loaded it keeps it until it restarts): `{"ok": true,
"freed_bytes": 1000000}`. Adapters elsewhere answer `409 not_deletable`. The reset gives an
adapter that was turned down its full strength back: `{"ok": true}`. Both answer `404
not_found` for a path that is not a known adapter.

### `GET /api/hub/search?q=<text>&limit=20`
Chat models on Hugging Face matching the text; an empty `q` lists the most downloaded ones.
`limit` is 1–50. `{"results": [{"repo": "author/name", "author", "name", "downloads", "likes",
"updated", "vision": false, "license": "apache-2.0"}]}`. Private and gated repositories are
left out.

### `GET /api/hub/files?repo=author/name`
The GGUF files of one repository: `{"repo": "...", "files": [{"file", "name", "size_bytes",
"sha256", "quant", "projector", "split", "installed", "fit"}]}`. `projector` marks a vision
projector file; `split` marks one part of a multi-part model; `installed` is true when the
file is already on disk at its full size.

### `POST /api/hub/downloads`
Body `{"repo": "author/name", "file": "model-Q4_K_M.gguf"}`. Queues the file. Size and
checksum come from Hugging Face, never from the request. `201 {"job": {...}}`. A file that is
already downloading returns that job. `404 not_found` when the file is not in the repository; `409 no_space` when the disk
cannot hold it.

Job: `{"id", "repo", "file", "name", "state", "received", "total", "speed_bps", "eta_s",
"error", "path", "started_at", "finished_at"}`. `state` is `queued`, `downloading`,
`verifying`, `done`, `error` or `cancelled`. A download is written to a `.part` file and
continues from it after a dropped connection (HTTP Range); the checksum is checked before the
file takes its name.

### `GET /api/hub/downloads`
`{"jobs": [...], "disk": {"models_bytes", "partial_bytes", "free_bytes"}}`. A download that
finished shows up in the library right away.

### `POST /api/hub/downloads/<job_id>/cancel` → `{"ok": true}`
`ok` is false when the job already finished. The partial file is kept for a later attempt.

### `POST /api/hub/downloads/clear` → `{"ok": true}`
Removes the finished, failed and cancelled jobs from the list.

Failed requests use `400 bad_request`, `404 not_found`, `502` when Hugging Face cannot be
reached, and `409` for other conflicts.

---

## Engine builds

Kiwibro runs models with a llama.cpp build. It installs one only when asked
(`POST /api/runtime/install`) and can also use a build already on the computer.

### `GET /api/runtime`
`{"ok": true, "mode": "local", "active": {...} | null, "installer": {...}}`. `active` is
the build in use, with the same fields as `runtime` in `GET /api/engine`.

### `POST /api/runtime/install`
Body `{"kind": "auto"}` (optional; `auto`, `vulkan` or `cpu`, else `400`). Starts the install
in the background and answers `202 {"installer": {...}}`. `auto` takes the Vulkan build when a
Vulkan loader and a GPU device are present, otherwise the CPU build. The build is taken from
the llama.cpp GitHub releases; the install needs the SHA-256 digest GitHub gives for the file.

`installer`: `{"state", "tag", "kind", "received", "total", "error", "supported", "arch",
"installed": [...], "current": {...} | null}`. `state` is `idle`, `checking`, `downloading`,
`verifying`, `unpacking`, `done`, `error` or `cancelled`. Each entry of `installed` has `name`,
`tag`, `kind`, `path`, `size_bytes`, `installed_at` and `current`. A second install while one
runs answers `409 busy`.

### `POST /api/runtime/cancel` → `{"ok": true, "installer": {...}}`
Stops a running install.

### `POST /api/runtime/use`
Body `{"name": "<installed build name>"}`. Makes that build the one Kiwibro starts models
with. `{"ok": true, "installer": {...}}`.

### `DELETE /api/runtime/<name>`
Deletes an installed build: `{"ok": true, "freed_bytes": 80000000, "installer": {...}}`.
`409 busy` while a model server runs on it.

---

## Conversations

### `GET /api/conversations`
```json
{"ok": true, "conversations": [
  {"id": "c_...", "title": "Kiwi facts", "model": "qwen3.8-2b", "pinned": false,
   "private": false, "message_count": 6, "created_at": "...", "updated_at": "..."}]}
```
Order: pinned first, then `updated_at` desc.

### `GET /api/conversations/search?q=<text>`
```json
{"ok": true, "matches": [{"id": "c_...", "snippet": "...the Lisbon trip budget is 900 euros..."}]}
```
Chats whose messages match `q` (full-text search over message content), best match first,
one entry per chat, at most 30. The snippet is a short cut around the match, with the common
markdown marks removed. `q` shorter than 2 characters returns `"matches": []`. Chat titles are
not searched here: the sidebar filter matches them itself.

### `POST /api/conversations`
Body `{"title": "...", "model": "...", "private": false}` (all optional; `title` is at most
200 characters and defaults to `New chat`; `private` must be a boolean, else `400`). Answers
`201` with the new conversation (fields as in `GET /api/conversations/<id>`) and an empty
`messages` list.

### `GET /api/conversations/<id>`
```json
{"ok": true, "conversation": {"id","title","model","pinned","title_auto","created_at","updated_at",
  "workspace": "/home/you/Desktop/app" | null, "summary": "..." | null, "summary_through_seq": 0,
  "compacted": true, "private": false},
 "messages": [{"id":"m_...","seq":1,"role":"user|assistant|tool","content":"...",
   "reasoning": "", "tool_calls": [...], "tool_call_id": null, "tool_name": null, "model": "...",
   "created_at": "...", "meta": {}}],
 "context": {"used": 1234, "budget": 30000, "percent": 4}, "busy": false}
```
`busy` is true while a reply is being generated for that chat. Opening a chat queues a prefill
of its history for the next turn; `?warm=0` reads it without that (an export). `?limit=<1–500>
&before_seq=<n>` pages through a long chat from its newest messages: the response then adds
`"page": {"has_more", "oldest_seq"}`, and the next page is requested with `before_seq` set to
the returned `oldest_seq`. 404 unknown id.

### `PATCH /api/conversations/<id>`
Body `{"title"?: str, "pinned"?: bool, "private"?: bool}` → `{"ok": true, "conversation": {...}}`.
Setting a title turns automatic titling off for that chat. A private chat is never saved to
memory and never shown to other chats. A body with none of these fields answers `400`.

### `DELETE /api/conversations/<id>`
`{"ok": true}`. Deletes messages via cascade; does not delete memories.

### `POST /api/conversations/<id>/clear`
Delete messages and summary of one conversation, keep the conversation.
`{"ok": true, "conversation": {...}}`.

### `POST /api/conversations/<id>/fork`
Body `{"seq": N}` (optional; a non-negative integer, else `400 bad_request`). Copies the
messages with `seq <= N` (all when omitted) into a new chat, in one transaction, keeping
their seqs. The copy keeps the model, the workspace and the private flag. The summary is kept
only when the source's summary reaches up to `N`; otherwise the copy has no summary. The title
is `"<title> (fork)"` and automatic titling is off for the copy. Returns `201`
`{"ok": true, "conversation": {...}}`. Unknown id: `404 not_found`.

### `POST /api/conversations/<id>/compact`
Folds older messages into the chat's summary now, as context work does when the window fills.
`{"ok": true, "conversation": {...}, "compacted": true}`; `compacted` is false when there was
nothing to fold. Unknown id: `404 not_found`.

---

## Chat (SSE)

### `POST /api/chat`
Body:
```json
{"conversation_id": "c_...", "message": "hello", "model": "qwen3.8-2b" /* optional */,
 "attachments": [{"path": "attachments/2026-10-22/1a2b3c4d-photo.png", "name": "photo.png"}] /* optional */}
```
- `attachments`: up to 8 records returned by `POST /api/attachments`; `kind` and `size`
  are re-derived from the file. With attachments, `message` may be empty (the file names
  become the caption) and `meta` is sent before the server describes them.
- If `conversation_id` is null/missing, create a conversation and use it; `"private": true`
  makes that new conversation private (an existing one changes with `PATCH`).
- If a turn is active for that conversation → `409 {"error":{"code":"busy"}}`.
- Response `200 text/event-stream`, headers: `Cache-Control: no-cache`,
  `X-Accel-Buffering: no`, `Connection: keep-alive`. Heartbeat comment `: ping`
  every 15 s while idle (e.g. waiting on approval).
- Validation: message non-empty, ≤ 400,000 chars → 400. A message longer than 40 % of the
  free prompt budget (at least 4,000 characters; the budget is counted at 3.6 characters per
  token) is stored in the document library; the model reads its start, its end and the handle
  to read the rest (`doc_read` / `doc_search`). The transcript keeps the full text.

SSE frames use `event: <type>\ndata: <json>\n\n`.

| event | data | when |
|---|---|---|
| `meta` | `{conversation_id, user_message_id, model, title, context:{used,budget,percent}}` | first |
| `status` | `{phase, label}`; `phase` is `thinking`, `seeing`, `web`, `reading`, `writing`, `running` or `tool` | optional, for activity line |
| `brain` | `{brain: "kiwimind"\|"helper", skill, hard, name?}` | which brain writes this reply: Kiwi Mind with the skill it chose (`skill`) and whether the turn was judged hard (`hard`), or the helper model that answers (`name`) |
| `delta` | `{text}` | streaming assistant text |
| `reasoning` | `{text}` | streaming reasoning fragment |
| `rewrite` | `{text, reasoning}` | the model wrote reasoning as plain text ending in `</think>`, or a reply was corrected: replace the reply text so far with `text` and show `reasoning` as reasoning |
| `retract` | `{text}` | text shown earlier in this turn says the same as the final answer: remove that piece |
| `tool_call` | `{id, name, args, risk, requires_approval}` | before approval/execution |
| `approval` | `{approval_id, tool, args, summary, risk, timeout_s, always_scope: "tool"\|"chat", level, reasons}` | user action needed |
| `tool_result` | `{id, name, ok, duration_ms, preview}` | after execution; `preview` ≤ 400 chars |
| `steering` | `{text}` | a note sent with `POST /api/chat/steer` was added to the turn |
| `context` | `{used, budget, percent, compacted}` | after context work |
| `memory` | `{added, updated, removed}` | optional; extraction is asynchronous after `done`, so the UI refreshes counts from `/api/bootstrap` or `/api/memories` |
| `plan` | `{steps: [{n, text, kind: "answer"\|"web"\|"write"\|"run"\|"read"\|"other", done, target?}]}` | a request with several steps, or naming a place outside the workspace: after `meta`, then whenever a step finishes, and at the end |
| `title` | `{id, title}` | auto-title updated |
| `done` | `{message_id, aborted, model, usage:{prompt_tokens,completion_tokens,total_tokens}\|null, stats, inference:{ttft_ms, tokens_per_s, prompt_tokens_est, reused_tokens_est, reuse_percent, thinking, steered?, acted_instead?, adapters_turned_down?}\|null, plan?}` | end (`plan`: the final step statuses, when the turn had a plan) |
| `error` | `{message, code, recoverable}` | end/failure |

`brain` comes before the first text of the reply. `skill` is the skill Kiwi Mind chose for the
turn (`chat` when it has no better answer; `null` or `""` when the helper model leads it), and
`hard` is whether the turn was judged hard, so a helper model may answer. A turn can emit more
than one `brain` event, for example when a hard turn passes to the helper.

`done.inference` holds the timing and reuse of the turn's last model request:
`ttft_ms` (time to first token), `tokens_per_s`, the estimated prompt size and how much of it
the server reused from its cache (`prompt_tokens_est`, `reused_tokens_est`, `reuse_percent`),
and whether the model thought (`thinking`).

- `steered`: the kinds of correction made to a reply in this turn, from Kiwi Sense:
  `other_identity`, `adopted_user`, `denial`, `unbacked_claim`, `unkept_promise`, `meta_leak`,
  `off_topic` or `wrong_result`. Present only when a correction was made.
- `acted_instead`: `web_search` or `note`, when the model only talked about a search or a note
  and Kiwibro made it itself. Present only then.
- `adapters_turned_down`: task adapters that the turn's replies were judged to spoil, as
  `{name, trouble, strength}`; `trouble` counts such replies and `strength` is the share of
  the adapter's scale it keeps now (`1.0`, `0.5` or `0.0`). Present only when one was.

Tool-call loop happens inside this one stream: after each tool result the model is
called again; assistant messages produced along the way are persisted and the final
`done` references the last assistant message id. Client may disconnect at any time;
server aborts the turn (partial text is saved).

### `POST /api/attachments`
`multipart/form-data` with one field `file` (≤ 25 MB). Stores it as
`<workspace>/attachments/<YYYY-MM-DD>/<8 hex characters>-<safe name>` so every tool can reach
it. `200 {"ok": true, "attachment": {"path", "name", "kind": "image|audio|video|svg|other", "size"}}`
(`path` is relative to the workspace); `400 bad_attachment` for empty or oversized files;
`500 storage` when the file cannot be written.

### `POST /api/chat/stop`
Body `{"conversation_id": "c_..."}` → `{"ok": true}`. Idempotent.

### `POST /api/chat/steer`
Body `{"conversation_id": "c_...", "message": "..."}`. Queues a steering note into the running
turn; the note is saved as a user message and an SSE `steering` event is emitted when the turn
takes it, at its next round. `200 {"ok": true, "queued": true}`; `409 not_busy` when no reply
is running for that chat; `404 not_found` for an unknown conversation; `400 bad_request` for an
empty message.

### `GET /api/turns`
`{"ok": true, "active_count": 1, "conversations": ["c_..."]}`: the chats with a reply being
generated now.

### `POST /api/approvals/<approval_id>`
Body `{"decision": "allow" | "always" | "deny"}` → `{"ok": true, "decision": "..."}`.
`always` follows the approval's `always_scope`: in Manual mode (`autonomy: supervised`,
scope `tool`) it sets that tool's permission to `auto`; in the automatic modes (scope
`chat`) it allows that tool's risky steps for the rest of the conversation (kept in
memory; critical steps stay refused). 404 if unknown/expired. When resolving, the
blocked SSE stream resumes.

---

## Memories

### `GET /api/memories?q=&kind=`
```json
{"ok": true, "memories": [{"id":"mem_...","kind":"preference","content":"Prefers dark themes",
 "pinned": false, "confidence": 1.0, "origin": "you", "source_conversation_id": null,
 "created_at":"...","updated_at":"...","last_used_at":null}],
 "total": 12}
```
Order: pinned, then updated_at desc. `q` = FTS search, `kind` filter optional (one of
`fact`, `preference`, `project`, `person`, `decision`; else `400`).
`origin`: `you` (added or edited in the Memory sheet), `chat` (saved by the model's
`remember` tool in `source_conversation_id`), `learned` (background extraction from
`source_conversation_id`), or `null` for memories saved before the origin was recorded.
The model never deletes a `you` or pinned memory, and pruning at the 400-memory cap never
evicts them.

### `POST /api/memories`
Body `{"content": "...", "kind": "fact"}` → `{"ok": true, "memory": {...}}` (origin `you`;
adding a near-duplicate of a learned memory merges into it and makes it yours).

### `PATCH /api/memories/<id>`
Body any of `{"content", "kind", "pinned"}` → `{"ok": true, "memory": {...}}`. Editing the
content makes the memory the user's (origin `you`).

### `DELETE /api/memories/<id>` → `{"ok": true}`

### `DELETE /api/memories` → `{"ok": true, "deleted": 12}` (forget all)

---

## Notes

Notes are longer pages than memories: a title, a markdown body and tags. The model keeps
notes with the `note` tool, and the Notes panel and this API read and write the same ones.

### `GET /api/notes?q=&tag=`
```json
{"ok": true, "notes": [{"id": "n_...", "title": "Packing list", "body": "- passport",
 "tags": ["travel"], "pinned": false, "origin": "you", "source_conversation_id": null,
 "created_at": "...", "updated_at": "..."}],
 "total": 1, "tags": [{"tag": "travel", "count": 1}]}
```
Order: pinned first, then `updated_at` desc. `q` is a full-text search over title, body and
tags; `tag` keeps the notes with that tag. `tags` lists every tag in use with its count, most
used first. `origin` is `you` or `kiwibro` (saved in a chat, by the model's `note` call or by
Kiwibro when the model did not make it); `source_conversation_id` is the chat a note came
from, if any.

### `POST /api/notes`
Body `{"body": "...", "title"?: "...", "tags"?: ["travel"] | "travel, lisbon"}`. `body` is
required (at most 20,000 characters). A missing title is taken from the first non-empty line
of the body (at most 60 characters). A given title is at most 120 characters. Tags are
lower-case, letters, digits, `-` and `_` only, at most 8. Returns `{"ok": true, "note": {...}}`;
`400 bad_request` for an empty or too long body.

### `GET /api/notes/<id>`
`{"ok": true, "note": {...}}`; `404 not_found` for an unknown id.

### `PATCH /api/notes/<id>`
Body any of `{"title", "body", "tags", "pinned"}` (`pinned` must be a boolean) →
`{"ok": true, "note": {...}}`. `404 not_found` for an unknown id.

### `DELETE /api/notes/<id>` → `{"ok": true}`

---

## Tasks and terminals

### `GET /api/tasks?conversation_id=&status=`
`{"ok": true, "tasks": [...]}`. The task board of a chat, as the `task_board` tool keeps it.

### `POST /api/tasks`
Body `{"action": "add", "conversation_id": "c_...", ...}`. `action` (default `add`) is `list`,
`add`, `update`, `delete`, `clear` or `summary`. The other fields are those of the `task_board`
tool: `title`, `notes`, `priority`, `depends_on` (add); `task_id`, `status`, `title`, `notes`,
`priority` (update); `task_id` (delete); `only_done` (clear). Returns `{"ok": ..., "tasks": [...],
"summary": "..."}`; `400` when `ok` is false (for example, a task id that does not exist).

### `GET /api/terminals?conversation_id=`
`{"ok": true, "terminals": [...]}`: the terminal sessions of a chat, newest first, without
their output.

### `POST /api/terminals/<session_id>/kill`
Stops one session: `{"ok": true, "terminal": {...}}`; `404 not_found` for an unknown session.

---

## Safe copy

A chat can open a safe copy of a folder: the agent then works in a copy kept in Kiwibro's data
folder (`<data>/safecopy/<session id>`), so no tool touches the real folder until the user
applies a reviewed proof. While a session is open the chat's file and command tools work in
the copy, commands run in the sandbox without network, and tools that act on the whole
computer (`open_path`, `clipboard`, `screenshot`, `remote`, `notify`, `system_info`, `disk_usage`,
`workspace`, `terminal_session`, `delegate` and third-party tools) are refused. Sessions are
opened, read and closed with the routes below; a chat remembers its own session, which is not
a field of the conversation API.

### `POST /api/safecopy/start`
Body `{"conversation_id": "c_...", "folder": "/home/you/project"}`. Copies the folder and binds
the session to the chat. A chat that already has a session gets it back unchanged, with
`"started": false`.
```json
{"ok": true, "started": true, "active": true, "real": "/home/you/project",
 "work": "/home/you/.local/share/kiwibro/safecopy/sc-1a2b3c4d5e6f/work", "changed_files": 0,
 "disk_bytes": 1048576, "left_out": [], "applied": false, "can_undo": false}
```
The fields are those of `GET /api/safecopy/status`.
`400 bad_request` without a conversation id or a folder; `400 not_a_folder` when the path is
not a folder; `400 refused_folder` for the filesystem root, the home folder itself or
Kiwibro's data folder; `404 not_found` for an unknown chat.

### `GET /api/safecopy/status?conversation_id=c_...`
Where the copy is and what waits.
```json
{"ok": true, "active": true, "real": "/home/you/project",
 "work": "/home/you/.local/share/kiwibro/safecopy/sc-1a2b3c4d5e6f/work", "changed_files": 2,
 "disk_bytes": 1048576, "left_out": [{"path": "big.bin", "reason": "..."}],
 "applied": true, "can_undo": true}
```
`real` is the folder that was copied, `work` the copy the tools work in. `changed_files` counts
the files that differ from the copy's first state and `disk_bytes` is the copy's disk use.
`left_out` lists what the copy could not hold, each entry `{path, reason}`. `applied` says
something was written to the real folder in this session and `can_undo` whether the last apply
can still be undone. A chat without a session answers `active: false` with `real` and `work`
`null`, `0` for the numbers, `false` for `applied` and `can_undo`, and an empty `left_out`.
`400 bad_request` without `conversation_id`.

### `GET /api/safecopy/proof?conversation_id=c_...`
What the copy changed, for the page that asks the user to approve it.
```json
{"ok": true, "proof": {
  "changes": [{"path": "notes.txt", "kind": "modified", "added_lines": 1, "removed_lines": 1,
               "binary": false}],
  "totals": {"files": 1, "added": 0, "modified": 1, "deleted": 0, "added_lines": 1, "removed_lines": 1},
  "diff": "diff --git a/notes.txt b/notes.txt\n...",
  "conflicts": [],
  "left_out": [{"path": "big.bin", "reason": "..."}],
  "removals": [{"path": "old.txt", "command": "rm -- /home/you/project/old.txt"}],
  "before_commit": "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
  "revert": {"command": "...", "paths": ["notes.txt"],
             "added": [{"path": "new.txt", "command": "rm -- /home/you/project/new.txt"}],
             "verified": true, "files_checked": 1, "real_is_clean_git": false},
  "fingerprint": "..."}}
```
`changes` lists every path that differs from the copy's first state, sorted by path; `kind` is
`added`, `modified` or `deleted`. `totals` adds those counts up with the line counts. `diff` is
the unified diff of the change (empty when there is none). `conflicts` lists paths the user
changed after the copy was made; `apply` refuses while it is not empty. `left_out` lists
changes that are never written, each `{path, reason}`. `removals` lists the files deleted in
the copy, each with the command the user may run: Kiwibro never deletes in the real folder.
`before_commit` names the saved state of the overwritten files. `revert` is the undo the proof
was rehearsed with: `command` (a shell command that brings the overwritten files back),
`paths` (those files), `added` (the files the apply creates, each with its remove command),
`verified` (the undo was rehearsed on a copy), `files_checked`, `real_is_clean_git` (the real
folder is a git tree with nothing pending) and, when that is true and files are overwritten,
`git_restore`. `fingerprint` names this exact proof.

`400 bad_request` without `conversation_id`; `404 not_found` when the chat has no session;
`400 git` when the copy has no record of its first state; `400 revert_unproven` when the undo
did not bring the old files back, so no proof is made.

### `POST /api/safecopy/apply`
Body `{"conversation_id": "c_...", "fingerprint": "<proof.fingerprint>"}`. Writes the added and
modified files of that proof into the real folder, never follows a link out of it, and adds an
assistant message to the chat with the undo command and the removals to run.
```json
{"ok": true, "applied": {"written": ["notes.txt"], "removals": [{"path": "old.txt", "command": "..."}],
 "revert": {...}, "before_commit": "...", "applied_commit": "...",
 "left_out": [{"path": "...", "reason": "..."}]}}
```
`409 stale` when the copy changed after the proof was made or the fingerprint is missing:
nothing is written and the proof must be read again. `409 conflict` when a file in the real
folder changed after the copy was made or got in the way while writing: nothing is written.
`409 partial` when the write failed halfway: the files named in the message were written and
`undo` puts them back. `400 nothing_to_apply` when the proof has nothing to write.
`400 bad_request` without a conversation id; `404 not_found` when the chat has no session;
`400 git` or `400 revert_unproven` as for the proof.

### `POST /api/safecopy/undo`
Body `{"conversation_id": "c_..."}`. Puts the files the last apply overwrote back and adds a
chat message about what stayed. Files that apply added are kept in the real folder and come
back as commands for the user to run.
```json
{"ok": true, "undo": {"restored": ["notes.txt"],
 "kept": [{"path": "new.txt", "command": "rm -- /home/you/project/new.txt"}], "before_commit": "..."}}
```
`404 not_found` when the chat has no session or nothing was applied in it; `409 conflict` when
a written file changed after the apply, so nothing is undone; `400 revert_unproven` when a file
did not come back; `400 bad_request` without a conversation id.

### `POST /api/safecopy/end`
Body `{"conversation_id": "c_...", "discard": false}`. Detaches the chat from its session and
gives it back the workspace it had before; `discard: true` also removes the copy's folder from
disk. `discard` must be a boolean.
```json
{"ok": true, "active": false, "discarded": true}
```
`400 bad_request` without a conversation id or when `discard` is not a boolean; `404 not_found`
when the chat has no session.

These routes answer a safe copy failure with the general error body and the status of its code:
`not_found` 404, `stale`, `conflict` and `partial` 409, every other code 400.

---

## Settings

### `GET /api/settings`
```json
{"ok": true, "settings": {
  "model": null, "server_url": "", "agent_enabled": true,
  "memory_enabled": true, "sense_enabled": true, "max_output_tokens": 2048, "temperature": 0.6,
  "workspace": "/home/you/Kiwibro", "user_name": "",
  "tool_profile": "auto", "reasoning": "auto", "max_tool_rounds": 16,
  "autonomy": "smart", "sandbox": "auto", "remote_hosts": [],
  "direct_runtime": "", "direct_models_dir": "", "direct_threads": 0, "direct_gpu": "auto",
  "permissions": {"web_search":"auto", "...every tool, see config.permission_defaults()":"…"}
}, "data_dir": "/home/you/.local/share/kiwibro"}
```
`server_url` is the attached server's address from `KIWIBRO_SERVER_URL` (`""` in local mode);
it is read-only here. Settings keys stored by older versions (the retired external-server address,
token and engine choice) are left out of responses.

### `PATCH /api/settings`
Body = subset of settings. Validate: temperature 0–2, max_output_tokens 256–8192,
tool_profile in {auto, full, minimal, coding, research, desktop}, reasoning in
{auto, on, off}, autonomy in {supervised, smart, autonomous}, sandbox in {auto,
strict, off}, remote_hosts a list of ≤32 ssh aliases or `user@host`, max_tool_rounds
1–32, workspace must be an absolute path that can be created inside `$HOME` (no root paths),
user_name ≤ 80 characters, `model` a string or null, agent_enabled / memory_enabled /
sense_enabled booleans, permissions modes in {auto, ask, deny} for any known tool (built-in
or extension), direct_runtime `""` or an absolute path to an executable `llama-server`,
direct_models_dir `""` or an existing absolute folder, direct_threads 0–256 (0 = every
physical core), direct_gpu in {auto, off}. Any other key, `server_url` included, is `400`.
Changing direct_runtime, direct_models_dir, direct_threads or direct_gpu stops the running
llama.cpp servers (the next request starts one with the new values). Returns merged
settings. Changing workspace does not move files; a conversation bound to its own
folder (`conversation.workspace`, set by the `workspace` tool) keeps using that folder.

Per-tool `permissions` apply in Manual mode (`autonomy: supervised`) only; `smart` and
`autonomous` use each tool's default and the risk policy decides each call.

### `GET /api/workspace`
`{"ok": true, "root": "/home/you/Kiwibro", "files": [{"path":"notes.md","type":"file","size":123,"modified":"..."}]}`
(depth 2, ≤200 entries, no dotfiles or dot-folders). Used by the settings sheet only.

---

## Tooling notes for implementers

- All DB writes happen through `kiwibro/db.py` helpers; timestamps `datetime.now(timezone.utc).isoformat()`.
- IDs: `c_<uuid4hex[:16]>`, `m_<uuid4hex[:16]>`, `mem_<uuid4hex[:16]>`, approval ids `ap_<uuid4hex>`.
  `seq` is `max(seq)+1` per conversation, allocated under the conversation lock.
- Auto title: deterministic first (derive 2–6 words from the first user message,
  stripping request/tool noise), then optionally accept a model-written title only if
  it validates as a title (≤40 chars, 2–6 words, no meta-instruction echo); fall back
  to the deterministic title. Emit `title`. This keeps `done` fast.
- Memory extraction runs in a daemon thread after `done`, so the stream closes
  immediately and latency stays low. It uses a short timeout and must never fail the
  turn. The stream's optional `memory` event is therefore not guaranteed; clients
  refresh memory counts from `/api/bootstrap` (or `/api/memories`) after a turn,
  when opening the memory sheet, or on window focus.
