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
{"ok": true, "version": "1.1.0", "started_at": 1761000000, "lmstudio": true,
 "lmstudio_url": "http://127.0.0.1:1234", "model": "qwen3.8-2b"}
```
`/health` also carries the runtime fields of `/api/version` below; `started_at` is when
the current server image started (Unix seconds, renewed by a self-update).

### `GET /api/version`
Cheap poll (never calls LM Studio): which code this process serves.
```json
{"ok": true, "version": "2.3.0", "started_at": 1761000000, "pid": 4242, "busy": false,
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
  "health": {"ok": true, "version": "1.1.0", "lmstudio": true, "lmstudio_url": "..."},
  "settings": {...},                       // see settings
  "models": [{"id": "qwen3.8-2b", "name": "qwen3.8-2b", "type": "llm",
              "state": "loaded", "context": 32768, "max_context": 262144,
              "tools": true, "quantization": "Q4_K_M"}],
  "current_model": "qwen3.8-2b",
  "conversations": [{"id": "...", "title": "...", "updated_at": "...", "message_count": 4, "model": "..."}],
  "memories_count": 12,
  "workspace": "/home/you/Kiwibro",
  "readiness": {"model": "qwen3.8-2b", "phase": "warming", "detail": "Getting ready", "context_length": 32768}
}
```
If LM Studio is offline: `health.lmstudio=false`, `models=[]`, still 200. With LM Studio
reachable, bootstrap also queues the selected model's load + warm-up (see
`GET /api/models/ready`); the response never waits for it.

---

## Models

### `GET /api/models`
```json
{"ok": true, "lmstudio": true, "current": "qwen3.8-2b",
 "models": [{"id": "...", "name": "...", "type": "llm|vlm", "state": "loaded|not-loaded",
             "context": 32768, "max_context": 262144, "tools": true,
             "publisher": "...", "quantization": "..."}]}
```
`context` = `loaded_context_length` if loaded else `min(max_context_length, 32768)`.
Embedding models are excluded. `tools` = capabilities include `tool_use`.
Additive fields from LM Studio's `/api/v1/models` when available: `key`, `display_name`,
`arch`, `params`, `size_bytes`, `quant_bits`, `vision`, `reasoning` (allowed options, e.g.
`["off","on"]`), `reasoning_default`, and for loaded models `loaded_config`
(`context_length`, `parallel`, `flash_attention`) plus `context_choices` (larger windows
the model supports, at most three).

### `GET /api/engine`
Which engine serves models and what the direct engine found:
```json
{"ok": true, "engine": "lmstudio|direct", "direct": {
  "available": true, "error": null, "models_count": 7, "threads": 6,
  "gpu": "AMD Radeon Graphics (RADV RENOIR)", "gpu_available": true,
  "runtime": {"path": ".../llama.cpp-linux-x86_64-vulkan-avx2-2.54.0/llama-server",
              "label": "LM Studio's llama.cpp 2.54.0 (VULKAN)", "kind": "vulkan", "source": "lmstudio"},
  "runtimes": [{"path": "...", "label": "..."}],
  "model_dirs": ["/home/you/.lmstudio/models"],
  "servers": [{"model": "bartowski/spark-x2.6-4b", "pid": 123, "port": 45885,
               "context_length": 32768, "parallel": 4, "threads": 6,
               "gpu": "AMD Radeon Graphics (RADV RENOIR)",
               "runtime": "...", "started_at": 1793018131, "busy": false}]}}
```
`gpu` is the GPU the next model would run on (`""` = CPU only, or graphics acceleration
off); `gpu_available` says whether any llama.cpp build found here sees a hardware GPU. A
server's `gpu` is `""` when it runs on the CPU.

In direct mode every model endpoint above keeps its shape: `lmstudio` means "the engine
can list models", models come from the GGUF files on disk (`state: loaded` while a
server runs), and `/api/models/context` restarts the server with the new window.
`/health` and `/api/bootstrap` include `engine`.

### `POST /api/engine/stop`
Stops every direct-engine server (frees its memory; the next message starts one again).
`{"ok": true, "direct": {...}}`; `409 busy` while a reply runs.

### `GET /api/models/ready`
`{"ok": true, "readiness": {"model", "phase", "detail", "context_length"}}`. `phase` is
`idle | loading | warming | ready | error`. Opening Kiwibro, selecting a model, opening a
chat and changing a prompt-shaping setting queue a background job: the model is loaded
explicitly with a working window (32K, or 16K/8K when LM Studio's guardrails refuse; a
loaded model below 16K is reloaded larger while no reply runs) and the exact leading
messages + tools of the next request are prefilled with `max_tokens: 1`, so the first
reply only processes the new message. A reply that starts during the job waits for it
(status `Getting the model ready`) when it is the same prefix, and cancels it otherwise.
`/api/bootstrap`, `/api/models` and `/api/models/select` include the same `readiness`
object. Disabled with `KIWIBRO_WARMUP=0` (the e2e harness does this).

### `POST /api/models/context`
Body `{"id": "spark-x2.6-4b", "context_length": 65536}` (one of the model's
`context_choices`). Reloads the loaded model with that window (LM Studio cannot resize
in place). `200 {"ok": true, "current": "<instance id>", "context_length": 65536}`;
`409 no_memory` when LM Studio refuses the larger window — the previous window is
loaded again; `409 busy` while a reply runs.

### `POST /api/models/select`
Body `{"id": "qwen3.8-2b"}`. Persists the model in settings; does **not** touch
conversations or memories. Response `{"ok": true, "current": "...", "readiness": {...}}`
(the load + warm-up runs in the background). 400 if model unknown. Selection applies to
new turns in any chat. Models shared over LM Link come through the local server's list
like local ones and are selected, loaded and warmed the same way.

---

## Conversations

### `GET /api/conversations`
```json
{"ok": true, "conversations": [
  {"id": "c_...", "title": "Kiwi facts", "model": "qwen3.8-2b", "pinned": false,
   "message_count": 6, "created_at": "...", "updated_at": "..."}]}
```
Order: pinned first, then `updated_at` desc.

### `GET /api/conversations/search?q=<text>`
```json
{"ok": true, "matches": [{"id": "c_...", "snippet": "...the Lisbon trip budget is 900 euros..."}]}
```
Chats whose messages match `q` (full-text; function words are ignored), best match
first, one entry per chat, at most 30. The snippet is a short cut around the match
with the common markdown marks removed. `q` shorter than 2 characters returns
`"matches": []`. Titles are not searched here: the sidebar filter matches them itself.

### `POST /api/conversations`
Body `{"title": "...", "private": false}` (both optional; `private` must be a boolean,
else `400`). Returns full conversation object
`{"id","title","model","created_at","updated_at","summary":null,"messages":[]}`.

### `GET /api/conversations/<id>`
```json
{"ok": true, "conversation": {"id","title","model","created_at","updated_at",
  "workspace": "/home/you/Desktop/app" | null, "summary": "..." | null, "compacted": true},
 "messages": [{"id":"m_...","seq":1,"role":"user|assistant|tool","content":"...",
   "reasoning": "", "tool_calls": [...], "tool_name": null, "model": "...",
   "created_at": "...", "meta": {}}],
 "context": {"used": 1234, "budget": 30000, "percent": 4}, "busy": false}
```
`busy` is true while a reply is being generated for that chat; the conversation object
includes `"private": bool`. Opening a chat queues a prefill of its history for the next
turn; `?warm=0` reads it without that (an export). 404 unknown id.

### `PATCH /api/conversations/<id>`
Body `{"title"?: str, "pinned"?: bool, "private"?: bool}` → `{"ok": true, "conversation": {...}}`.
A private chat is never saved to memory and never shown to other chats.

### `DELETE /api/conversations/<id>`
`{"ok": true}`. Deletes messages via cascade; does not delete memories.

### `POST /api/conversations/<id>/clear`
Delete messages and summary of one conversation, keep the conversation.
`{"ok": true, "conversation": {...}}`.

### `POST /api/conversations/<id>/fork`
Body `{"seq": N}` (optional; a non-negative integer, else `400 bad_request`). Copies the
messages with `seq <= N` (all when omitted) into a new chat, in one transaction, keeping
their seqs. The copy keeps the workspace and the model, and its extraction watermark is
clamped to `N`. The summary and its segments are kept only when `N` reaches the summary
boundary (`summary_through_seq`); otherwise they are left out. The title is
`"<title> (fork)"` and is not re-titled automatically; a fork of a private chat is private.
Returns `201`
`{"ok": true, "conversation": {...}}`. Unknown id: `404 not_found`.

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
- Validation: message non-empty, ≤ 400,000 chars → 400. A message longer than about 40 %
  of the model's window is stored in the document library; the model reads its start,
  its end and the handle to read the rest (`doc_read` / `doc_search`). The transcript
  keeps the full text.

SSE frames use `event: <type>\ndata: <json>\n\n`.

| event | data | when |
|---|---|---|
| `meta` | `{conversation_id, user_message_id, model, title, context:{used,budget,percent}}` | first |
| `status` | `{phase: "thinking"\|"web"\|"tool"\|"reading"\|"writing"\|"running", label}` | optional, for activity line |
| `delta` | `{text}` | streaming assistant text |
| `reasoning` | `{text}` | streaming reasoning fragment |
| `rewrite` | `{text, reasoning}` | the model wrote reasoning as plain text ending in `</think>`: replace the reply text so far with `text` and show `reasoning` as reasoning |
| `tool_call` | `{id, name, args, risk, requires_approval}` | before approval/execution |
| `approval` | `{approval_id, tool, args, summary, risk, timeout_s, always_scope: "tool"\|"chat", level, reasons}` | user action needed |
| `tool_result` | `{id, name, ok, duration_ms, preview}` | after execution; `preview` ≤ 400 chars |
| `context` | `{used, budget, percent, compacted}` | after context work |
| `memory` | `{added, updated, removed}` | optional; extraction is asynchronous after `done`, so the UI refreshes counts from `/api/bootstrap` or `/api/memories` |
| `plan` | `{steps: [{n, text, kind: "answer"\|"web"\|"write"\|"run"\|"read"\|"other", done, target?}]}` | a request with several steps, or naming a place outside the workspace: after `meta`, then whenever a step finishes, and at the end |
| `title` | `{id, title}` | auto-title updated |
| `done` | `{message_id, aborted, model, usage:{prompt_tokens,completion_tokens,total_tokens}\|null, stats, inference:{ttft_ms, tokens_per_s, prompt_tokens_est, reused_tokens_est, reuse_percent, thinking}\|null, plan?}` | end (`plan`: the final step statuses, when the turn had a plan) |
| `error` | `{message, code, recoverable}` | end/failure |

Tool-call loop happens inside this one stream: after each tool result the model is
called again; assistant messages produced along the way are persisted and the final
`done` references the last assistant message id. Client may disconnect at any time;
server aborts the turn (partial text is saved).

### `POST /api/attachments`
`multipart/form-data` with one field `file` (≤ 25 MB). Stores it as
`<workspace>/attachments/<YYYY-MM-DD>/<id>-<safe name>` so every tool can reach it.
`200 {"ok": true, "attachment": {"path", "name", "kind": "image|audio|video|svg|other", "size"}}`;
`400 bad_attachment` for empty or oversized files.

### `POST /api/chat/stop`
Body `{"conversation_id": "c_..."}` → `{"ok": true}`. Idempotent.

### `POST /api/chat/steer`

Body: `{"conversation_id": "c_…", "message": "…"}`. Queues a steering note into the
running turn; the worker persists it as a user message and emits an SSE `steering`
event at the next tool round. `200 {"ok": true, "queued": true}`, `409` when no
reply is running, `404` for unknown conversations.

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
Order: pinned, then updated_at desc. `q` = FTS search, `kind` filter optional.
`origin`: `you` (added or edited in the Memory sheet), `chat` (saved by the model's
`remember` tool in `source_conversation_id`), `learned` (background extraction from
`source_conversation_id`), or `null` for memories saved by versions before 2.1. The
model never deletes a `you` or pinned memory, and pruning at the 400-memory cap never
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

## Settings

### `GET /api/settings`
```json
{"ok": true, "settings": {
  "model": null, "lmstudio_url": "http://127.0.0.1:1234", "lmstudio_token_set": false,
  "agent_enabled": true,
  "memory_enabled": true, "max_output_tokens": 2048, "temperature": 0.6,
  "workspace": "/home/you/Kiwibro", "user_name": "",
  "tool_profile": "auto", "reasoning": "auto", "max_tool_rounds": 16,
  "autonomy": "smart", "sandbox": "auto", "remote_hosts": [],
  "engine": "lmstudio", "direct_runtime": "", "direct_models_dir": "", "direct_threads": 0,
  "direct_gpu": "auto",
  "permissions": {"web_search":"auto", "...every tool, see config.permission_defaults()":"…"}
}}
```
### `PATCH /api/settings`
Body = subset of settings. Validate: temperature 0–2, max_output_tokens 256–8192,
tool_profile in {auto, full, minimal, coding, research, desktop}, reasoning in
{auto, on, off}, autonomy in {supervised, smart, autonomous}, sandbox in {auto,
strict, off}, remote_hosts a list of ≤32 ssh aliases or `user@host`, max_tool_rounds
1–32, workspace must be creatable inside `$HOME` (no root paths), lmstudio_url
http(s), permissions modes in {auto, ask, deny} for any known tool (built-in or
extension), lmstudio_token a single token ≤ 512 chars (`""` removes it), engine in
{lmstudio, direct}, direct_runtime `""` or an absolute path to an executable
`llama-server`, direct_models_dir `""` or an existing absolute folder, direct_threads
0–256 (0 = every physical core), direct_gpu in {auto, off}. Switching to `direct`
unloads the models Kiwibro loaded in LM Studio; switching away, or changing a `direct_*` value, stops the direct servers
(the next request starts one with the new values). Returns merged
settings. Changing workspace does not move files; a conversation bound to its own
folder (`conversation.workspace`, set by the `workspace` tool) keeps using that folder.

`lmstudio_token` is write-only: it is sent as `Authorization: Bearer …` to an LM Studio
server with authentication on (`KIWIBRO_LMSTUDIO_TOKEN` works too) and is never
returned; responses carry `lmstudio_token_set` instead. Per-tool `permissions` apply in
Manual mode (`autonomy: supervised`) only; `smart` and `autonomous` use each tool's
default and the risk policy decides each call.

### `GET /api/workspace`
`{"ok": true, "root": "/home/you/Kiwibro", "files": [{"path":"notes.md","type":"file","size":123,"modified":"..."}]}`
(depth 2, ≤200, no dotdirs). Used by the settings sheet only.

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
