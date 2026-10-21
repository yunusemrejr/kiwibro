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
{"ok": true, "version": "1.0.0", "lmstudio": true, "lmstudio_url": "http://127.0.0.1:1234", "model": "qwen3.8-2b"}
```

### `GET /api/bootstrap`
Aggregate for first paint.
```json
{
  "health": {"ok": true, "version": "1.0.0", "lmstudio": true, "lmstudio_url": "..."},
  "settings": {...},                       // see settings
  "models": [{"id": "qwen3.8-2b", "name": "qwen3.8-2b", "type": "llm",
              "state": "loaded", "context": 32768, "max_context": 262144,
              "tools": true, "quantization": "Q4_K_M"}],
  "current_model": "qwen3.8-2b",
  "conversations": [{"id": "...", "title": "...", "updated_at": "...", "message_count": 4, "model": "..."}],
  "memories_count": 12,
  "workspace": "/home/you/Kiwibro"
}
```
If LM Studio is offline: `health.lmstudio=false`, `models=[]`, still 200.

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

### `POST /api/models/select`
Body `{"id": "qwen3.8-2b"}`. Persists the model in settings; does **not** touch
conversations or memories. Response `{"ok": true, "current": "..."}`. 400 if model
unknown. Selection applies to new turns in any chat.

---

## Conversations

### `GET /api/conversations`
```json
{"ok": true, "conversations": [
  {"id": "c_...", "title": "Kiwi facts", "model": "qwen3.8-2b", "pinned": false,
   "message_count": 6, "created_at": "...", "updated_at": "..."}]}
```
Order: pinned first, then `updated_at` desc.

### `POST /api/conversations`
Body `{"title": "..."}` (optional). Returns full conversation object
`{"id","title","model","created_at","updated_at","summary":null,"messages":[]}`.

### `GET /api/conversations/<id>`
```json
{"ok": true, "conversation": {"id","title","model","created_at","updated_at",
  "summary": "..." | null, "compacted": true},
 "messages": [{"id":"m_...","seq":1,"role":"user|assistant|tool","content":"...",
   "reasoning": "", "tool_calls": [...], "tool_name": null, "model": "...",
   "created_at": "...", "meta": {}}],
 "context": {"used": 1234, "budget": 30000, "percent": 4}}
```
404 unknown id.

### `PATCH /api/conversations/<id>`
Body `{"title"?: str, "pinned"?: bool}` → `{"ok": true, "conversation": {...}}`.

### `DELETE /api/conversations/<id>`
`{"ok": true}`. Deletes messages via cascade; does not delete memories.

### `POST /api/conversations/<id>/clear`
Delete messages and summary of one conversation, keep the conversation.
`{"ok": true, "conversation": {...}}`.

---

## Chat (SSE)

### `POST /api/chat`
Body:
```json
{"conversation_id": "c_...", "message": "hello", "model": "qwen3.8-2b" /* optional */}
```
- If `conversation_id` is null/missing, create a conversation and use it.
- If a turn is active for that conversation → `409 {"error":{"code":"busy"}}`.
- Response `200 text/event-stream`, headers: `Cache-Control: no-cache`,
  `X-Accel-Buffering: no`, `Connection: keep-alive`. Heartbeat comment `: ping`
  every 15 s while idle (e.g. waiting on approval).
- Validation: message non-empty, ≤ 32000 chars → 400.

SSE frames use `event: <type>\ndata: <json>\n\n`.

| event | data | when |
|---|---|---|
| `meta` | `{conversation_id, user_message_id, model, title, context:{used,budget,percent}}` | first |
| `status` | `{phase: "thinking"\|"web"\|"tool"\|"reading"\|"writing"\|"running", label}` | optional, for activity line |
| `delta` | `{text}` | streaming assistant text |
| `reasoning` | `{text}` | streaming reasoning fragment |
| `tool_call` | `{id, name, args, risk, requires_approval}` | before approval/execution |
| `approval` | `{approval_id, tool, args, summary, risk, timeout_s}` | user action needed |
| `tool_result` | `{id, name, ok, duration_ms, preview}` | after execution; `preview` ≤ 400 chars |
| `context` | `{used, budget, percent, compacted}` | after context work |
| `memory` | `{added, updated, removed}` | optional; extraction is asynchronous after `done`, so the UI refreshes counts from `/api/bootstrap` or `/api/memories` |
| `title` | `{id, title}` | auto-title updated |
| `done` | `{message_id, aborted, model, usage:{prompt_tokens,completion_tokens,total_tokens}\|null}` | end |
| `error` | `{message, code, recoverable}` | end/failure |

Tool-call loop happens inside this one stream: after each tool result the model is
called again; assistant messages produced along the way are persisted and the final
`done` references the last assistant message id. Client may disconnect at any time;
server aborts the turn (partial text is saved).

### `POST /api/chat/stop`
Body `{"conversation_id": "c_..."}` → `{"ok": true}`. Idempotent.

### `POST /api/approvals/<approval_id>`
Body `{"decision": "allow" | "always" | "deny"}` → `{"ok": true, "decision": "..."}`.
`always` sets that tool's permission mode to `auto`. 404 if unknown/expired.
When resolving, the blocked SSE stream resumes.

---

## Memories

### `GET /api/memories?q=&kind=`
```json
{"ok": true, "memories": [{"id":"mem_...","kind":"preference","content":"Prefers dark themes",
 "pinned": false, "confidence": 1.0, "created_at":"...","updated_at":"...","last_used_at":null}],
 "total": 12}
```
Order: pinned, then updated_at desc. `q` = FTS search, `kind` filter optional.

### `POST /api/memories`
Body `{"content": "...", "kind": "fact"}` → `{"ok": true, "memory": {...}}`.

### `PATCH /api/memories/<id>`
Body any of `{"content", "kind", "pinned"}` → `{"ok": true, "memory": {...}}`.

### `DELETE /api/memories/<id>` → `{"ok": true}`

### `DELETE /api/memories` → `{"ok": true, "deleted": 12}` (forget all)

---

## Settings

### `GET /api/settings`
```json
{"ok": true, "settings": {
  "model": null, "lmstudio_url": "http://127.0.0.1:1234", "agent_enabled": true,
  "memory_enabled": true, "max_output_tokens": 2048, "temperature": 0.6,
  "workspace": "/home/you/Kiwibro", "user_name": "",
  "permissions": {"web_search":"auto","fetch_page":"auto","read_file":"auto",
    "list_files":"auto","write_file":"ask","run_command":"ask","remember":"auto"}
}}
```
### `PATCH /api/settings`
Body = subset of settings. Validate: temperature 0–2, max_output_tokens 256–8192,
workspace must be creatable inside `$HOME` (no root paths), lmstudio_url http(s),
permissions modes in {auto, ask, deny}. Returns merged settings. Changing workspace
does not move files.

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
