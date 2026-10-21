# Kiwibro — QA & Verification

Independent QA for Kiwibro v1. Contracts under test:
[ARCHITECTURE.md](ARCHITECTURE.md) and [API.md](API.md). This file is the evidence
log: what was run, what passed, what broke, and what is still residual risk.

**Status: COMPLETE — 2026-10-21 22:49 +0300.**

| Front | Result |
|---|---|
| Deterministic Python suite (unit + offline + API + chat/tools + compaction + no-tools) | **54/54 PASS** (final run on shipped code) |
| Browser rig, mock mode | **18/18 PASS** (9 checks + 7 flows; `.dev/qa/browser/shots/`) |
| Browser rig, fake-LM mode against the real Flask app | **18/18 PASS** (`.dev/qa/browser/fake-shots/`) |
| Browser rig, live mode (qwen3.8-2b) | checks 10/10 PASS; flows 5/7 live (hero, models, chat, memory, settings), approval/mobile completed from the mock-mode UI run (identical frontend) — see §6 |
| Live API/LM checks | 8 PASS, 3 model-behaviour WAIVED, 6 harness defects/inconclusive fixed post-run — see §6 |
| Committed screenshots | 7/7 in `docs/screenshots/`, inspected with vision |
| P0/P1 open bugs | **none** |

> Live verification used port **8799** + `.dev/qa-data` only; the canonical 8777
> server was never touched by QA.

---

## 1. Environment & guard rails

| Item | Value |
|---|---|
| Repo | `/home/yemre/Desktop/kiwibro` |
| Host Python | 3.14 system (`flask`, `requests`, `bs4`; no pytest dependency for the runner) |
| LM Studio | `http://127.0.0.1:1234`, model `qwen3.8-2b` (tool-capable, JIT load 30–120 s) |
| QA port | `8799` (canonical dev server 8777 is owned by Lead/backend) |
| Scratch paths | `.dev/qa/runtime/<phase>/{data,workspace,app.log}` |
| Browser | system Chrome 155 (`/usr/bin/google-chrome`), playwright-core in `.dev/qa/node_modules` |
| Data safety | every harness phase owns its own `KIWIBRO_DATA_DIR` / `KIWIBRO_WORKSPACE`; no test writes `~/.local/share/kiwibro` or `~/Kiwibro` |

Live calls are serialized (one turn at a time) because LM Studio is occasionally slow
under mixed load.

---

## 2. Deterministic API/SSE/security suite

The suite runs a **real `python3 app.py` child process** and a **fake LM Studio**
(`tests/e2e/fake_lm.py`) that speaks the same wire format: `/api/v0/models`,
`/v1/chat/completions` SSE (`text/event-stream`, no charset — like the real server),
delta/reasoning/tool-call fragments, `[DONE]`, usage chunks, scripted HTTP 400s.
No LM Studio and no internet are required.

```bash
# standalone (no pytest needed)
python3 tests/e2e/run_e2e.py                 # all phases
python3 tests/e2e/run_e2e.py --only chat     # one phase
python3 tests/e2e/run_e2e.py --list          # list checks

# from pytest (CI)
python3 -m pytest tests/e2e -q               # 1 test wrapping the suite
```

Evidence artifacts: `.dev/qa/runtime/evidence.json` + `evidence.md` (not committed;
regenerated on every run).

### Phases

| Phase | Checks | What it proves |
|---|---|---|
| `unit` | 9 | sandbox traversal/symlink/write-cap, command blocklist + timeout + output cap, SSRF guard (13 private targets), context budget math, memory dedupe/parse, DB/FTS sync + cascade, settings validation, LLM helpers, title derivation |
| `offline` | 4 | LM Studio down: `/health` graceful, bootstrap 200 models `[]`, `/api/models` 200, chat emits `meta` → `error{connection,recoverable}` in <0.1 s with no traceback |
| `api` | 12 | models list/select validation, conversation CRUD/pin/clear/cascade, memory CRUD + dedupe + FTS + hostile queries, settings range/type/whitelist validation, workspace listing, headers, Host/DNS-rebinding guard, error envelopes, static shell, restart persistence |
| `chat` | 24 | fake-LM streaming (text/markdown/reasoning/UTF-8/no-usage/empty), auto-title + prompt-echo rejection, async memory extraction, approval allow/always/deny, tool errors, `remember`, write/read traversal, SSRF, `sudo` blocklist, 409 busy + stop, client-disconnect abort + partial save, context-limit recovery, tools-unsupported retry, mid-chat model swap |
| `compaction` | 4 | tiny 2 K context: rolling compaction fires, transcript kept, summary injected into the next prompt, `meta` remains the first SSE event |
| `notools` | 1 | models without `tool_use` are called without tools |

### Evidence snapshot (latest run)

```text
unit 9/9 · offline 4/4 · api 12/12 · chat 24/24 · compaction 4/4 · notools 1/1
54/54 checks passed in 14.5 s
```

Notable assertions (full text in `evidence.md`):

- UTF-8 streaming: `Grüße — 世界 🥝 café ✓ 100%` survives a `text/event-stream`
  response with **no charset** (regression for the mojibake bug).
- Approval flow order: `tool_call → approval → tool_result → … → done`; `deny`
  leaves no file on disk; `always` upgrades the tool to `auto` across turns and
  conversations and only one approval card is shown.
- Security through the real tool pipeline: `write_file ../../pwned.txt` refused,
  `read_file ../../../etc/passwd` refused without leaking content,
  `fetch_page 169.254.169.254` refused before any socket, `run_command sudo rm -rf /`
  blocked even after Allow.
- Busy/abort: second turn on a streaming conversation → `409 busy`; stop emits
  `done{aborted:true}` and saves partial text; closing the SSE body server-side
  aborts the turn and frees the conversation.
- Recovery: LM Studio HTTP 400 context-limit → summarise → retry succeeds without
  surfacing an error; persistent 400s surface one calm `error` event and the
  conversation stays usable; a `tools unsupported` 400 is retried once without tools.
- Compaction: after compaction the summary text is part of the next request's system
  prompt (`## Earlier conversation summary`) — older context is not silently lost.

### Fake-LM scenarios

`tests/e2e/fake_lm.py` behaviour is selected by markers in the test prompt
(`[[scenario:plain]]`, `[[scenario:tool_write]]`, `[[scenario:context_400_once]]`, …).
Standalone mode for browser tests:

```bash
python3 tests/e2e/fake_lm.py --port 0 --print-url
```

Highlighted categories: plain/markdown/UTF-8/reasoning/no-usage/empty streams;
`write_file` allow/deny/always; `run_command` blocklist; read/write traversal;
SSRF `fetch_page`; unknown/bad-arg tools; `remember`; context-400 ×1/×2/always;
tools-unsupported retry; slow streams for stop/disconnect; non-stream title,
memory-extraction and summary answers.

---

## 3. Browser E2E rig (real Chrome)

Playwright-core drives headless Chrome against a spawned app (or an existing URL).

```bash
# validate the rig/UI without any backend state (in-browser mock backend)
node tests/e2e/browser/run_browser.mjs --mode mock --only checks,hero,chat --out .dev/qa/browser/shots

# full deterministic browser pass against the real app + fake LM
node tests/e2e/browser/run_browser.mjs --mode fake --out .dev/qa/browser/shots

# live pass against LM Studio (task-3; serialized)
node tests/e2e/browser/run_browser.mjs --mode live --out docs/screenshots

# or attach to an already-running server (8799)
node tests/e2e/browser/run_browser.mjs --mode live --base http://127.0.0.1:8799 --out docs/screenshots
```

Artifacts: screenshots + `browser-evidence.json` in the output dir; console errors,
page errors, failed requests and 4xx/5xx responses are recorded for every run.
Selectors are centralized in `tests/e2e/browser/selectors.mjs`.

### Browser checks

| Check | What it asserts |
|---|---|
| `console` / `console-after-flows` | zero console errors, zero uncaught page errors |
| `resources` | no failed requests, no 5xx |
| `contract` | `/api/bootstrap` shapes (health, models, settings.permissions, workspace, memories_count) and the header model pill match `current_model` |
| `xss` | 9 hostile markdown payloads (`<script>`, `<img onerror>`, `javascript:` links, `data:` URLs, inline-code HTML, svg/iframe) render inert: no execution, no dangerous tags, links only `http(s)` with `rel=noopener noreferrer` |
| `sse` | renders the real `consumeSSE` parser against a synthetic stream cut at arbitrary byte boundaries, including a UTF-8 emoji split, heartbeats and multi-line data |
| `a11y` | icon buttons/inputs/images labelled, `lang`, viewport + color-scheme meta, dialogs labelled, no positive tabindex, one `h1` |
| `shortcuts` | `Ctrl+N` new chat, `Ctrl+,` settings, `Esc` closes sheets, `Shift+Enter` newline (README keyboard table) |
| `no-external` | zero cross-origin requests — offline-capable, no CDNs/fonts |
| `responsive-desktop` / `responsive-mobile` | no horizontal overflow and no chrome past the viewport at 1440×900 and 390×844 |

Flows also *exercise* the UI, not just screenshot it: approval flow clicks Allow and
waits for the resolved card, model flow switches to another model and restores the
canonical one, memory flow adds/pins/deletes a row through the sheet, settings flow
toggles Agent mode off/on and verifies the change persisted through the API.

### Flows → committed screenshots

| Flow | File (task-3) | Scene |
|---|---|---|
| `hero` | `docs/screenshots/hero.png` | empty state, sidebar with history, model pill |
| `chat` | `docs/screenshots/chat.png` | streamed markdown answer + reasoning row + composer |
| `approval` | `docs/screenshots/approval.png` | inline write-file approval card (Allow/Always/Deny) |
| `memory` | `docs/screenshots/memory.png` | memory sheet with search, kinds, pinned rows |
| `settings` | `docs/screenshots/settings.png` | LM Studio status, model list, permissions, behaviour |
| `models` | `docs/screenshots/models.png` | model popover with state/context/tool markers |
| `mobile` | `docs/screenshots/mobile.png` | 390×844 conversation view (drawer closed) |

Rig validation (2026-10-21): mock mode **18/18**, fake-LM mode against the real
Flask app **18/18**; every committed screenshot was inspected with vision. See §6
for which committed shots are live vs mock-mode. `browser-evidence.json` is kept in
`.dev/qa/browser/`, never in `docs/screenshots/`.

---

## 4. Security probe matrix

| Attack | Guard | Expected | Result |
|---|---|---|---|
| `read_file ../../../etc/passwd` | workspace `resolve()` containment | refused, no content | PASS |
| `write_file ../../pwned.txt` | workspace containment (after Allow) | refused, nothing outside | PASS |
| symlink inside workspace → `/etc/passwd` | `resolve()` + `is_relative_to` | refused | PASS |
| `run_command sudo …` / `rm -rf /` / fork bomb / mkfs / dd to device | blocklist | refused even with approval | PASS |
| `fetch_page http://169.254.169.254/…` | SSRF guard (literal + DNS) | refused before socket | PASS |
| `fetch_page http://127.0.0.1…`, `file://`, `::1` | SSRF guard | refused instantly | PASS |
| `Host: evil.example` / `127.0.0.1.evil.example` | loopback Host allowlist | 403 `forbidden_host` | PASS |
| `Host: localhost:8799` / `[::1]:8799` | allowlist strips port | 200 | PASS |
| markdown `<script>` / `onerror` / `javascript:` / `data:` | escape-first renderer | inert output | PASS (real Chrome) |
| memory search `'; DROP TABLE …` / `!!!` / `AND OR` | quoted FTS + LIKE fallback | 200 JSON, no 500 | PASS |
| oversized JSON body (>4 MiB) | `MAX_CONTENT_LENGTH` | rejected, no 500 | PASS (413 `too_large`) |
| static path traversal `/static/../app.py` | Flask static handler | 404/403, never source | PASS |

---

## 5. Audit findings log

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | P1 | LM-offline chat crashed (`self.context.compute_budget` did not exist) → `error{internal}`, no `meta`, traceback in log, no calm offline state | **FIXED** (backend), verified by `offline/chat_offline_is_graceful` |
| 2 | P1 | Rolling-compaction summary stored but never injected into the prompt → compacted history silently lost | **FIXED** (backend `combine_system_prompt`), verified by `compaction/summary_reaches_the_model` |
| 3 | P2 | `meta` not the first SSE event on compacting turns (`context` was emitted before `meta`) | **FIXED**, verified by `compaction/meta_is_first_event` |
| 4 | P2 | Memory extraction became async but docs still promised a synchronous `memory` event; UI badge went stale | **RESOLVED BY DESIGN** — Lead amended `docs/API.md`/`ARCHITECTURE.md` (event optional; clients poll `/api/bootstrap`/`/api/memories`), frontend refreshes counts after turns and on focus; verified by `chat/memory_extraction` + browser contract check |
| 5 | P3 | SSE wire has duplicate/conflicting `Connection: keep-alive` + `Connection: close` (Werkzeug dev server always closes connections) | **ACCEPTED RESIDUAL** — app sets the contract header; Werkzeug appends `close`; browsers handle the close correctly. Only affects the dev server |
| 6 | P3 | `POST /api/models/select` accepts unknown model ids while LM Studio is offline (contract: 400 if unknown) | OPEN, low impact (the list is unknowable while offline) |
| 7 | P3 | Context-limit *recovery* compaction does not emit a `context` event, so the UI compaction chip is not shown (compaction itself works) | OPEN, cosmetic |
| 8 | P3 | `Workspace.list_files` followed symlinked dirs and could expose absolute outside paths in the listing | **FIXED** — escaping symlinks skipped, `public()` no longer leaks absolute paths |
| 9 | P3 | Per-conversation lock object was popped in `_cleanup`; a narrow race could create a second lock for the same conversation | **FIXED** — lock objects are kept for the process lifetime |
| 10 | P3 | `MAX_CONTENT_LENGTH` was silently `None`; >4 MiB bodies returned 400 instead of 413 | **FIXED** — 413 `too_large` verified |
| 11 | P3 | SSE `Cache-Control` was overwritten to `no-store` by the global API header middleware | **FIXED** — chat stream keeps `no-cache` |

Already-fixed Lead-reported issues re-verified by the deterministic suite:
UTF-8 mojibake in streamed text, auto-title prompt echo (deterministic titles now),
sync title+memory latency (title is local; memory runs after `done`).

---

## 6. Live verification evidence (task-3)

Run: `python3 tests/e2e/run_live.py --shots-out docs/screenshots` against the latest
code on **8799** with scratch `./.dev/qa-data`, model `qwen3.8-2b`, one call at a time.

### Live API/LM checks (run 2026-10-21 22:25–22:30 +0300)

| Check | Result | Evidence |
|---|---|---|
| `live_health` | PASS | real list, 11 chat models, current `qwen3.8-2b` |
| `live_write_file_approval` | PASS | tool_call → approval → **Allow** → file written in workspace (25.5 s) |
| `live_write_file_deny` | PASS | **Deny** → tool_result error, no file on disk (38.3 s) |
| `live_run_command` | PASS | exec approval → Allow → command output returned (44.6 s) |
| `live_stop` | PASS | stop mid-stream → `done{aborted}` + partial text saved (15 s) |
| `live_context_compaction` | PASS | tiny-context fake hook seeded compaction, summary injected, live qwen continued with no context error (29.7 s) |
| `live_restart_persistence` | PASS | conversations + memories + settings survived a restart |
| `live_offline_state` | PASS | closed LM port → `meta` then calm `connection` error, no crash |
| `live_plain_stream` | INCONCLUSIVE (harness) | real stream ran; assertion compared a retry conversation id. Harness fixed afterwards; deterministic UTF-8 streaming + live streamed chat in the browser prove the path |
| `live_markdown_and_title` | INCONCLUSIVE (harness) | title assertion expected raw prompt text; deterministic titles are by design. Browser chat.png shows live streamed markdown |
| `live_model_swap` | INCONCLUSIVE (harness) | assertion ran against an empty conversation due to the shared-conversation bug; swap path itself was fixed to server-persist + restore. Per Lead, no generation on slow spark |
| `live_web_search_tool` | WAIVED | tool ran (`web_search` auto, Bing fallback after Brave 429); final citation is model behaviour. Deterministic fake-LM + live_search_provider cover the pipeline |
| `live_blocked_command` | WAIVED | qwen3.8-2b refused to invoke `run_command` for a safety-sounding command (model safety, not app). Blocklist proven deterministically through the real tool pipeline |
| `live_memory_roundtrip` | WAIVED | extraction is async/best-effort by design; persistence proven deterministically + browser memory sheet CRUD on live backend |
| `direct_search_provider` / `direct_fetch_page` | HARNESS FIX | in-process imports needed repo root on `sys.path`; fixed in `run_live.py`; search chain verified indirectly by live web tool (Bing fallback, Brave 429) |

Live browser checks (same run): console 0 errors, resources clean, no external
requests, contract, XSS, SSE parser, a11y, responsive, keyboard shortcuts **10/10 PASS**.

### Committed screenshots

All 7 files verified as real PNGs and inspected visually.

| File | Source | Content |
|---|---|---|
| `hero.png` | live run | empty state, 1440×900@2x, clean sidebar history |
| `chat.png` | live `qwen3.8-2b` | real streamed Python code answer, clean titles, 5 memories |
| `models.png` | live | model popover over the live transcript, loaded/context/tools markers |
| `memory.png` | live | memory sheet with curated, real memories on the scratch DB |
| `settings.png` | live | LM Studio status + model list + Auto/Ask/Deny permissions + toggles |
| `approval.png` | **mock-mode UI run** (same static frontend) | pending write_file approval card with Deny / Always / Allow + clean sidebar — live qwen answered without calling write_file twice, so per Lead this fallback is used and disclosed |
| `mobile.png` | **mock-mode UI run** | 390×844 empty state, clean titles — live mobile flow cascaded from the approval miss |

The mock-mode screenshots use the exact committed frontend (`?mock=1` swaps only the
JS API layer); backend behaviour is evidenced by the deterministic suite + the live
PASS checks above. Everything else is a live-model screenshot.

### How the live run was executed

One command starts the latest code on **8799** with scratch state, serializes all
model calls on `qwen3.8-2b`, runs the live checks, then runs the browser pass and
writes the committed screenshots:

```bash
python3 tests/e2e/run_live.py --shots-out docs/screenshots
# or attach to an already-running 8799 server:
python3 tests/e2e/run_live.py --base http://127.0.0.1:8799 --no-browser
```

Live checks (`tests/e2e/checks_live.py`, all serialized):

| # | Check | Evidence |
|---|---|---|
| 1 | `live_health` | real model list, current model, per-model fields |
| 2 | `live_plain_stream` | TTFT/total, real streaming, UTF-8 umlauts/CJK/emoji intact, usage |
| 3 | `live_markdown_and_title` | streamed markdown + persisted auto-title |
| 4 | `live_web_search_tool` | real `web_search` auto-run, non-empty results, answer cites links |
| 5 | `live_write_file_approval` | approval card → Allow → file written inside workspace |
| 6 | `live_write_file_deny` | Deny → tool_result error, no file on disk |
| 7 | `live_run_command` | exec risk card → Allow → command output returned |
| 8 | `live_blocked_command` | `sudo rm -rf /` blocked even after Allow |
| 9 | `live_stop` | stop mid-stream → `done{aborted}` + partial saved |
| 10 | `live_model_swap` | mid-chat swap to a second loaded model, per-message models kept, memories unchanged |
| 11 | `live_memory_roundtrip` | remember tool / extraction writes a durable memory |
| 12 | `live_context_compaction` | tiny-context fake hook seeds compaction (summary injected), then real qwen continues with no context error |
| 13 | `live_restart_persistence` | conversation + memories survive a real server restart |
| 14 | `live_offline_state` | second app on closed LM port: calm connection error, no crash |
| 15 | `direct_search_provider` | real provider chain returns ≥1 result with non-empty snippets |
| 16 | `direct_fetch_page` | real page fetch + readable-text extraction |

After the live pass: inspect every committed screenshot with vision, re-run the
deterministic suite on the exact shipped code, check README claims, fill the evidence
numbers above, then report residual risks and close task-3/task-4.

---

## 7. Residual risks & limitations

- Sandbox and command blocklist are **best-effort** (documented in ARCHITECTURE):
  no container isolation; a crafted shell line can still do damage inside the user's
  account. Approvals are the real control.
- SSRF guard resolves DNS before connecting (TOCTOU/DNS-rebinding window exists);
  re-resolution per redirect hop mitigates typical redirect tricks.
- Approvals and abort flags are in-memory (single process) by design; a restart
  clears them.
- Fake LM is a protocol stub: it proves contract handling, ordering and error
  recovery, not real model quality/latency. Live pass covers that.
- Browser rig needs Chrome + playwright-core (dev install under `.dev/qa`); CI runs
  the Python suite, not the browser rig, unless Chrome/playwright are installed.
- Mock mode exercises UI states but not backend behaviour; it is a rig validation
  tool, not evidence for API acceptance.
- Werkzeug dev server disables HTTP keep-alive (see finding 5); production packaging
  is a browser app on a local single-user server, so this is cosmetic.

---

## 8. Command appendix

```bash
# deterministic suite
python3 tests/e2e/run_e2e.py
python3 -m pytest tests/e2e -q

# UI/backend unit suite (backend-owned)
python3 -m pytest -q

# live LM Studio verification (serialized; 8799 + scratch data; screenshots last)
python3 tests/e2e/run_live.py --shots-out docs/screenshots

# browser rig
node tests/e2e/browser/run_browser.mjs --mode mock  --out .dev/qa/browser/shots
node tests/e2e/browser/run_browser.mjs --mode fake  --out .dev/qa/browser/shots
node tests/e2e/browser/run_browser.mjs --mode live  --out docs/screenshots

# fake LM standalone
python3 tests/e2e/fake_lm.py --port 0 --print-url
```

---

## 9. Final acceptance summary

- **No P0/P1 bugs open.** Both P1s found in the static audit (LM-offline crash,
  compaction summary loss) were fixed by backend and re-verified by the deterministic
  suite; the P2s (meta ordering, memory-event contract) are fixed/amended-docs.
- **Contracts**: docs/API.md + ARCHITECTURE.md surfaces covered end to end; loopback
  Host guard, no-store/no-cache headers, error envelopes, 413, cascade delete,
  approvals, abort, model swap, persistence all asserted.
- **Safety**: path traversal (write + read + symlink), SSRF (local/private/link-local,
  redirects re-checked), command blocklist, output/timeout caps, markdown XSS
  (9 payloads inert in real Chrome), FTS injection, request-size limit.
- **README claims verified**: all 7 referenced screenshots exist as PNGs; no external
  fonts/CDNs (browser asserts zero cross-origin requests); scripts pass `bash -n` and
  `python.sh` venv fallback exists; icons 64/128/256/512 + favicon present; `.desktop`
  template substitution in `install.sh`; pytest suite green (backend 67 + QA e2e).
- **Residual risks** (unchanged, documented): best-effort sandbox/blocklist (not a
  container), DNS-rebinding TOCTOU window in the SSRF guard, in-memory approvals
  (lost on restart), Werkzeug dev-server duplicate `Connection: close`, unknown-model
  selection while LM Studio is offline, no `context` event on context-limit recovery.

---

## 10. Addendum — v1.1.0 identity & visual update (post-QA)

Requested after the QA run above: models must know they are Kiwibro, and the green
status dots had to become a different shape. Evidence for the v1.1.0 code paths:

- **Tests:** deterministic suite **54/54 PASS** and pytest **68 passed** on the v1.1
  code; fake-LM prompt matching made case-insensitive after the prompt rewrite.
- **Identity prompt (live):** with `qwen3.8-2b`, asking who it is answers e.g.
  *"I am an AI assistant named Kiwibro, running on this local machine…"*.
  `test_system_prompt_identity_and_capabilities` asserts the prompt includes the
  Kiwibro identity, local-LM-Studio framing, current model id, workspace, tool list +
  permission split, memory/summary context and local time, plus the agent-disabled
  and model-without-tools variants.
- **Status markers:** round green dots replaced by small diamond markers (loaded /
  idle / warning) in the header pill, sidebar footer, model popover, LM Studio status
  card and model list; the memory metadata separator is a thin dash. All seven
  screenshots in `docs/screenshots/` were re-captured and inspected in Chrome with
  zero console errors.
- **Launcher self-update:** `/health` exposes `started_at`; `scripts/kiwibro.sh`
  restarts a running server when repository sources are newer. Verified locally:
  cold start → chrome args correct; second launch keeps the server when unchanged;
  after touching sources the server restarts and `started_at` increases.
- **Screenshot provenance (supersedes §6):** hero, chat, models, memory, settings and
  mobile were captured against the live app with `qwen3.8-2b`; `approval.png` uses the
  mock-mode UI state (identical committed frontend) because the small live model
  sometimes answers tool requests as plain text. The real approval flow (allow/deny,
  file written) remains live-verified in §6.
