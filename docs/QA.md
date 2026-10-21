# Kiwibro — QA & Verification

Independent QA for Kiwibro v1. Contracts under test:
[ARCHITECTURE.md](ARCHITECTURE.md) and [API.md](API.md). This file is the evidence
log: what was run, what passed, what broke, and what is still residual risk.

**Status:** deterministic (fake-LM) suite green on latest backend; live LM Studio +
committed screenshot pass pending QA authorization. Last deterministic run:
`2026-10-21 18:14 +0300` — **54/54 checks passed**, mock-mode browser rig
**8/8 checks + 7 screenshot flows passed**.

> Live verification (task-3) is gated on: backend P1/P2 fixes in, Lead restarts the
> canonical 8777 server from latest code, and LM Studio confirmed healthy. QA uses
> port **8799** + `.dev/qa-data` only and never touches 8777.

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
| `responsive-desktop` / `responsive-mobile` | no horizontal overflow and no chrome past the viewport at 1440×900 and 390×844 |

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

Mock-mode rig validation (2026-10-21): **8/8 checks, 7/7 flows**, screenshots
inspected visually (`.dev/qa/browser/shots/`). These are rig-validation shots, not
committed evidence; committed screenshots require the live LM Studio run.

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
| 5 | P3 | SSE wire has duplicate/conflicting `Connection: keep-alive` + `Connection: close` (Werkzeug dev server always closes) | OPEN (harmless for browser fetch; docs claim keep-alive) — residual risk |
| 6 | P3 | `POST /api/models/select` accepts unknown ids while LM Studio is offline (contract: 400 if unknown) | OPEN (unknowable when offline; low impact) |
| 7 | P3 | Context-limit *recovery* compaction does not emit a `context` event, so the UI chip is not shown (compaction itself works) | OPEN, cosmetic |
| 8 | P3 | `Workspace.list_files` follows symlinked dirs and can expose absolute outside paths in the listing (read/write remain guarded) | OPEN, low impact path disclosure |
| 9 | P3 | Per-conversation lock object is popped in `_cleanup`; a narrow race could create a second lock for the same conversation | OPEN, theoretical |

Already-fixed Lead-reported issues re-verified by the deterministic suite:
UTF-8 mojibake in streamed text, auto-title prompt echo (deterministic titles now),
sync title+memory latency (title is local; memory runs after `done`).

---

## 6. Live verification plan (task-3, gated)

1. Start app from latest code on 8799 with scratch state:
   ```bash
   KIWIBRO_PORT=8799 KIWIBRO_DATA_DIR=$PWD/.dev/qa-data KIWIBRO_WORKSPACE=$PWD/.dev/qa-workspace \
   KIWIBRO_LMSTUDIO_URL=http://127.0.0.1:1234 python3 app.py
   ```
2. Serialized live checks with `qwen3.8-2b`: streamed chat + markdown/UTF-8,
   real `web_search` tool, `write_file` approval (allow), `run_command` approval
   (deny + allow/always), stop mid-stream, model swap mid-chat (transcript + memory
   preserved), restart persistence, compaction (long conversation; no context error),
   LM Studio offline state (point app at closed port).
3. Browser live pass on the same server (mode `live`, `--base`), screenshots to
   `docs/screenshots/`; every screenshot inspected with vision before committing.
4. Re-run the deterministic suite one final time on the exact commit to be shipped.
5. Fill the evidence table + README claims check; report residual risks; mark task-3
   complete only with no P0/P1 open.

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

# browser rig
node tests/e2e/browser/run_browser.mjs --mode mock  --out .dev/qa/browser/shots
node tests/e2e/browser/run_browser.mjs --mode fake  --out .dev/qa/browser/shots
node tests/e2e/browser/run_browser.mjs --mode live  --out docs/screenshots

# fake LM standalone
python3 tests/e2e/fake_lm.py --port 0 --print-url
```
