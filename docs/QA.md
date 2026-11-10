# Kiwibro: testing

How Kiwibro is tested, how to run each layer, and what was checked for version 3.0.0.
The contracts under test are [ARCHITECTURE.md](ARCHITECTURE.md) and [API.md](API.md).

| Layer | Command | 3.0.0 (10 Nov 2026) |
|---|---|---|
| Unit, API and security tests | `python3 -m pytest -q` | 1,489 passed, 5 skipped |
| End-to-end: the real app with a scripted model server | `python3 tests/e2e/run_e2e.py` | 56 of 56 |
| Browser, mock backend | `node tests/e2e/browser/run_browser.mjs --mode mock` | 21 of 21 |
| Browser, the real app with the scripted model server | `node tests/e2e/browser/run_browser.mjs --mode fake` | 21 of 21 |
| By hand, with real small models | see [Tried with real models](#tried-with-real-models) | passed, with the limits listed there |

No test needs a model, a GPU or the internet. Every run uses its own data folder and
workspace (`KIWIBRO_DATA_DIR`, `KIWIBRO_WORKSPACE`), so nothing touches
`~/.local/share/kiwibro` or `~/Kiwibro`. Tests cannot reach a search engine: the search
providers are cut off for the whole suite (`tests/conftest.py`).

The 5 skipped tests are opt-in: three talk to a real model server (`KIWIBRO_LIVE=1`), one
runs LibreOffice (`KIWIBRO_TEST_LIBREOFFICE=1`), one needs the `wasmtime` package. CI has
a second job that installs `wasmtime` and runs the Kiwi Sense tests on the WebAssembly
core (`.github/workflows/ci.yml`).

---

## 1. Unit, API and security tests

```bash
python3 -m pip install --user -r requirements.txt pytest
python3 -m pytest -q
```

One file per module under `tests/`. The model is a scripted stand-in, so each test states
exactly what the model "says" and checks what Kiwibro does with it: the agent loop, tool
calls written badly, steering, memory, notes, compaction, the engine, downloads, the
command sandbox.

## 2. End-to-end suite

Starts a real `python3 app.py` and a fake model server (`tests/e2e/fake_lm.py`) that
speaks the same wire format as llama.cpp: streamed text, reasoning and tool-call
fragments, usage, and scripted errors.

```bash
python3 tests/e2e/run_e2e.py                 # all phases
python3 tests/e2e/run_e2e.py --only chat     # one phase
python3 tests/e2e/run_e2e.py --list          # list the checks
python3 -m pytest tests/e2e -q               # the same suite as one pytest test
```

| Phase | Checks | What it proves |
|---|---|---|
| `unit` | 9 | workspace path containment, command blocklist with timeout and output cap, SSRF guard, context budget, memory dedupe, database and search index, settings validation, title derivation |
| `offline` | 4 | no engine: `/health`, bootstrap and the model list still answer, and a chat ends with one recoverable error and no traceback |
| `api` | 12 | model selection, conversation and memory CRUD, settings validation, workspace listing, headers, Host guard against DNS rebinding, error shapes, state kept across a restart |
| `chat` | 26 | streaming (text, markdown, reasoning, UTF-8, empty, silent model), titles, memory extraction, approval allow / always / deny, a workspace write under the default autonomy, tool errors, path traversal, SSRF, blocked commands, busy and stop, client disconnect, context-limit recovery, a model without tools, a model swap mid-chat |
| `compaction` | 4 | with a tiny context the chat is compacted, keeps going, the summary reaches the model, and `meta` stays the first event |
| `notools` | 1 | a model without tool support is called without tools |

The fake server picks its behaviour from a marker in the prompt
(`[[scenario:tool_write]]`, `[[scenario:context_400_once]]`, …). Evidence is written to
`.dev/qa/runtime/evidence.md` on every run and is not committed.

## 3. Browser rig

`playwright-core` drives headless Chrome. The rig looks for the package in
`.dev/qa/node_modules`, then in Node's own search path; `PLAYWRIGHT_CORE_PATH` names another place.

```bash
node tests/e2e/browser/run_browser.mjs --mode mock --out .dev/qa/browser/shots   # UI with its in-browser mock backend
node tests/e2e/browser/run_browser.mjs --mode fake --out .dev/qa/browser/shots   # the real app + the fake model server
node tests/e2e/browser/run_browser.mjs --mode live --base http://127.0.0.1:8777  # a running app with a real model
```

| Check | What it asserts |
|---|---|
| `console`, `console-after-flows` | no console errors and no uncaught page errors |
| `resources` | no failed requests, no 5xx |
| `no-external` | no request leaves the page's origin: no CDN, no web fonts |
| `contract` | `/api/bootstrap` has the documented shape and the header shows the current model |
| `xss` | 9 hostile markdown payloads render inert |
| `sse` | the stream parser survives a stream cut at arbitrary bytes, including inside an emoji |
| `a11y` | buttons, inputs, images and dialogs are labelled; language, viewport and colour-scheme are set |
| `shortcuts` | `Ctrl+N`, `Ctrl+,`, `Esc` and `Shift+Enter` behave as the README says |
| `responsive-desktop`, `responsive-mobile` | no horizontal overflow at 1440×900 and 390×844 |

The flows use the interface as a person would and each saves one screenshot:

| Flow | Screenshot | What it does |
|---|---|---|
| `hero` | `docs/screenshots/hero.png` | the home screen |
| `models` | `docs/screenshots/models.png` | opens the model picker, switches model and back |
| `chat` | `docs/screenshots/chat.png` | sends a message and waits for the streamed reply |
| `approval` | `docs/screenshots/approval.png` | a file write that asks first; clicks Allow and waits for the result |
| `memory` | `docs/screenshots/memory.png` | adds, pins and deletes a memory |
| `notes` | `docs/screenshots/notes.png` | writes a note, finds it in the list, deletes it |
| `library` | `docs/screenshots/library.png` | the Models panel |
| `how` | `docs/screenshots/how.png` | the How it works panel |
| `settings` | `docs/screenshots/settings.png` | turns Agent mode off and on and checks that it was saved |
| `mobile` | `docs/screenshots/mobile.png` | a conversation at 390×844 |

The committed screenshots come from the mock run, whose data is fixed. For 3.0.0 the
panels were also looked at in Chrome at 360, 768 and 1440 pixels wide.

## 4. Security probes

| Attack | Guard | Expected |
|---|---|---|
| `read_file ../../../etc/passwd` | workspace containment after `resolve()` | refused, no content |
| `write_file ../../pwned.txt` | workspace containment, also after Allow | refused, nothing written outside |
| a symlink in the workspace pointing at `/etc/passwd` | `resolve()` + `is_relative_to` | refused |
| `sudo …`, `rm -rf /`, a fork bomb, `mkfs`, `dd` to a device | command analyser | refused even when approved |
| `fetch_page http://169.254.169.254/…`, `127.0.0.1`, `::1`, `file://` | SSRF guard (literal and DNS) | refused before a socket opens |
| `Host: evil.example`, `127.0.0.1.evil.example` | loopback Host allowlist | 403 `forbidden_host` |
| markdown with `<script>`, `onerror`, `javascript:`, `data:` | escape-first renderer | inert, checked in Chrome |
| memory search `'; DROP TABLE …` | quoted full-text query with a LIKE fallback | 200, no error |
| a JSON body over 4 MiB | `MAX_CONTENT_LENGTH` | 413 `too_large` |
| `/static/../app.py` | Flask's static handler | 404, never the source |

All of these are part of the suites above.

## Tried with real models

Scripted tests cannot show how a small model behaves, so 3.0.0 was also used by hand on
this hardware: a Ryzen 5 7530U laptop with integrated Radeon graphics, running the
llama.cpp build (Vulkan) that LM Studio had installed, through an isolated Kiwibro
instance.

| Model | What was tried | Result |
|---|---|---|
| a 0.8B Qwen distil | sums and percentages, "who are you and what can you do?", a remembered pet's name, saving a note, a population lookup | correct; the sums are Kiwibro's own |
| a 1.2B Qwen distil | dictated notes, facts about family, birthday and allergies and recalling them, two questions in one message, a lookup on the web | correct |
| LFM2.5 1.2B | recall across chats, a note the model tried to save with the wrong tool, a self-description, a version lookup | correct; Kiwibro saved the note |

Limits of this check:

- It was a few dozen turns per model, in English, not a benchmark.
- Kiwibro's own engine download (Models → Install engine) was not run on this machine; the
  download, checksum and unpacking are covered by tests with a local file.
- Kiwi Sense ran on its Python twin on the server. The WebAssembly core is covered in CI
  and runs in the browser.
- The 4B and larger models were not re-run for this version.
