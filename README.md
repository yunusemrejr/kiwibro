<div align="center">

<img src="docs/screenshots/hero.png" alt="Kiwibro — a quiet local AI agent" width="920">

# Kiwibro

**A local AI agent for Linux. Quiet, capable, and yours.**

[![CI](https://github.com/yunusemrejr/kiwibro/actions/workflows/ci.yml/badge.svg)](https://github.com/yunusemrejr/kiwibro/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-9BCE6B.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org)

`Flask` · `LM Studio` · `Browser GUI with a native Ubuntu soul` · `No cloud`

</div>

---

Kiwibro is a local-first AI agent that runs on your machine and talks to
[LM Studio](https://lmstudio.ai) through its local OpenAI-compatible API. Pick any
model you have, chat with it, let it search the web or write code — with your
permission, inside its own workspace.

- 🥝 **Model freedom.** Choose a model from the header pill; swap mid-chat. Nothing
  else changes — not memories, not transcripts.
- 🧠 **Memory, not chaos.** Durable facts are extracted after each turn,
  deduplicated, categorised and editable. Memory is global and model-independent.
- 🪶 **Context that never breaks.** Rolling summaries, dynamic token budgets and
  hard fallbacks mean long conversations keep going. No "context length exceeded".
- 🛠 **Agent, on a leash.** Web search, page reading, file edits and shell commands.
  Reads are automatic; writes and commands ask first. `Always` is one click away.
- 🖥 **Native feel.** A dark GNOME-style window, a hand-drawn kiwi mark, an app icon
  and a `.desktop` launcher. It is a browser GUI on purpose: no Electron, no bundle.
- 🔒 **Local.** SQLite on disk, loopback server, no telemetry, no accounts. Web tools
  only touch the network when you (or the agent, visibly) use them.

<br>

<div align="center">
<img src="docs/screenshots/chat.png" alt="A conversation with streamed code" width="440">
<img src="docs/screenshots/approval.png" alt="A pending permission request" width="440">
<br>
<img src="docs/screenshots/memory.png" alt="Memory sheet" width="440">
<img src="docs/screenshots/settings.png" alt="Settings and permissions" width="440">
</div>

## Features

| | |
|---|---|
| **Chat** | Streaming replies, markdown + code with copy, collapsible reasoning, stop/abort. |
| **Models** | Live list from LM Studio, loaded/available state, context size, tool capability. |
| **Context** | Automatic rolling summarisation with a quiet context meter. |
| **Memory** | Global, deduped, FTS-searchable. Pin, edit, delete, forget all. |
| **Agent tools** | `web_search`, `fetch_page`, `read_file`, `list_files`, `write_file`, `replace_in_file`, `run_command`, `remember`. |
| **Permissions** | Auto / Ask / Deny per tool, inline approval cards, "Always" remembers. |
| **Safety** | Workspace sandbox, path-traversal guards, SSRF protection, command blocklist, timeouts, output caps. |
| **Privacy** | Loopback-only server, local SQLite (`~/.local/share/kiwibro`), no external assets. |

## Requirements

- Linux (developed on Ubuntu; any GTK/GNOME desktop works)
- Python 3.10+
- [LM Studio](https://lmstudio.ai) running with its local server enabled
  (default `http://127.0.0.1:1234`), and at least one chat model
- Chrome/Chromium (for the native app window) or any modern browser

## Quick start

```bash
git clone https://github.com/yunusemrejr/kiwibro.git
cd kiwibro

# start it (installs Flask/requests/beautifulsoup4 into a local .venv if needed)
./scripts/run.sh
```

Prefer doing it by hand?

```bash
python3 -m pip install --user -r requirements.txt
python3 app.py
```

Open **<http://127.0.0.1:8777>**.

In LM Studio, enable the local server (Developer → Start Server). Kiwibro lists every
chat model you have; models not currently loaded are loaded just in time by LM Studio
(the first message may take a little longer).

### Install as a desktop app

```bash
./scripts/install.sh
```

Kiwibro appears in your application grid with its kiwi icon. Launching it starts the
local server (if needed) and opens a Chrome app window — a browser GUI that looks and
feels like a native Ubuntu application. No root required.

## Using it

| Action | How |
|---|---|
| New chat | `Ctrl/⌘ + N` or the pencil button |
| Change model | Click the model pill in the header |
| Memory | Brain button in the header |
| Settings & permissions | Gear button or `Ctrl/⌘ + ,` |
| Send / newline | `Enter` / `Shift + Enter` |
| Stop a reply | The square stop button |
| Close a sheet | `Esc` |

**Context.** You never need to manage it. When a conversation approaches the model's
window, Kiwibro summarises the older part into a rolling summary and keeps recent
turns verbatim. A subtle chip tells you when that happens; your transcript is never
rewritten.

**Memory.** After each reply Kiwibro quietly extracts durable facts ("prefers Python",
"working on a Flask app", "lives in Istanbul"). They are deduplicated and merged, and
only the relevant ones are included in the prompt. You can review, edit, pin or
delete everything in the memory sheet. Memories are shared across chats and survive
model swaps and restarts.

**Permissions.** Read-only tools (search, page read, file read/list, remember) run
automatically. Tools that can change your machine (`write_file`, `replace_in_file`,
`run_command`) pause and ask, with the exact command or file preview in the card.
Choose **Allow** once, **Always** to stop being asked for that tool, or **Deny**.
Every mode is configurable in Settings.

> Commands run as your user, inside the workspace, with a timeout and a blocklist.
> That is a strong guardrail, not a container — review commands in the approval card.

**Workspace.** Files the agent reads and writes live in `~/Kiwibro` by default. Change
the folder in Settings; it never moves or deletes your files.

## Configuration

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `KIWIBRO_HOST` | `127.0.0.1` | Bind address (keep loopback) |
| `KIWIBRO_PORT` | `8777` | HTTP port |
| `KIWIBRO_DATA_DIR` | `~/.local/share/kiwibro` | Database and logs |
| `KIWIBRO_LMSTUDIO_URL` | `http://127.0.0.1:1234` | LM Studio server |

Runtime settings (model, permissions, workspace, temperature, output tokens, user
name) live in the Settings sheet and are stored in SQLite.

## How it works

```
Browser GUI (vanilla JS, SSE)
        │
        ▼
Flask API ── agent loop ── tools (permissions + sandbox)
        │                      │
        │                      ├─ web search / page fetch
        │                      └─ workspace files / shell
        ▼
context engine (token budget, rolling summaries)
        │
        ▼
LM Studio (OpenAI-compatible, localhost)
        │
        ▼
SQLite (chats, memories, settings, usage)
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/API.md](docs/API.md).

## Privacy & safety

- The Flask server binds to `127.0.0.1` only and rejects non-local `Host` headers.
- Your chats, memories and settings stay in `~/.local/share/kiwibro/kiwibro.db`.
- No analytics, no accounts, no CDNs, no fonts fetched from the internet.
- File tools are confined to the workspace; page fetching refuses localhost, private
  and link-local addresses (anti-SSRF).
- Shell commands are blocklisted, time-limited and always approved by default.

## Verification

Kiwibro's contract is tested at three levels, all runnable on your machine:

```bash
python3 -m pytest -q              # 67 unit/API/security tests (offline, fake LM)
python3 tests/e2e/run_e2e.py      # 54 deterministic end-to-end checks (real app + fake LM)
node tests/e2e/browser/run_browser.mjs --mode mock   # browser rig (needs playwright-core + Chrome)
```

The deterministic suite covers SSE framing, UTF-8 streaming, approvals
(allow / always / deny), stop/abort, context-limit recovery, rolling-summary
injection, model swap preserving transcript and memory, LM Studio-offline
behaviour, path-traversal/SSRF/shell-blocklist probes and restart persistence.
The full evidence log, including live LM Studio runs and residual risks, lives in
[docs/QA.md](docs/QA.md). CI runs the deterministic suite on every push.

## Development

```bash
python3 -m pip install --user -r requirements.txt pytest
python3 -m pytest -q
python3 app.py
```

Project layout and contracts are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Contributions are welcome — keep it local, calm and dependency-light. See
[CHANGELOG.md](CHANGELOG.md) for the release notes.

## License

MIT © 2026 Yunus Emre Vurgun
