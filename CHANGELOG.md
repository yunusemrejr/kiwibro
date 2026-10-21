# Changelog

All notable changes to Kiwibro are documented here.

## [2.0.0] — 2026-10-22

The agent release: complex tool use, model teamwork and autonomous-work safeguards,
on top of the context/memory efficiency overhaul (see Unreleased notes below, which
ship in this version).

### Added

- **27 new tools (35 total).** Parallel web research (`multi_search`, `fetch_pages`,
  `web_research` with numbered evidence, `browse_page`, multi-tab `browser`);
  codebase management (`grep_files`, `glob_files`, `read_files`, `file_outline`,
  atomic `apply_patch` with diff + pre-edit backup, `file_manage`, read-only `git`,
  `multi_command` pipelines, persistent `terminal_session`); sandboxed
  `run_python`/`run_javascript` snippets; computer use (`system_info`, `open_path`,
  `clipboard`, `notify`, `screenshot`); persistent `task_board`; `recall` over
  memory + episodic history. Optional extension tools plug into the same catalog.
- **Capability-aware model teamwork.** `model_info` lists every local LM Studio
  model with capabilities; `delegate`/`describe_image` hand bounded side-questions
  to the smallest adequate installed model (e.g. vision) and return to the main
  model; `study_file` answers questions about long files by chunk→notes→merge.
  When no model fits, the playbook falls back to solving it with sandboxed code.
- **Tool profiles & parallel execution.** Per-turn task-shaped tool subsets
  (auto/full/minimal/coding/research/desktop) keep 1–4B models accurate and cheap;
  independent read/search calls in one round run concurrently.
- **Mid-turn steering.** `POST /api/chat/steer` (send while a reply runs) applies
  user corrections at the next step; `done` now reports turn `stats`.
- **Autonomy safeguards.** Duplicate-call loop guard, per-turn write budget,
  mass-delete caps, all-or-nothing patches.
- **Tool calls that work on 1–4B models.** Malformed JSON, misspelled tool and argument
  names, wrong types and out-of-range values are repaired against each tool's schema;
  calls written as text in ten common formats (Hermes, Qwen, Llama, Mistral, harmony,
  DeepSeek, Gemma `tool_code`, ReAct, JSON, `name(...)`) are recovered, with their markup
  hidden while streaming; errors show the exact valid usage. Models without native tool
  calling get a text tool protocol. A silent reply after tool results gets one nudge.
- **Content guard.** Secrets in tool output become `[SECRET_n]` placeholders (filled in
  for local files and commands, refused in web requests, never stored in chat history);
  prompt-injection text in pages, documents and files is fenced and makes risky tools
  ask for approval for a few turns; overwrites that would replace code with
  `... rest unchanged` placeholders or clobber binary files are blocked, and big
  shrinks or folder deletes are escalated.
- **Checkpoints and `undo_changes`.** Every agent file change (write, edit, patch,
  move, copy, delete) is snapshotted outside the workspace and can be undone step by step.
- **Document library.** `doc_search`, `doc_read`, `doc_add` and `doc_list` over an
  on-disk FTS5 index of PDF, Word, ODT, EPUB, PowerPoint, Excel, HTML, Markdown, text,
  code, CSV and JSON; workspace documents index automatically. Over-long tool output is
  paged into the library (head and tail inline, the rest readable by handle) instead of
  being truncated. `convert_document` converts between Markdown, HTML, Word, ODT, EPUB, RST,
  LaTeX, text, PDF and Office formats with the installed pandoc / WeasyPrint / LibreOffice /
  pdftotext (never overwriting; undoable). `study_file` now reads PDF/Office documents and
  studies the parts of long documents that answer the question.
- **Offline compute.** `calculate` (exact math, unit conversion, date arithmetic) is
  offered on every turn; `analyze_table` profiles, filters and aggregates CSV/JSON files
  by streaming them.

### Changed

- Tool rounds per reply are configurable (`max_tool_rounds`, default 16).
- Model entries expose `vision` + `capabilities`; vision shown in model lists.
- Tool results are head+tail shaped before storage; fetched pages are cached (5 min).

## [Unreleased]

### Changed

- **Major context and efficiency overhaul of the hot path.** Token estimation is
  C-speed and now self-calibrates per model from LM Studio's reported
  `prompt_tokens`. Old tool results decay with age (newest outputs stay detailed;
  older ones shrink to a head+tail excerpt) instead of being re-sent in full on
  every turn — a 360-message tool-heavy conversation now sends ~56% fewer prompt
  tokens. Compaction summarises only the oldest slice needed to get back under
  budget (more recent history stays verbatim), and the summariser now sees the
  whole segment rather than only its newest part. Agent tool rounds append only
  new messages instead of rebuilding the full history from SQLite each round.
- **Faster plumbing.** SQLite connections are reused per thread, memory counting
  uses `COUNT(*)`, settings reads are served from a write-versioned cache, model
  list fetches are single-flight, and streamed token deltas are coalesced into
  fewer SSE frames. Context preparation on the benchmark conversation dropped
  from ~13 ms to ~3 ms.

## [1.1.0] — 2026-10-21

Identity and polish update.

### Changed

- **Models now know they are Kiwibro.** The system prompt explicitly names the agent,
  its local LM Studio setup, the current model id, the workspace, the tools available
  this turn (and their permission model), memory/summary context, and the current
  local time. Utility prompts (titles, summaries, memory extraction) are branded too.
- **Status markers are no longer green dots.** Loaded/idle/warning states use a small
  geometric diamond marker (header model pill, sidebar footer, model popover, settings
  status card and model list). The memory metadata separator is a thin dash.
- **Desktop launcher self-updates.** `/health` now reports `started_at`; launching
  Kiwibro restarts a running server when repository sources changed since it started,
  so the app always serves current code after an update.

### Fixed

- Fake-LM e2e harness matches prompt text case-insensitively after the prompt rewrite.

## [1.0.0] — 2026-10-21

First release: a local-first AI agent for Linux with a native-feel browser GUI.

### Added

- **Local chat with LM Studio** — streaming replies, reasoning shown collapsed,
  markdown and copyable code blocks, stop/abort, automatic model listing with
  loaded state, context size and tool capability. Switch models from the header
  without touching transcripts or memory.
- **Context engine that never breaks** — token estimation, dynamic per-model
  budgets, rolling conversation summaries injected into the prompt, tool-output
  trimming and hard fallbacks. Context-limit errors from LM Studio are recovered
  automatically instead of surfacing.
- **Memory without chaos** — async extraction of durable facts after replies,
  normalisation + fuzzy dedupe, six categories (fact, preference, project, person,
  decision, plus pinned), FTS search, editable/deletable from the memory sheet.
  Memory is global, model-independent and capped.
- **Agent tools with permissions** — `web_search`, `fetch_page`, `read_file`,
  `list_files`, `write_file`, `replace_in_file`, `run_command`, `remember`.
  Read-only tools run automatically; writes and commands ask with an inline
  approval card (Allow / Always / Deny, 120 s timeout). `Always` persists.
- **Safety rails** — workspace sandbox with symlink-safe path containment,
  SSRF-guarded page fetching, command blocklist, timeouts and output caps,
  loopback-only server with DNS-rebinding host guard, 4 MiB request cap.
- **Native Ubuntu feel** — dark GNOME-style window, hand-drawn kiwi mark, app
  icons and a `.desktop` launcher that opens a Chrome app window. Browser GUI
  only: no Electron and no external assets.
- **Fully local** — SQLite (WAL + FTS5) in `~/.local/share/kiwibro`, workspace in
  `~/Kiwibro`, loopback server, no telemetry, no CDN/fonts.
- **Quality gate** — 67 pytest unit/API tests and 54 deterministic fake-LM
  end-to-end checks (streaming, approvals, compaction, recovery, security),
  plus a Playwright browser rig and a GitHub Actions CI workflow.

### Notes

- Browsers with ES modules and `ReadableStream` are required (current Chrome,
  Chromium, Firefox, Edge).
- LM Studio must run its local server (default `http://127.0.0.1:1234`).
- Memory extraction runs asynchronously after a reply; the UI refreshes counts
  shortly after `done`, when the memory sheet opens and on window focus.
