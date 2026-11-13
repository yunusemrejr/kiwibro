# Changelog

All notable changes to Kiwibro are documented here.

## [Unreleased]

### Added

- **Kiwi Mind: Kiwibro's own model.** Kiwibro now thinks with a model stack of its own: one
  4-bit model (Qwen3-0.6B, trained on the grid it is stored in), seven skills it swaps by
  itself (chat, web, code, terminal, computer, tools, write) and a control head that picks
  the skill for each message. It runs on the CPU in WebAssembly, starts and stops with
  Kiwibro and needs about 1 GB of memory. The pack (under 500 MB) is fetched once from Hugging
  Face (`yunusemrejr/kiwimind`); `scripts/mind/train.py` is the whole training run.
- **Helper models.** The model picked in the header is now a helper. Kiwi Mind decides which
  tools to use and answers everyday requests itself; when its control head marks a request
  as heavy and a helper is selected, the helper writes that answer. *No helper* is a valid
  choice. Until the pack is there, the selected model answers as before and the chat says so.
- **Safe copy.** *Work in a safe copy* takes a folder and lets Kiwibro work in a copy of it:
  its own browser, a sandboxed terminal without network, and nothing outside the copy. Each
  step is recorded in git. The real folder changes only after *Review changes* has shown
  what changed, where, and the git command that puts the old files back, which Kiwibro
  tries out on a scratch copy first. Files deleted in the copy are never deleted for real:
  the `rm` commands are handed to you.
- **Components.** A library of 38 ready parts (web pages and apps, Python, Node and Bash
  starters, a GTK app for Ubuntu, scripts, and recipes for browser and desktop chores) that
  every model can search and use with the new `component` tool, so its work does not start
  from a blank page.
- **Session ledger.** Every step of a conversation stays stored in full. After older turns
  are summarised, a short index of what was asked, which files changed, which commands ran
  and which pages were read stays in view, and `recall("step 14")` reads a step again word
  for word.
- **Without internet.** Kiwibro notices a lost connection within seconds, tells the model,
  and web tools fail at once with a plain message instead of timing out. Kiwi Mind, files,
  notes, memory and documents work as before. A reply that needed the web then starts with
  "No internet connection, so I could not look this up. What follows is from memory and may
  be out of date.", and citation marks that point at no source are removed.
- **Checks on what the model wrote.** A file Kiwibro writes is parsed right away (Python,
  JSON, TOML, shell, JavaScript; nothing is run). If it does not parse, the model is told
  the line and the reason so it can repair the file, and a file that is still broken at the
  end is named under the reply by Kiwibro itself. A tool call that arrives damaged is not
  run: the model is asked to make it again.
- **How it works, under the hood.** Each of the ten stops now names what is behind it, in
  two columns, model layer and app layer: quantization-aware training, LoRA skill
  swapping, the control head (the gating idea of a mixture of experts), the KV cache and
  prefix caching, retrieval before generation, full-text search, the sandbox.
- **Kiwi Sense helpers.** Five tiny trained deciders read every message before the model,
  in a few milliseconds and the same way for every model: its language, what it wants
  (intent, web, time, topic, whether it says something worth remembering), its time
  phrases and demands, whether it follows up or changes topic, and which saved memories
  and notes bear on it. They ride in `sense.bin` and run in the WebAssembly core and its
  Python twin.
- **English only.** A message that is not in English gets a short reply, written by code,
  asking for English and naming the language it reads as. Quoted words, names, code and
  links inside English messages pass. A new *English only* switch in Settings turns it off.
- **Dates worked out, demands passed on.** "Last week", "since March", "3 days ago" are
  turned into real dates in the turn's note, with what the answer must honour ("3 bullet
  points", "under 100 words").
- **Topic switches.** A follow-up, or another question on the same subject, is looked up
  together with the exchange before it; a new topic is marked only when it is plain, so the
  model answers it on its own. Memories that only share a
  word with the message, and facts from hypotheticals or passing moods, stay out.

### Changed

- **Lighter app window.** The window's private Chrome profile is started with a 64 MB cap on
  its page cache and without Chrome's own on-device model downloads, which had grown the
  profile past 200 MB.

### Fixed

- **"Search for…" searches before anything is written.** A message that asks for a search,
  or for news, is searched by Kiwibro before the model writes a word, and the model is
  told to answer only from the results. Small models used to answer first from memory
  and made up whole bulletins.
- **News comes from dated news feeds.** News questions read Google News and Bing News with
  the window the message gives ("today and yesterday only", "this week"), one copy per
  story, newest first, grouped under Today and Yesterday with outlet, time, link and the
  feed's own first sentence. A plain news request gets that bulletin as the reply, word
  for word; a request for more (a summary, an explanation) gets the model's answer with
  the real list under it. A quiet day widens to the past 24 hours and says so.
- **Closing the window stops the model.** When no Kiwibro window is open (it said it was
  closing and did not come back within 15 s, or it went silent for 3 minutes), the local
  model server is stopped once it is idle and its memory is freed. It starts again with
  the next message. A reply or memory pass still running is never cut off.
- **Memories and notes are saved readily, whatever the model.** "Remember…", "could you
  remember…", "don't forget…", "save this to your memory", standing wishes ("keep answers
  short from now on"), moves, things learned, owned and spoken, and facts about people
  the user names are saved before the reply. The model's memory pass now saves what the
  user says about themselves unless it is clearly passing. Dictated and titled notes
  ("note titled Groceries: …") and "make a note about X" are kept even when the model
  only writes the text.
- **"That's wrong", "those are old", "no, the one about…" do the request again.** A message
  that turns the last answer down is no longer searched for by its own words ("all wrong"
  used to find a TV series). The request before it is done again: a news request lists
  stories that were not shown, a message that says what was meant keeps the subject and
  adds what it names, and the answer that was turned down is not shown to the model again.
  A message with a question of its own ("no, mine is washed already, what then?") is
  answered as asked, so reading a normal message as a complaint by mistake starts no
  search. Typos in the complaint are read ("wrng", "serach agian").
- **Questions about this computer are answered from what is there.** "How many files are
  on my desktop", "which one is the biggest", "what CPU and how much RAM do I have", "how
  much free disk space": Kiwibro looks first and writes the count, the file or the part
  from the result's own figures, without asking the model, which used to answer with
  instructions or made-up numbers. Anything more than the figures ("is that enough for
  games?") still goes to the model with the result. Free space is that of the disk the
  home folder is on.
- **An answer has to come from the sources.** After a search, names, numbers and dates
  found in no result are sent back once; if they are still there, they are removed, and
  when nothing is left the stories or facts are shown as the sources give them, or the
  reply says it was not found. A reply is never empty and never a fill-in template: a
  greeting answered with one gets a plain greeting.
- **A saved fact is the answer to a question about it.** "Where do I live?", "what's my
  dog's name?": if the reply leaves out the saved fact, the model is shown it once more,
  and after that Kiwibro says it ("From what you told me: you live in Bursa").
- **"As one paragraph" gives one paragraph**, lines a model repeats word for word are kept
  once, and an answer given earlier to another question is not given again.
- **A note Kiwibro kept for you is confirmed in the tool's own words** ("Added to note
  “Packing”: sunscreen."), instead of asking the model for a sentence it sometimes got
  wrong.
- **A bracketed stand-in is never the answer.** "Here is the summary as one paragraph.
  [One coherent paragraph summarizing…]" used to reach the screen. A bracketed sentence
  that stands where the answer should be is now asked for once more and then left out,
  and a second attempt that kept the form and lost the substance ("The answer is.") is
  replaced by what the sources say. Links, code and "[Reuters]" after a Reuters story pass.
- **"Are they at the same commit?" is answered from git.** A question whether a checkout
  matches its remote ("is my repo in ./site up to date with origin?") runs the read-only
  compare first and is answered in its figures: the same commit or not, how many commits
  ahead or behind, the last local commit, and uncommitted files.
- **Weather comes from forecast data.** "What's the weather in Izmir" is answered from
  Open-Meteo's figures for the place (now, and three days), with the source and its time.
  Weather pages keep their numbers in scripts, so a search left the model guessing. Only
  the place name is sent; if the service does not know a place by that name, the web
  search runs as before.
- **Kiwibro's own searches read the pages.** When Kiwibro searches before the model writes,
  it now opens the three best results and passes on the passages that bear on the question
  (9 seconds at most). The step existed but was never switched on.
- **How it works: checked answers.** A ninth stop shows how a draft is compared with what
  was found. The KV cache and retrieval texts now say what the code does.

### Checks

- `scripts/replay.py` replays made-up conversations (`tests/replay/scenarios.json`) against
  a test instance, with the helpers on or off, and saves every reply for reading side by
  side. `scripts/detectors.py` counts what the complaint and news readers catch, miss and
  read wrongly on three sets of made-up messages (`tests/data/messages`), and how many
  wrong readings would start a search.

### Safety

- **Kiwibro's own code, start-up files and credentials stay protected inside a working
  folder.** A chat working in `~/.config`, `~/.local` or a `~/Desktop` that holds the
  Kiwibro checkout could write log-in scripts (`~/.config/autostart`, `systemd --user`
  units, `~/.local/bin`) or rewrite Kiwibro itself, which restarts onto changed code.
  These places are now judged before the workspace and refused, and the sandbox mounts
  them read-only (credentials hidden) on top of the writable folder. The workspace
  setting refuses them too.
- **Removals and bulk edits.** Removing or rewriting files that are only known when a
  command runs (`ls | xargs rm`, `find … -exec sed -i`, `sed -i … $(git ls-files)`) asks
  first; a glob counts every file it matches, and an in-place edit of more than 20 files
  asks; `mv` into `/tmp` counts as a removal (commands get a private `/tmp` that is thrown
  away); removing a file larger than an undo copy (20 MB) asks; more than 24 removals or
  120 file changes in one chat within an hour ask, so a long unattended session cannot
  spread a spree over many turns. Autonomous mode refuses all of these.
- **Every route to a mass change asks.** One command that changes or removes more than 20
  existing files asks however it gets there: brace ranges (`rm f{1..30}.py`), several small
  globs, a loop, `truncate`, `tee`, `ln -sf`, `yq -i`, Ruby one-liners (`ruby -pi`,
  `File.delete`), inline code that rewrites files it lists, `zip -m` and `7z -sdel` (which
  delete their inputs), scripted `vim`/`ex`/`ed` edits, `xargs -I{} sh -c 'rm {}'` and
  `git rm -r`. A symlink is judged where it is made, so `ln -sf … ~/.bashrc` is refused
  instead of passing as a write into `/tmp`. Making new files never counts.
- **Commands can be undone.** Before a command runs, the workspace files it will change or
  remove are checkpointed (`rm`, `sed -i`, redirects, `mv`, `truncate`, and the working
  tree before `git reset --hard`, `git checkout .`, `git restore`, `git clean`), so
  `undo_changes` puts them back.
- **"Allow in this chat" covers one kind of step**, not the whole tool, and is never offered
  for a step that can destroy data or that a breaker or prompt-injection warning escalated.
  Before, allowing one `kill` let every later risky command of that chat run unasked.
- **Approved removals stay in the sandbox.** An approved high-risk step leaves the sandbox
  only when it has to (`git push` needs `~/.ssh`); removing a workspace folder, `git reset --hard`
  and other workspace-only steps run confined.

## [3.0.0] — 2026-11-10

Kiwibro runs models itself, watches every reply with a second small model, keeps a
notebook, and explains itself. The LM Studio API is gone; LM Studio's downloaded models
are still found and used. Covered by unit, API and browser tests, and tried by hand on
0.8B to 1.2B models through the llama.cpp build LM Studio installed.

### Added

- **Kiwi Sense** (`kiwibro/sense`, `steer.py`): a reading model that runs beside whichever
  language model is loaded. The core is WebAssembly (`static/sense/sense.wasm`, 9 KB) with
  static word-piece embeddings and a trained sentence probe (`sense.bin`, 2.7 MB); the
  server runs it through `wasmtime` when that package is installed and through a
  bit-identical Python twin otherwise, the browser always runs the WebAssembly. It reads
  each sentence of a reply in under a millisecond and `steer.py` checks it against the
  facts of the turn: speaking as another assistant or as the user, denying a tool that is
  available, claiming work no tool did, promising work and stopping, thinking out loud,
  answering the previous question, looping. A wrong opening is cut off while it streams;
  the model gets one corrected attempt and, failing that, the wrong sentences are removed.
  What a model needed correcting for is kept per model and rides as a standing reminder.
  Switch: Settings → Behaviour → Kiwi Sense.
- **Kiwibro does it itself.** What code can do for certain is not left to the model. Sums
  and unit conversions in a message are calculated first and handed over, and a reply
  with a different number is redone. When a model only talks about searching or about
  saving a note, Kiwibro makes the search or saves the note and says so under the reply.
  A message with several questions is checked for a skipped one. A reply that repeats
  your words as its own (“My dog is named Biscuit”) is turned round, an answer the model
  gave twice in a turn is shown once, and `<answer>`-style wrappers are removed.
- **More is remembered without the model's help**: birthdays, the names of your people
  and pets, where they live, children, allergies and health conditions, one favourite
  per thing. A question about yourself is answered from memory, not from the web.
- **Models panel**: what is on this computer with disk use and memory fit, delete with the
  space it frees (Kiwibro's own models; LM Studio's are removed in LM Studio), Hugging Face
  search and resumable downloads checked against sha256, and the installed engine builds.
- **Engine installer**: one click downloads the official llama.cpp release build for the
  machine (Vulkan or CPU), verifies its sha256 and unpacks it into
  `~/.local/share/kiwibro/runtime`. Nothing is downloaded or run without that click.
- **Notes**: a notebook (`notes.py`, the `note` tool, the Notes panel). The agent saves,
  appends to, reads and searches notes; the notes closest to a message ride in its context
  note; you write them too, in Markdown with tags and pins. Private chats neither read nor
  write notes.
- **Recall by meaning**: when no word of a question matches, memories, notes and the
  compacted part of a long chat are searched by meaning ("what's my wife's name" finds
  "married to Selin"; "that dairy shop" finds the exchange about the cheese place).
- **Adapters adjust themselves**: when a reply goes wrong with a task adapter on, the same
  request first runs again without it. Right without it counts against the adapter: half
  scale after two such replies, off after four, forgotten after 30 days. The Models panel
  lists adapters with their tasks and strength, with Restore and Delete. The task is also
  read by meaning when a message uses no task word.
- **How it works**: eight animated figures in the app (the big picture, tokens and the
  context window, the KV cache, tools, compaction, memory and notes, Kiwi Sense, adapters)
  and a box that runs the real Kiwi Sense in the page.
- **Maths** in replies: `\( … \)`, `\[ … \]`, `$…$` and `$$…$$` render as MathML.
- **Home screen and sidebar places** for Notes, Memory, Models and How it works; a new
  mascot and a palette taken from the kiwi fruit.

### Changed

- **One engine.** Kiwibro starts llama.cpp itself for every model (the former "optimized
  engine" is the only one). A model server you run yourself can be attached by address.
  `KIWIBRO_LMSTUDIO_URL`, `KIWIBRO_LMSTUDIO_TOKEN` and `KIWIBRO_ENGINE` are replaced by
  `KIWIBRO_SERVER_URL` and `KIWIBRO_SERVER_KEY`.
- **First run** is two visible steps, Install engine and Get a model, instead of an
  offline notice.
- **Web search**: a follow-up query that points at the conversation ("how old is he") is
  completed with the name it points at before it leaves, and merged results lead with the
  one that answers the question by words and meaning.
- **Identity prompt**: Kiwibro is the assistant, never the user and never the engine
  underneath; memories in the user's own words are told about the user.
- `remember` refuses what the user neither asked to keep nor said about themselves.
- An empty turn is asked once more and never ends empty.

### Removed

- The LM Studio API client, LM Link listing and the engine switch.

## [2.6.0] — 2026-11-10

Task adapters, Kiwibro's own model and runtime folders, and an engine that lets go of the
model when nothing is asked of it. Covered by unit tests; not yet measured on a live model.

### Added

- **Task adapters** (`adapters.py`, optimized engine). LoRA adapters (GGUF files with
  `general.type = adapter`) placed in `~/.local/share/kiwibro/adapters/` (or
  `KIWIBRO_ADAPTERS_DIR`, or next to the models) are loaded with every base model they fit
  — same architecture, and the base model the adapter names — at scale 0
  (`--lora … --lora-init-without-apply`). Each request then switches on the adapter for
  the task the user is on (code, math, writing, summarize, translate, research; llama.cpp's
  per-request `lora` field). The task is read from the message; a follow-up without a signal
  of its own keeps the previous task, so the adapter — and the chat's KV cache, which
  llama.cpp clears when the adapter changes — only changes when the task does. What an
  adapter is for comes from a sidecar `<file>.json` (`{"tasks": ["code"], "scale": 0.8}`,
  `"*"` = always on) or from its name; one of unknown purpose is never switched on. A model
  that does not start with its adapters starts without them. `GET /api/engine` lists the
  adapters found and the ones each running server holds. Kiwibro ships no adapters.
- **Kiwibro's own folders, no LM Studio needed**: GGUF models in
  `~/.local/share/kiwibro/models/` and a `llama-server` build in
  `~/.local/share/kiwibro/runtime/` (or one folder below) are found before LM Studio's;
  llama.cpp's own download cache (`~/.cache/llama.cpp`) is scanned too.

### Changed

- **An idle engine holds no model memory**: llama.cpp unloads the model after 30 minutes
  without a request (`--sleep-idle-seconds`; `KIWIBRO_IDLE_SLEEP_S`, 0 = never) and reloads
  it on the next one. The first message after such a pause re-reads the chat.

## [2.5.0] — 2026-10-30

The optimized engine: LM Studio's models on the GPU (the integrated one too), thinking that
never makes the model re-read the chat, memory that stays flat through long sessions, and
background work only when it is worth it. Measured on a Ryzen 5 7530U laptop (Radeon Vega 7
iGPU, 38 GB) with `spark-x2.6-4b` IQ2_M; package power from the APU's own sensor.

### Added

- **Graphics acceleration** for the direct engine, now called the *Optimized engine* (a
  switch in Settings → Models; new setting `direct_gpu`: `auto` | `off`, default `auto`).
  Kiwibro asks each GPU build of llama.cpp LM Studio installed (Vulkan, CUDA, ROCm) which
  GPUs it sees (`--list-devices`, once per build; software renderers ignored) and runs the
  model on one with every layer that fits (`--gpu-layers auto`). LM Studio itself ran this
  laptop's models on the CPU. Vulkan on the iGPU against all six CPU cores:

  | | CPU | iGPU |
  |---|---|---|
  | prompt processing, spark-x2.6-4b | 54 tok/s | 85–107 tok/s |
  | decode, spark-x2.6-4b | 7.4 tok/s | 8.9–11.7 tok/s |
  | energy per generated token | 3.3 J | 2.4–2.9 J |
  | CPU time per generated token | 0.81 s | 0.04 s |
  | prompt processing / decode, qwen3.8-2b | 112 / 13.6 tok/s | 249 / 17.3 tok/s |

  A Kiwibro chat (3.5k-token prompt) answered its first message in 38 s instead of 67 s,
  and every later turn in 5–7 s instead of 10–15 s; five turns on the iGPU used 14
  CPU-seconds, four on the CPU 601. A build that cannot load a model falls back to the
  next one, ending on the CPU. The settings card names the GPU in use.
- **Thinking budgets** (direct engine). llama.cpp ends a thought itself when a request's
  `thinking_budget_tokens` run out, and the model answers or acts in the same request. The
  template now stays in thinking mode on every turn and each decision is a budget: 0 (no
  thinking), 1,000 tokens for rounds that must act, 2,500 for a turn's first round. Before,
  turning thinking off re-rendered spark-x2.6's system block (measured: 466 of 466 tokens
  re-read, against 1 with a budget of 0), and an acting round that overthought was cut off
  and retried, re-reading the whole chat (190 s for a 6k-token chat on the CPU). A short
  sentence in the model's voice closes a cut-off thought (`reasoning_budget_message`): with
  it the model answered at once (112 tokens), without it it kept reasoning in its answer
  until the cap (400). "Auto" thinking may now switch per turn at no cost.
- **Exact cache telemetry**: llama.cpp's per-request `timings` (`cache_n`, `prompt_n`,
  prompt time, decode speed) replace the estimates in the reply footer ("98% cached"), the
  log and the speed book that sizes background timeouts.
- **Several tool calls per reply** with the direct engine (`parallel_tool_calls`): the
  prompt already asks for batched independent calls; llama.cpp allowed one per reply.

### Changed

- **Long sessions stay flat in memory** (direct engine). Sliding-window and hybrid models
  (spark-x2.6, qwen3.8, lfm2) can only rewind to saved checkpoints; llama.cpp kept 32 per
  slot at 54 MB each, so host memory grew ~110 MB a turn up to ~1.7 GB per slot. Now 8
  (`--ctx-checkpoints 8`): with 4, eight turns stayed at 0.4 GB instead of 1.1 GB, and a
  turn after six tool rounds re-read 445 tokens more (~5 s on the iGPU). The RAM prompt
  cache is a sixth of free memory, 1–4 GiB (was a quarter, up to 8).
- **Background calls no longer move the chat out of the KV cache**
  (`--no-cache-idle-slots`). llama.cpp copied idle slots to the RAM prompt cache and cleared
  them on every new request, so each memory or summary call between two turns moved the
  chat's state out and back, and with a small cache lost it (a full 3.5k-token re-read,
  40 s). Now the chat stays resident; when the KV pool is full llama.cpp drops the least
  recently used idle slot, which in testing was the background one.
- **Memory is learned in the background only when something memorable was said.** An
  extraction is a fresh ~3k-token prompt plus up to 400 tokens of output: on the CPU a
  minute or two of every core after nearly every reply. It now runs when the user states
  something about themselves, their work or a standing preference (a first-person sentence
  that is not a question, or "remember", "prefer", "always", "never", "from now on", …), or
  once a full window has gathered; questions and plain requests wait and are read together
  with the next memorable message. Clear facts are still saved before every reply by code.
- **Opening an old chat prefills at most ~3 minutes of it** ahead of time, at the model's
  measured speed (6k–24k tokens), instead of always up to 24k tokens: on the CPU that was up
  to 7 minutes of every core, wasted when the chat was only read.
- Runtime choice: only the newest LM Studio install's engine preference counts (a left-over
  native install preferred a 2.17 engine that cannot load current models), CPU builds come
  before GPU builds when acceleration is off, and four runtimes are tried before giving up.
- Measured and left off: KV-cache quantization (q8_0 decoded 13% slower on the iGPU), MTP
  drafting for qwen3.8 (no gain), fewer CPU threads (slower and more energy per token),
  `--poll 0` (no change).

## [2.4.0] — 2026-10-26

A direct engine: LM Studio's downloaded models run with llama.cpp itself, without the
LM Studio API or app, tuned for the machine — plus a smaller "smarter" fix for small
models doing arithmetic in code.

### Added

- **Direct engine** (`engine.py`, `gguf.py`; Settings → Models → *Engine*). With
  `engine: "direct"` Kiwibro scans the GGUF files LM Studio downloaded (native and
  Flatpak installs, or any folder), picks LM Studio's bundled `llama-server` (or a custom
  build from Settings / `KIWIBRO_LLAMA_SERVER`, or one on PATH) and runs the selected
  model as its own child process on a private localhost port with a random API key.
  Everything above the client is unchanged: model list (with tools, vision, thinking
  switch, size, trained context), JIT loading, readiness load + warm-up, context
  choices, model teamwork. Measured against the LM Studio API on a Ryzen 5 7530U with
  `spark-x2.6-4b` IQ2_M under the same load: model start 3.6 s; prompt processing
  37–43 tok/s vs 27–34 (every physical core instead of LM Studio's 4 threads; 12 SMT
  threads were slower); decode 6.8 vs 4.7 tok/s on a reply that repeats its input
  (n-gram speculative decoding, no draft model; unchanged on free text); and the
  LM Studio app (~720 MB) need not run. Also on: unified KV pool, `--cache-reuse`, a
  RAM prompt cache sized to free memory, flash attention auto.
- Older llama.cpp builds work: flags missing from a runtime's `--help` are left out; a
  runtime that exits while loading is retried without speculative decoding and then
  the next runtime is tried.
- Servers stop with Kiwibro (also across self-update `execv`), when switching back to
  LM Studio, on *Stop* in Settings (`POST /api/engine/stop`), or when a third model
  needs room; a server answering a request is never stopped. `GET /api/engine` reports
  the runtime, folders and running servers.
- Settings `engine`, `direct_runtime`, `direct_models_dir`, `direct_threads`; env
  `KIWIBRO_ENGINE`, `KIWIBRO_LLAMA_SERVER`, `KIWIBRO_MODELS_DIR`. Switching to direct
  unloads the models Kiwibro's warm-up had loaded in LM Studio (models you loaded there
  yourself stay loaded).

### Changed

- **`run_python` prints a final bare expression**, like a REPL. Small models sent
  `1234 * 5678`, got "(no output)" and then guessed the product (seen live: 7853476 for
  7006652).
- The header, model menu and composer hints name the direct engine instead of
  LM Studio when it is selected.

## [2.3.0] — 2026-10-25

Memory that finds what matters and sessions that keep their thread: the right memories
reach a small model's prompt (and the noise stays out), a chat can pick up what another
chat found, retries and interrupted replies no longer double the request, chats are
findable by what was said in them, and a chat can be private. The app also updates itself
when its code changes, and saves what the user states about themselves before every reply.

### Added

- **Self-updating server** (`selfupdate.py`). The server keeps a content fingerprint of
  the code it serves (`code_version` in `/health` and the new cheap `GET /api/version`).
  A watcher notices changes on disk, waits for them to settle (20 s), checks that the
  new code compiles and imports, and re-executes the server in place once no reply or
  memory extraction is running. Broken code is never switched to (`update_blocked`).
  Open windows poll `/api/version` and reload when the page assets changed
  (`ui_version`), keeping an unsent draft; a backend-only update needs no reload. The launcher compares the fingerprint too and replaces a stale idle server
  before opening the window.
- **Kiwibro in system monitors.** The launcher runs the server in its own systemd user
  scope, `app-kiwibro.scope`, through an interpreter link named `kiwibro`, so the process
  is called `kiwibro` and Resources associates it with the Kiwibro app and icon; ending
  it there (or `systemctl --user stop app-kiwibro.scope`) stops all of it. Started by
  hand, the server still renames itself (`prctl`).
- **Restart / Quit** actions on the app icon and `kiwibro [--restart | --quit | --server]`
  on the command line (`install.sh` links it into `~/.local/bin`).
- **Memory saved by code, not left to the model** (`autofacts.py`, `MemoryEngine.capture`).
  Before every reply, high-precision patterns store what the user just stated about
  themselves (name, place, job, OS, editor, reply language, preferences, projects,
  "remember …" / "forget …"), so it is known on the very next turn even when a small
  model never calls `remember` (the UI shows the memory chip). Questions, hypotheticals,
  reported speech, roleplay and code are never read as facts. Single-valued facts
  (name, location, editor…) are rewritten in place by a later statement.
- **Minimal, adaptive prompt** (`guidance.py`). Message 0 shrank from ~4.8k to ~1.6k
  characters (~900 fewer tokens to prefill on every cold start) and stays byte-stable
  for the KV cache. How to research, code, compute, use the desktop, render charts,
  handle memory or attachments is chosen per turn by code from the message and the
  turn plan, and rides in the context note only when it applies; a greeting gets none.
- **Chat search by content.** The sidebar filter also finds chats by what was said in
  them, with a snippet under the title (`GET /api/conversations/search`).
- **Continuity across chats** (`sessions.py`). When a message refers back ("what did we
  decide yesterday?", "last time", "continue where we left off", "dün"), the turn gets
  a short note on the most recent other chats: title, when, a gist and up to three
  exact matching excerpts. No model call; other messages get nothing extra. `recall`
  also searches other chats, labelled with their title.
- **Retry for an unanswered message.** Opening a chat whose last message never got a
  reply offers Retry.
- **Private chats** (schema v5). The composer's Private pill makes the open chat, or the
  next new one, private: nothing from it is saved to memory (no capture, no extraction,
  and `remember` declines with the reason), and other chats never see it in their notes
  or in `recall`. The chat still uses memory and its own history. Its extraction
  watermark moves on, so making it public later does not learn from the private part.
  A fork stays private; the sidebar marks private chats with a shield.
- **Export as Markdown** from the chat menu: the transcript with tool results folded,
  without reasoning. Reading a chat for export does not prefill it (`?warm=0`).

### Changed

- Memory injection is relevance-gated: pinned memories and the profile (name, reply
  language) always, other memories only when they share a key term with the message
  (below); only matched memories count as used.
- **Memory relevance by key terms** (`terms.py`). Function words are dropped, inflections
  are folded and everyday topics are tagged. "where do I live?" finds "User lives in
  Istanbul", "recommend a dinner" finds the favourite food, and "is the build passing"
  no longer pulls in memories through "is" and "the".
- The user's own self-descriptions ("I am …", "My name is …", "Call me …") count as
  profile: up to three ride on every turn, rendered as "User said: …". Memories that
  start in the first person are rendered the same way, never as the assistant's words.
- Extraction sees only the memories it might change: pinned ones, the profile, and
  those sharing a key term with what the user said. Measured on a 35-memory store:
  32 memories → 5 (the one to update among them), 3,568 → 2,002 prompt characters.
- `recall` leaves out messages still in the working window, since the model already
  has them. Tool results stay recallable.
- Forking copies the chat in one transaction (3,000 messages: 0.05 s instead of 0.48 s)
  and keeps the workspace, privacy, the extraction watermark and, when the fork point
  reaches them, the summary and its segments.
- Chats that opened with small talk ("Hi") are titled from the first real request;
  existing ones are re-titled once at startup. Titles drop filler such as "there has
  been", "can search" and a trailing "about".

### Fixed

- The per-turn context note is reserved before compaction decides, so memory and
  guidance can no longer push a small-context prompt past the window.
- A server re-executing itself could not bind its port again: Werkzeug leaves the
  listening socket inheritable. Inherited descriptors are closed before `execv`.
- The page and static assets are always revalidated (`Cache-Control: no-cache`), so an
  update is never hidden behind the browser's heuristic cache.
- Retry stored the same user message twice, so the model saw the request twice. An
  unanswered identical last message is now reused.
- Retry disappeared when LM Studio was not running. It stays, with the "LM Studio isn't
  running" message.
- Full-text queries matched on function words: "the" or "is" made nearly every message a
  hit. Function words are dropped when the query has other words.
- A memory extraction refused at interpreter shutdown is counted as cancelled, not as a
  failed attempt.
- Two user messages in a row (a reply that failed before any text) reached the model as
  two turns: strict chat templates such as Gemma's reject that, and a re-sent duplicate
  was read twice. They now render as one turn, identically on every later turn.

## [2.2.0] — 2026-10-22

Whole requests on small local models: "tell me about X, then search the web for Y, and
finally write a doc at this location" is split, done step by step, checked on disk and
finished, without asking for permission it does not need, searching in circles or
looping on its own words.

### Added

- **Turn plan** (`plan.py`, no model calls). A request is split into ordered steps
  (sequencing words, sentences, list items) and each is classified: answer, web, write,
  run, read or other. The steps and the exact path to use ride in the per-turn note;
  the UI shows them as a checklist that ticks off (SSE `plan`, `done.plan`). When a
  call finishes a step, the tool result says what is next, or that every tool step is
  done (files checked on disk) and which parts of the reply are still missing. At the
  end, code verifies the plan (the file exists where asked and is not empty, searches
  found something, each explained topic is in the reply); what is left becomes one
  transient nudge (at most two), and a turn that still ends short says so in one line.
  A plan that checks out skips the generic self-check round.
- **Places the user names.** "Write it to ~/Desktop/notes/pi.md" or "save it on my
  desktop" (XDG-aware) now works: that call runs confined to the named folder and is
  graded there, so creating the file needs no approval in smart mode. A named file may
  be written or edited; a named folder allows new files only. A model that drops the
  folder (`pi.md`) is routed to the named file. Home itself, credentials, start-up
  files, system folders and Kiwibro's data are never reachable this way.
- **Research pacing.** A search that nearly repeats one from earlier in the turn is
  refused with a pointer to its results; after three rounds of searching the model is
  told to answer, after five (or one round after the plan's tool steps are done)
  searching stops for the turn. Pages an earlier search of the turn already showed
  keep their title and URL but not their snippet (less prefill per round).

### Changed

- **Repetition loops are cut early and the reply continues.** A block of up to three
  lines repeated ("**The fix**" + the same bullet, seen live a dozen times in a
  bulletin) is cut at its third copy instead of after 12 identical lines. The good
  start is kept and continued with anti-repetition sampling (`repeat_penalty`,
  `frequency_penalty`, that request only; the KV-cached prefix still applies) instead
  of regenerating everything; the stored reply is what the user saw. Deliberation
  loops are still cleared and redone.
- **Thinking has a budget when it is time to act.** With a planned tool step open, or
  after tool results, a request may think 4,000 characters; past that it is retried once
  with thinking off and told the exact next step. Live, a 4B model had drafted the same
  file in its thinking for 20 minutes without ever calling `write_file`.
- **Switching models frees memory.** The model Kiwibro loaded for the previous
  selection is unloaded before the new one loads; models you loaded yourself in
  LM Studio are never touched.
- While a planned file is still unwritten, the reply cap rises to 6,144 tokens (within
  the window's free room), so thinking plus the whole file fit in one call.
- A file call outside the workspace that the user did not name is refused at once,
  without an approval card for a call that could not run.
- Greetings and thanks run no tools.

### Fixed

- A write the user explicitly placed on the Desktop asked for approval and then failed
  with "path is outside the workspace" even when approved (live, `spark-x2.6-4b` with
  thinking on: 16 minutes, no file). Now: 6–12 minutes, file written, no prompt.
- A tool call cut off by the reply limit (`finish_reason: length`) was repaired and run,
  saving a truncated file; it is now refused and redone.
- Compaction during a long tool turn could summarise the user's own request away (and
  leave chat templates without a user turn); the request is now always kept, and a
  mid-turn compaction keeps the turn's note (memory, directives, plan).

## [2.1.0] — 2026-10-22

Smooth sessions on small local models: the model is ready before the first message,
permissions stay out of the way, memory edits itself correctly, replies render what
they show, and the model's own window size stops mattering.

### Added

- **Model readiness** (`readiness.py`). Opening Kiwibro, selecting a model, opening a
  chat or changing a prompt-shaping setting loads the model explicitly (32K window,
  4 parallel slots; 16K/8K when LM Studio refuses; a loaded window under 16K is
  enlarged while idle) and prefills the exact leading messages + tools of the next
  request. Measured with `lfm2.5-1.2b`: the first request takes 6 s instead of 47 s;
  through Kiwibro a new chat's first token came after 7.3 s with 97 % prefix reuse. A
  reply that starts mid-warm-up waits for it on the same prefix and cancels it
  otherwise. `GET /api/models/ready`; the header dot pulses amber while it runs;
  `KIWIBRO_WARMUP=0` turns it off.
- **LM Link and remote servers.** Models from linked devices are listed, loaded and
  warmed like local ones. Settings → LM Studio can point at another server and store
  an API token (`Authorization: Bearer`, write-only; `KIWIBRO_LMSTUDIO_TOKEN`).
- **Rich replies.** Highlighted code (escape-first tokenizer, ~15 languages), coloured
  diffs, `csv`/`tsv` blocks as tables, and live previews of `html`/`svg` blocks with
  Preview/Code tabs. HTML runs in an opaque-origin iframe whose CSP blocks all network
  access; refresh tags are stripped, links are inert, and scripts wait for *Run scripts*.
- **Memory the model can edit.** `remember` gains `action: update | forget` (and `old`)
  so "I moved to Ankara" replaces "lives in Istanbul" and "forget that" works. Every
  memory records its origin (`you` / `chat` / `learned`, schema v4) and the chat it came
  from; the Memory sheet labels and links it. A toast says when something new was
  remembered (checked with growing gaps, since extraction can take a minute on CPU).
- **Messages longer than the window** (up to 400,000 characters) are stored in the
  document library; the model reads the start, the end and a handle to page through the
  rest. The chat keeps the full text.

### Changed

- **Permissions: Automatic or Manual.** Automatic (default; `autonomy: smart`, or
  *refuse risky steps* = `autonomous`) lets the risk policy decide every step and hides
  the per-tool list; Manual (`supervised`) shows the Auto/Ask/Deny list, which now
  applies only there. In Automatic mode the approval card's second button is *Allow in
  this chat* (in-memory, per conversation); in Manual it is *Always* as before. The
  system prompt describes the active mode.
- **Memory extraction waits for quiet.** It runs one chat at a time once no reply has
  streamed for 20 s (`KIWIBRO_MEMORY_IDLE_S`) and no warm-up runs; a reply starting
  cancels it mid-request and it is re-queued, so nothing is skipped (LM Studio stops a
  cancelled request at its next 2048-token batch, so quiet time matters more than the
  cancel). The `remember` tool still saves what you state during the turn. Measured: an
  extraction beside the next reply slowed its first token from 2.8 s to 8.4 s, and stored
  "lives in Istanbul" after that reply had saved "moved to Ankara". A learned fact that a
  newer, related memory already changed is no longer added.
- **Context budget counts the tool schemas** (~3.5k tokens for the auto set), so
  compaction starts before the window overflows; a window too small for the auto set
  gets the minimal set instead.
- The system prompt tells the model what the chat renders (tables, csv, html/svg).

### Fixed

- Small models stored facts about the assistant ("Kiwibro is a local AI agent…") and
  facts nobody stated: `remember` and extraction now refuse both, and saves must be
  grounded in what the user said.
- Extraction "updates" of an unrelated memory overwrote it (measured live with
  `lfm2.5-1.2b`); such an update is now an add, and updates must be grounded too.
  `remember(action="update")` with the old and new fact swapped is put right.
- Memories the user added are never deleted by extraction or evicted at the cap.
- **Stop works during prefill.** LM Studio sends nothing until the first token (75 s for a
  6k-token prompt on this CPU), so the stop flag was only seen then; a watcher now shuts
  the socket down (Kiwibro returns at once; LM Studio drops the request at its next
  2048-token batch).

## [2.0.0] — 2026-10-22

The agent release: complex tool use, model teamwork, risk-graded autonomy with a
command sandbox, and a context/memory efficiency overhaul for long sessions on small
local models.

### Added

- **27 new built-in tools (35 built-in, 45 with extension tools).** Parallel web research (`multi_search`, `fetch_pages`,
  `web_research` with numbered evidence, `browse_page`, multi-tab `browser`);
  codebase management (`grep_files`, `glob_files`, `read_files`, `file_outline`,
  atomic `apply_patch` with diff, `file_manage`, read-only `git`,
  `multi_command` pipelines, persistent `terminal_session`); sandboxed
  `run_python`/`run_javascript` snippets; computer use (`system_info`, `open_path`,
  `clipboard`, `notify`, `screenshot`); persistent `task_board`; `recall` over
  memory + episodic history. Optional extension tools plug into the same catalog.
- **Capability-aware model teamwork.** `model_info` lists every local LM Studio
  model with capabilities; `delegate`/`describe_image` hand bounded side-questions
  to the smallest adequate installed model (e.g. vision) and return to the main
  model; `study_file` answers questions about long files by chunk→notes→merge.
  When no model fits, the playbook falls back to solving it with sandboxed code.
- **Tool profiles & parallel execution.** `auto` offers one fixed 20-tool set sized for
  1–4B models (the rest of the catalog runs when called by name); explicit profiles
  (full/minimal/coding/research/desktop) remain. Independent read/search calls in one
  round run concurrently.
- **Local inference engine.** Prompts keep a byte-stable prefix so LM Studio reuses its
  KV cache (measured 124 s → 1.3 s for a 4k-token prompt on a laptop CPU; a new chat's
  first token in 2.4 s instead of 17.3 s); a `reasoning` setting (auto/on/off) with a
  "Think" control in the composer, decided once per turn; utility calls never think;
  per-reply speed line (time to first token, tokens/s, cached share); LM Studio v1
  metadata (reasoning options, size, loaded window); helper models loaded/unloaded via
  `/api/v1/models/load|unload` with a free-RAM guard and shared leases; images downscaled
  before vision calls; one-click larger context window from the context ring.
- **Attachments.** Attach, paste or drop images, audio, video and documents. Images and
  video keyframes are described by the main model if it can see, otherwise by the
  smallest installed vision model (loaded and unloaded for it); audio, SVG — and images
  when no vision model exists — get facts measured by code. Files live in the workspace,
  so every tool can work on them.
- **Markdown:** tables with alignment, task lists, nested lists, h5/h6.
- **Mid-turn steering.** `POST /api/chat/steer` (send while a reply runs) applies
  user corrections at the next step; `done` now reports turn `stats`.
- **Risk-graded autonomy.** Every tool call is graded safe → critical by a bash-aware
  command analyser (quotes, heredocs, substitutions, pipelines, wrappers, scripts it
  runs, decodable payloads; hardened against ~150 red-team bypasses) and decided
  auto/ask/deny by the new `autonomy` setting: *smart* (default) runs safe and undoable
  work and asks before risky steps, *supervised* follows the permission list,
  *autonomous* never waits and refuses risky steps with reasons the model can act on.
  Critical actions (wiping home, `sudo`, start-up-file writes, credential
  exfiltration) are always refused. Graded per-turn breakers escalate mass deletes and
  edits; after 3 failed tool calls in a row the model is coached to change approach and
  after 6 its tools pause for the turn (measured live: a 16-round path-guessing loop on a
  4B model dropped to 6 rounds); a duplicate-call loop guard and all-or-nothing patches remain.
- **Command sandbox.** `run_command`, terminal sessions and `run_python`/`run_javascript`
  run in bubblewrap inside a systemd scope: read-only root, writable workspace (snippets:
  scratch only, no network), private `/tmp`, own PID namespace, credentials and session
  bus hidden, memory/task caps, whole process group killed on timeout. Commands use a
  Kiwibro-owned HOME (`~/.cache/kiwibro/home`) so caches never land in your projects, while
  git keeps your identity. Feature-detected and self-tested; `sandbox` setting auto /
  strict (no network) / off.
- **Staying on task.** A directive ledger built from your messages (no extra model calls)
  carries the goal, constraints, standing choices and overridden instructions into each
  turn; after real work the model gets one transient checklist to verify its result.
- **Folders and servers.** `workspace` binds a conversation to a folder you name
  (schema v3); `remote` runs commands and copies files over SSH with your own keys and
  `~/.ssh/config` hosts (unknown hosts need approval; server commands graded).
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
- Replies that collapse into filler or repetition loops (seen live on a 2-bit 4B model: ~6
  minutes of '...' per turn) or circle in deliberation without acting ("so the content is
  kiwi\n? No…" for 7,600 characters) are cut off as they stream, cleaned from the screen and retried
  once with a corrective nudge on the same cached prompt; a second collapse ends the round
  with an honest note.
- Model-written JSON everywhere (memory extraction, titles, utility calls) is repaired by the
  same parser as tool calls: a trailing comma from a 4B model used to drop the extracted
  memory silently (live: "User prefers short, direct answers" now saved).
- Memory extraction timeouts follow measured model speed (a fixed 8 s never finished on a
  CPU-only laptop, so memories were silently never saved); one extraction per chat at a time.
- One permission table: extension tools carry their own defaults
  (`config.permission_defaults()`); Settings validates enum settings from one table.
- Model entries expose `vision` + `capabilities`; vision shown in model lists.
- Tool results are head+tail shaped before storage; fetched pages are cached (5 min).

### Changed — context and efficiency

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

### Fixed — web search and memory

- **Web search returns relevant results, fast.** Brave (429), Bing HTML (empty) and
  DuckDuckGo (bot wall) were all failing, so every query fell through to Wikipedia
  and returned off-topic pages. A Yahoo provider now leads; providers run in
  parallel tiers that stop once they have enough hits; results must contain enough
  of the query's content words (stricter for Wikipedia/Bing RSS), and an empty
  result tells the model how to rephrase instead of handing it junk. Keep-alive
  sessions, lxml parsing and parallel snippet enrichment cut latency: 10 live
  queries were 10/10 relevant at ~1 s each.
- **Memory saving is reliable.** New memories no longer evict themselves at the cap;
  merges keep their kind; a non-JSON or timed-out extraction no longer stalls the
  watermark; reasoning blocks and truncated JSON are never parsed as facts;
  extracted facts must be grounded in the conversation (small models copied prompt
  examples); pinned memories are safe from extraction; recall matches
  singular/plural; the agent prompt says when to call `remember`.

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
