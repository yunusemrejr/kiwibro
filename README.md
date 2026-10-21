<div align="center">

# Kiwibro

**A local AI agent for Linux — quiet, capable, and yours.**

`Flask` · `LM Studio` · `Native-feeling dark GUI` · `No cloud`

</div>

> Work in progress. Full README, screenshots and install guide land with v1.0.

## What it is

Kiwibro is a local-first AI agent that runs on your machine and talks to
[LM Studio](https://lmstudio.ai). It picks your model, manages context
automatically, remembers what matters, and can search the web and write code —
with your permission.

- **Pick any LM Studio model** — swap instantly, mid-chat.
- **Context that never breaks** — automatic rolling compaction, no limit errors.
- **Memory that stays organised** — extracted, deduped, editable, model-independent.
- **Agent tools with a permission gate** — web search, page reading, file edits and
  shell commands; reads are automatic, writes and commands ask first.
- **Native Ubuntu feel** — a dark GNOME-style window, kiwi icon, `.desktop` launcher.
- **100% local** — SQLite on disk, browser GUI from localhost, no telemetry.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

Open <http://127.0.0.1:8777>. Start [LM Studio](https://lmstudio.ai) with a chat
model loaded (or let Kiwibro just-in-time load one).

## License

MIT © 2026 Yunus Emre Vurgun
