# Kiwibro's independent memory system

Kiwibro's memory does not live in the model's context window. The window is
only a *view*: everything durable sits in SQLite/FTS5 on disk, and each turn
assembles the smallest prompt that still contains what matters. That is what
lets a 1–4B model with a 4k window hold a conversation of thousands of turns
without losing facts or slowing down — per-turn work is O(window), never
O(history), and no layer keeps growing RAM structures.

## The three tiers

| Tier | What | Where | Prompt share |
| --- | --- | --- | --- |
| 2 · Durable memory | Facts about the user across all chats | `memories` + FTS5 | scaled cap (`memory_token_cap`, 150–800 tok) |
| 1 · Episodic memory | Every message ever sent, searchable; per-range summaries | `messages_fts`, `summary_segments` | digest in system prompt + recalled excerpts ≤ budget/8 |
| 0 · Working window | Newest verbatim turns | `messages` after `summary_through_seq` | the remainder of the budget |

Nothing is ever deleted by compaction. Compaction only changes what is
*sent*; Tier 1 can always bring exact wording back.

## How a turn is assembled (`kiwibro/context.py`)

1. **Budget** from the model's real context (`compute_budget`), calibrated
   against server-reported usage.
2. **Segmented summarisation** when over the trigger: the oldest needed
   prefix is summarised in chunks sized by `utility_limits(ctx)` — the
   summariser call itself always fits (max ~1.4k prompt tokens on a 4k
   model; v1 sent a fixed ~4k and silently fell back to trimming). Each
   chunk persists as a `summary_segments` row; partials merge through a
   bounded tree reduce into the rolling digest (`conversations.summary`).
   All-or-nothing: history never advances past unsummarised content.
   Hysteresis targets 60% of the trigger so compaction (a full KV
   re-prefill) stays rare.
3. **Episodic recall**: the newest user messages query `messages_fts`
   (bm25) for messages *older than the window*; exact excerpts plus the
   original-task anchor are appended to the newest user message only —
   volatile text never enters the system prompt, keeping the server's KV
   prefix byte-stable between turns. Tool-result caps likewise key on
   user-turn age with a single transition.
4. **Durable memory**: ranked retrieval (pinned → query-token coverage →
   Jaccard → use count → recency) rendered within the scaled cap.

## Durable memory (`kiwibro/memory.py`)

- **Code saves first, the model is the backup** (`kiwibro/autofacts.py`,
  `MemoryEngine.capture`). Before every reply, high-precision patterns read what the
  user just stated about themselves (name, location, hometown, job, employer, OS,
  editor, reply / native language, favourites, preferences, projects, "remember …",
  "forget …") and store it at once (origin `learned`, SSE `memory`), so the next turn
  already knows it even when the model never calls `remember`. Questions,
  hypotheticals ("if I…"), reported speech ("my friend said…"), roleplay and code
  never yield a fact. Single-valued facts have a slot with a canonical wording
  ("User lives in …"): a later statement rewrites that memory instead of adding a
  contradicting one. LLM extraction still runs when the user is idle, for what
  patterns cannot read.
- **What rides along each turn.** Pinned memories and the profile (name, preferred
  reply language) always, plus up to three self-descriptions the user wrote in their
  own words ("I am …", "I'm …", "My name is …", "Call me …"), rendered as `User said: …`
  so they are never read as the assistant's words. Every other memory only when it
  shares a *key term* with the message. Key terms (`kiwibro/terms.py`) drop function
  words, fold inflections to one stem (lives, living, lived → `liv`) and tag a few
  everyday topics (food, home, job, family, health and others), so "where do I live?"
  finds "User lives in Istanbul". The 8 recently used ones are no longer injected
  regardless of relevance. Only memories that matched count as used, so `use_count` no
  longer rewards whatever happened to be injected.
- **Extraction watermark**: `extracted_through_seq` advances only after a
  successful extraction response, so a timed-out small model retries next
  turn instead of silently skipping facts. The extraction timeout is sized
  from the model's measured prefill/decode speed (`inference.SPEEDS`, 8 s
  floor, 180 s ceiling) so a CPU-only laptop actually finishes it, and only
  one extraction runs at a time, only while no reply streams (≥ 20 s of quiet) and
  no warm-up runs; a reply starting cancels it and re-queues the chat. The extractor
  sees the oldest
  pending window and only the existing memories it might change: pinned ones, the
  profile, and those sharing a key term with what the user said (at most 32). A new
  fact that duplicates a stored one is still caught in code (`find_similar`), so
  unrelated memories stay out of the prompt.
- **Dedupe**: FTS-prefiltered candidates compared on stored normalised text
  (~2 ms at 400 memories, proven equivalent to the old full scan).
- **One recall engine**: `MemoryEngine.recall(query, conversation_id, limit)`
  merges durable memories and conversation episodes for both prompt
  injection and the `recall` tool — one behaviour, two surfaces. Episodes leave out
  messages still in the working window (the model already has them); tool results stay
  recallable. The user's other chats are searched too, each item labelled with its
  chat's title.
- Memories carry `use_count`; pruning evicts unused low-confidence entries
  first. Everything is user-editable/deletable; model swaps never touch it.
- **Origins** (schema v4): `you` (Memory sheet, or edited there), `chat` (the
  `remember` tool), `learned` (extraction), with the source conversation. The
  user's memories are never deleted by extraction or evicted by pruning.
- **The model edits memory, carefully.** `remember` saves, `update`s (`old` names
  the outdated memory; ≥ 50 % of its key words must match) or `forget`s. A save or
  update must be grounded in what the user said, and facts about the assistant are
  refused. Measured small-model mistakes are corrected rather than stored: an
  extraction "update" of an unrelated memory becomes an add (`memory.related`), and
  a `remember` update whose old/new facts are swapped is turned around (the newest
  user message carries the new fact). A learned fact is skipped when a related memory
  about the same subject changed after the message that stated it (`_superseded`:
  "moved to Ankara" saved by a later turn beats "lives in Istanbul" from an earlier one).

## Across chats (`kiwibro/sessions.py`)

Each chat's prompt holds only that chat. When a message points back in time, the turn
gets a short note about the other chats. Otherwise nothing is added.

- **Triggers.** `refers_back` matches phrases such as "yesterday", "last time", "the
  other chat", "we discussed", "you said", "continue where we left off", and the
  Turkish "dün" and "geçen sefer".
- **Contents.** "Your other recent chats": up to five, newest first, each with its
  title, when (today, yesterday, N days ago, or the date) and a gist. The gist is the
  rolling summary when the chat has one, else its first real request and last reply.
  Then up to three exact excerpts from other chats that match the message: replies
  first, duplicates dropped, each cut to 360 characters.
- **Budget.** At most 500 tokens and at most a tenth of the model's budget. Below 60
  tokens nothing is added.
- **No model call.** Gists come from stored text, so the note costs no generation.
- **Private chats** (schema v5) are never listed, excerpted or recalled in another chat,
  and nothing from them is saved: no capture, no extraction (the watermark moves on, so
  making a chat public later does not learn from its private part), and `remember`
  declines. A private chat still reads memory and its own history.

## Capability routing (`kiwibro/routing.py`)

When the main model lacks a capability, `run_helper(llm, need, messages)`
runs the smallest adequate installed model once (loaded first), then stops
it again if we loaded it — never the main or a previously loaded model.
Needs: `vision`, `audio`, `reasoning`, `coding`, `long_context`, `fast`,
`embeddings`, `tools`. No match → typed `no_model` error, and the agent
solves the problem with code/tools instead. Capability facts come from
`kiwibro/capabilities.py`; serving primitives (`catalog`,
`complete_vision`, `embed`, best-effort `load_model`/`unload_model`) live
in `kiwibro/llm.py` — routing is only ranking + lifecycle.

## Invariants (all test-pinned)

- No summariser/extraction call ever exceeds the model's context.
- Compaction never advances past unsummarised history; trimming is the
  last resort and is announced, and even then messages remain searchable.
- The rendered prefix (system + older turns) is byte-identical between
  consecutive turns without compaction (KV reuse).
- The recall note never pushes a prompt over budget; recall failure
  degrades to summary-only, never to a failed turn.
- Extraction failures retry; memory writes dedupe; helpers never unload
  the main model.
- Model-written memories are grounded in the user's words; an update only
  rewrites a related memory; `you` and pinned memories are never removed by
  the model or by pruning.
- A user message longer than ~40 % of the budget reaches the model as a fixed
  excerpt + library handle (`meta.prompt_text`), so it never overflows the
  window and still renders identically on every later turn.
- Retrying an unanswered message never stores it twice.
- Function words alone pull no memory into a prompt.
- Extraction sees only the memories it might change.
- Recall leaves out messages still in the working window.
- A message that does not refer back gets no earlier-chats note.
- A forked chat keeps every seq; it keeps the summary and its segments only when the
  cut reaches them.
- A title the user set is never replaced by automatic titling.

## Long sessions in the API

`GET /api/conversations/<id>?limit=N&before_seq=S` pages messages keyset
style (`page: {has_more, oldest_seq}`); without `limit` the response is
unchanged. Verified live against LM Studio with a 4B model: a fact
compacted out of the verbatim window was answered exactly via recall.
