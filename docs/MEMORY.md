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

- **Extraction watermark**: `extracted_through_seq` advances only after a
  successful extraction response, so a timed-out small model retries next
  turn instead of silently skipping facts. The extraction timeout is sized
  from the model's measured prefill/decode speed (`inference.SPEEDS`, 8 s
  floor, 180 s ceiling) so a CPU-only laptop actually finishes it, and only
  one extraction per conversation runs at a time. The extractor sees the oldest
  pending window and only *relevant* existing memories (pinned + search +
  recent, char-capped) — small, accurate utility prompts.
- **Dedupe**: FTS-prefiltered candidates compared on stored normalised text
  (~2 ms at 400 memories, proven equivalent to the old full scan).
- **One recall engine**: `MemoryEngine.recall(query, conversation_id, limit)`
  merges durable memories and conversation episodes for both prompt
  injection and the `recall` tool — one behaviour, two surfaces.
- Memories carry `use_count`; pruning evicts unused low-confidence entries
  first. Everything is user-editable/deletable; model swaps never touch it.

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

## Long sessions in the API

`GET /api/conversations/<id>?limit=N&before_seq=S` pages messages keyset
style (`page: {has_more, oldest_seq}`); without `limit` the response is
unchanged. Verified live against LM Studio with a 4B model: a fact
compacted out of the verbatim window was answered exactly via recall.
