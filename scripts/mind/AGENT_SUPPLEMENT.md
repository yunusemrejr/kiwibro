# Successor training supplement

This training shard contains new scenarios for six recurring decision failures.
The default build produces 40 episodes across 10 authored scenario shapes,
including 20 reasoning-mode episodes. Variants from the same scenario share
structure and stay in training.

- Exact helper selection covers an unavailable sole recipient, an authorized named substitute and a local calculator fallback.
- Web evidence covers a confirmed fee with an unsupported refund claim, and an equipment rule absent from the linked policy and help pages.
- Memory tasks require finishing both the saved memory and a separately verified artifact or source reference.
- Visual repair uses one selector edit, a clean 360 px preview and an immediate stop.
- A DC energy calculation keeps missing current symbolic and computes the coefficient from two instrument logs.
- A retired TSV column produces a real typed tool error, followed by a profile call and the corrected aggregate.

Build and check the shard from the repository root.

```sh
work=$(mktemp -d)
python3 -m scripts.mind.agent_corpus build "$work/base"
python3 -m scripts.mind.agent_supplement build "$work/supplement" --verify-render
python3 -m scripts.mind.agent_supplement check "$work/supplement"
cat "$work/base/train.jsonl" "$work/supplement/train.jsonl" > "$work/base/fit-train.jsonl"
```

Use `"$work/base/fit-train.jsonl"` as the Gemma fit input. This path keeps the
base training images relative to the merged input. The separate supplement's
manifest records the training shard SHA-256 and counts. The base holdout and
next-action probes serve as frozen evaluation. Supplemental families have
their own names, so the fitter's per-family sampler can retain the new
trajectories explicitly.

Each `thinking: true` row has a short, evidence-based `reasoning` field on every
assistant target. Plain rows omit both fields. Keep tool-call arguments typed
through preprocessing and use the native template for the row's declared mode.

Table observations come from KiwiBro's real table tool on fixture TSVs. The
optional Chrome check exercises the startup error, the single patch, the
repaired phone preview and the button. The training trajectory records the
app's preview alongside the edit result. The delegate, web and memory
observations are authored. All domains are fictional, and the build accesses
local synthetic files.
