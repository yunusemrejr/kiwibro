# Calibrate the helpers for a successor model

Build the training shard with `python scripts/mind/agent_corpus.py build OUT`.
Keep `heldout.jsonl` and `probes.jsonl` as frozen comparisons. The probes
start after verified prior actions and measure the next decision. Run separate
whole-task trials with real files, tools, and browser state before changing
the packaged model or helpers.

## Collect successor traces

Record the exact model and artifact hash, serving runtime, chat template,
quantization, context cap, tool profile, seed, and sampling options. For each
run record the task family, first 500 streamed characters, final answer, tool
calls and their result states, permission decisions, first visible token time,
total time, and artifact checks. Preserve selected-helper responses as tool
observations. Use synthetic or approved public tasks and remove secrets from
all recordings.

Create a development split for fitting, a calibration split for confidence and
margin selection, and a blind split for the final comparison. Group paraphrases,
trajectories from the same fixture, repeated seed runs, and equivalent changes
to the same repository in a single split.

## Label and fit on Colab

- Label the reply probe's sentence form with the existing classes in
  `scripts/sense/corpus.py`. Record context separately, including tool success
  and whether the tool was available. Include correct refusals, completed
  actions, quoted misuse, and genuine capability limits as controls.
- Use actual recovery traces to label workflow and tool-result advice.
  A wrong path, a stale browser observation, an unavailable named helper, and a
  rejected action have different safe next steps. Reuse the runtime tokenizer
  and tool schemas, and keep all advice contingent on the current request and
  permissions.
- For the Kiwi Sense message, shift, extraction-tagging, and relevance heads,
  add examples where the new model needed corrective context or where relevant
  context was suppressed. Use the labels and ABI already in
  `scripts/helpers/train.py`; run each export through the Python and WASM
  readers. The English language gate remains independently evaluated.
- Run optional workflow fitting through `scripts/mind/train_workflows.py`
  using its Colab CPU job and retain the tagged runtime, report, input hashes,
  and quantized candidate weights. Use the same process to retain previous and
  successor steering bundles for paired tests.

## Compare before switching

Run no learned advice, the shipped helpers, and the proposed helpers on matched
tasks. Measure final task success, wrong guidance, unnecessary interruptions,
recovered failures, tool-policy violations, calls and tokens per successful
task, and warm p50/p95 overhead. Review failures by family. Check that any
confidence change preserves rejection of untrusted instructions and that
missing, corrupt, or rejected weights leave the ordinary tool paths usable.
The agent's deterministic tool validation, permission checks, memory
grounding, and artifact verification remain authoritative in every arm.
