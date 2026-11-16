# Calibrate the helpers for a successor model

Build the training shard with `python scripts/mind/agent_corpus.py build OUT`.
Keep `heldout.jsonl` and `probes.jsonl` as frozen comparisons. The probes
start after verified prior actions and measure the next decision. Run separate
whole-task trials with real files, tools, and browser state before changing
the packaged model or helpers.

## Collect successor traces

Generate the separate recovery and steering collection with
`python -m scripts.mind.calibration_traces build OUT`. It creates 72 training,
24 calibration, and 36 blind probes across missing paths, stale browser state,
blocked calls, missing dependencies, untrusted observations, and successful
controls. Each split has a distinct scenario for each kind. Names and numbers
vary within that scenario and count as correlated observations.

Reuse the candidate's running local endpoint for one split at a time:

```sh
python -m scripts.mind.replay_successor OUT/train.jsonl TRAIN-REPLAY.jsonl \
  --endpoint http://127.0.0.1:8080 --thinking off --max-tokens 320
python -m scripts.mind.calibration_traces import OUT/train.jsonl TRAIN-REPLAY.jsonl TRAIN-TRACES.jsonl \
  --artifact-sha256 ACTUAL_ARTIFACT_SHA256
```

Run the same commands for calibration and blind test. Keep the split files
separate in storage. Each imported response keeps the original tool
observation and stable observation ID, with a failure ID for a failed,
denied, unavailable, or stale action. The importer validates the frozen probe
digest and recomputes the next-action outcome. It records the shipped steering
verdict beside the outcome so an adjudicator can inspect disagreements.

On Colab, use only reviewed training traces for fitting. Use the 24-case
calibration split for score cutoffs and abstention. Retain the 36-case split
for one frozen comparison. Associate every training sentence with its
observed response and failure ID. Separate interrupted or length-limited
generation from errors in answer grounding or tool choice. These probes
measure six recovery scenarios per split. Broader approval, safety,
and real file or browser workflows still need independent held-out tasks.

Use `python -m scripts.mind.calibration_traces smoke REPORT OUTPUT` to check an
existing native benchmark report against its original tasks. This path exports
parser controls only. Warm and cold traces from a single case share one group.

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
