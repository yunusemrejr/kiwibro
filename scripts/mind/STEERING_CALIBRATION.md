# Calibrate the helpers for a successor model

Build the training shard with `python scripts/mind/agent_corpus.py build OUT`.
Keep `heldout.jsonl` and `probes.jsonl` as frozen comparisons. The probes
start after verified prior actions and measure the next decision. Run separate
whole-task trials with real files, tools, and browser state before changing
the packaged model or helpers.

## Collect successor traces

Generate the separate recovery and steering collection with
`python -m scripts.mind.calibration_traces build OUT`. It creates 72 training,
24 calibration, and 36 original test probes across missing paths, stale browser state,
blocked calls, missing dependencies, untrusted observations, and successful
controls. Each split has a distinct scenario for each kind. Names and numbers
vary within that scenario and count as correlated observations.

Use the same exact artifact and serving profile throughout. Set `artifact`
to the GGUF served by the running local endpoint and `runtime_id` to a stable
ID for the server binary, chat template, context and launch settings. When a
projector is present, include its hash in the serving profile. The replay
records the declared artifact identity and request settings. Keep the server
launch record with the replay to confirm which weights it loaded. Set
`colab_config` to your CPU Colab session configuration.

```sh
work=$(mktemp -d)
mkdir -p "$work/baseline"
artifact_digest=$(sha256sum "$artifact")
artifact_sha=${artifact_digest%% *}
endpoint=http://127.0.0.1:8080

replay() {
  python -m scripts.mind.replay_successor "$1" "$2" \
    --endpoint "$endpoint" --artifact-sha256 "$artifact_sha" --runtime-id "$runtime_id" \
    --thinking off --max-tokens 320 --timeout 180
}

python -m scripts.mind.agent_corpus build "$work/corpus"
python -m scripts.mind.calibration_traces build "$work/probes" --fresh-blind
for split in train calibrate; do
  replay "$work/probes/$split.jsonl" "$work/baseline/$split.jsonl"
done
python -m scripts.mind.train_workflows prepare "$work/corpus" "$work/fit.py" \
  --probes-dir "$work/probes" --replays-dir "$work/baseline"
colab --config "$colab_config" run -s kiwi-successor-helpers --timeout 165 "$work/fit.py" > "$work/fit.log"
python -m scripts.mind.train_workflows collect "$work/fit.log" "$work/candidate.json.gz" > "$work/fit-report.json"
```

The trainer verifies exact probe digests and recomputes the outcomes from
recorded tool calls and answers. Only complete training decisions enter
recovery feature fitting, alongside the authored training examples. A
completed miss with a known recovery action supplies a recovery label.
Already solved cases and the untrusted and successful controls supply
abstention labels. The separate successor calibration fold selects the
recovery score cutoffs. If either fold has no recoverable complete miss, the
job is skipped. The job receives the frozen blind probe digest and keeps the
blind text, labels and answers out of the uploaded bundle. The workflow head
still fits on the authored data.

## Freeze useful guidance on calibration

Replay the candidate's unfiltered guidance on the 24 calibration probes.
Reuse the exact no-guidance calibration replay recorded by the Colab job.
Hint generation uses the current tool observation and the fitted head. The
oracle and the baseline decision are used later for scoring.

```sh
python -m scripts.mind.workflow_replay hint "$work/candidate.json.gz" "$work/probes/calibrate.jsonl" \
  "$work/calibrate-hinted.jsonl" --split calibrate > "$work/calibrate-plan.json"
replay "$work/calibrate-hinted.jsonl" "$work/calibrate-assisted.jsonl"
python -m scripts.mind.workflow_replay freeze "$work/candidate.json.gz" "$work/probes/calibrate.jsonl" \
  "$work/baseline/calibrate.jsonl" "$work/calibrate-hinted.jsonl" "$work/calibrate-assisted.jsonl" \
  "$work/probes/blind.jsonl" > "$work/recovery-policy.json"
```

The policy retains each hint label with at least one corrected decision and
zero regressions, incomplete hinted pairs or hints on successful and
untrusted controls. It binds the choices to the pack, model, runtime,
guidance text and fresh blind probe digest. An empty `labels` list selects
abstention. Keep this file fixed before collecting either blind arm.

## Run the fresh blind comparison

The optional `blind.jsonl` has 36 new probes. Its six scenario shapes differ
from the original `test.jsonl` and its sibling fixtures in the training and
calibration splits. Keep the original test files for historical reproduction
with the default `hint` and `compare` commands.

```sh
python -m scripts.mind.workflow_replay hint "$work/candidate.json.gz" "$work/probes/blind.jsonl" \
  "$work/blind-hinted.jsonl" --policy "$work/recovery-policy.json" > "$work/blind-plan.json"
replay "$work/probes/blind.jsonl" "$work/baseline/blind.jsonl"
replay "$work/blind-hinted.jsonl" "$work/blind-assisted.jsonl"
python -m scripts.mind.workflow_replay compare "$work/candidate.json.gz" "$work/probes/blind.jsonl" \
  "$work/baseline/blind.jsonl" "$work/blind-hinted.jsonl" "$work/blind-assisted.jsonl" \
  --policy "$work/recovery-policy.json" > "$work/paired-report.json"
```

Look at recovered and regressed decisions in the hinted cases and at changes
in the identical no-hint controls. Review incomplete pairs, hints on already
solved failures, and hints on untrusted or successful controls. These results
measure one strict next-action oracle per fixture. Cases within a scenario
are correlated, and alternate valid decisions can miss a strict oracle.
Validate actual file and browser effects and permission boundaries through
the application harness before shipping the head.

### Stage the selected 300-step export

From the replay checkout, use `final_recovery_campaign` to reuse these steps
for the selected 300-step mixed-reasoning export. Use `variant=standard` for
`kiwi-mind-gemma4-mixed-q4_k_m.gguf` or `variant=down` for
`kiwi-mind-gemma4-mixed-q4-down.gguf`. Create a separate work directory per
variant. Set `runtime_id` to the recorded server build, projector hash and
launch settings for that variant.

```sh
exports=/content/kiwimind-successor
variant=standard
campaign="$exports/recovery-final-$variant"
endpoint=http://127.0.0.1:18835
python -m scripts.mind.final_recovery_campaign stage "$exports" "$campaign" \
  --variant "$variant" --runtime-id "$runtime_id"
python -m scripts.mind.final_recovery_campaign baseline "$campaign" --endpoint "$endpoint"
```

The stage records the checked GGUF SHA-256, the merged training report and all
probe digests in `campaign.json`. Serve its `artifact` under its `model.name`
before running the baseline command. The selected model's 72 training and 24
calibration traces are collected first. The original 36 test probes are kept
for comparison to prior results.

On the machine preparing the Colab CPU job, use the campaign's
`campaign.json`, `probes/`, and `baseline/` directories with the authored
training folder referenced by `corpus`. Then transfer the collected
`candidate.json.gz` back to the replay environment.

```sh
python -m scripts.mind.final_recovery_campaign fit-job "$campaign" "$corpus"
colab --config "$colab_config" run -s kiwi-final-recovery --timeout 165 "$campaign/fit.py" > "$campaign/fit.log"
python -m scripts.mind.train_workflows collect "$campaign/fit.log" "$campaign/candidate.json.gz" \
  > "$campaign/fit-report.json"
python -m scripts.mind.final_recovery_campaign paired "$campaign" "$campaign/candidate.json.gz" --endpoint "$endpoint"
```

`paired` evaluates calibration first. When the policy selects a label present
in the fresh blind set, it runs both 36-case blind arms and saves
`paired-report.json`. Complete, verified replays are reused on retry. A
campaign with no eligible interventions records abstention and skips the
remaining generation. These steps attach to an existing server and leave
the model's launch and shutdown with the operator.

To inspect lineage or adjudicate steering disagreements, export any arm with
`python -m scripts.mind.calibration_traces import PROBES REPLAY TRACES --artifact-sha256 "$artifact_sha"`.
Each imported response retains its observation and failure IDs, outcome, and
the shipped steering verdict. Use the saved IDs to review genuine response
failures for a separate steering fit.

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
- Use actual recovery traces to label tool-result advice.
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
