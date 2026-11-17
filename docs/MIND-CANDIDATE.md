# Native Kiwi Mind candidate

The candidate runs a checked GGUF pack through its native llama.cpp chat template. Gemma 4
26B A4B is a sparse mixture of experts. A learned router selects a subset of expert networks
for each token. Its vision projector provides image input and its chat template supports
structured tool calls. [Gemma 4 model card](https://huggingface.co/google/gemma-4-26B-A4B-it)

KiwiBro uses its existing tool catalog, argument validation, approval controls, evidence
checks, and SQLite memory and notes. Chat, vision and schema-constrained memory requests share
the authenticated local engine. A chat marked Private skips memory capture and extraction.
[Architecture](ARCHITECTURE.md) · [Memory](MEMORY.md)

## Stage and run an isolated candidate

Use a llama.cpp build that supports the candidate GGUF architecture and its native chat
template. Choose a GGUF and its matching projector from the same candidate build. Record
their source revision and check their SHA-256 values against the artifact publisher's
checksums before staging. [llama.cpp multimodal support](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)

From the repository root, stage the verified files. The command reads context, vision
dimensions, quantization and expert counts from the GGUF headers. It checks the supplied
digests and links the files into the isolated pack.

```bash
export KIWIBRO_CANDIDATE_STAGE="$(mktemp -d -t kiwibro-candidate-XXXXXXXX)"
export KIWIBRO_CANDIDATE_WEIGHTS=/absolute/path/to/verified-model.gguf
export KIWIBRO_CANDIDATE_PROJECTOR=/absolute/path/to/matching-mmproj.gguf
export KIWIBRO_CANDIDATE_MODEL_SHA256=publisher_model_sha256
export KIWIBRO_CANDIDATE_VISION_SHA256=publisher_projector_sha256

python3 -m scripts.mind.stage_native_pack \
  "$KIWIBRO_CANDIDATE_WEIGHTS" "$KIWIBRO_CANDIDATE_PROJECTOR" \
  "$KIWIBRO_CANDIDATE_STAGE/mind/candidate" --version candidate-local --context 8192 \
  --model-sha256 "$KIWIBRO_CANDIDATE_MODEL_SHA256" \
  --projector-sha256 "$KIWIBRO_CANDIDATE_VISION_SHA256"

KIWIMIND_CANDIDATE=1 KIWIBRO_DATA_DIR="$KIWIBRO_CANDIDATE_STAGE" \
  KIWIMIND_AUTOSTART=0 KIWIBRO_WARMUP=0 KIWIBRO_HOST=127.0.0.1 KIWIBRO_PORT=8778 \
  ./scripts/run.sh
```

Open [the isolated app](http://127.0.0.1:8778), or inspect its checked pack metadata with
`curl --fail --silent --show-error http://127.0.0.1:8778/api/mind`. Expect
`engine: "native-gguf"` and `active: "candidate"` when the pack can run.
Before the first request, `running` can be false. A missing pack, corrupt file or unavailable
runtime appears in `reason`.

If Vulkan reports that the graphics device was lost, the candidate switches to CPU
for the app session. It retries the current request once when generation has yet to
produce any answer or reasoning text.
For the managed b12443 runtime, `KIWIMIND_CPU_DRAFT=1` enables the experimental
CPU mode that drafts from repeated text in the current context. The model verifies
the tokens as it generates the reply.
[Ryzen laptop measurements](../scripts/mind/NATIVE_CPU.md#repeated-text-on-cpu)

For a compacted Gemma pack, pass `--architecture path/to/architecture.json` from
[the expert export](../scripts/mind/prune_gemma4.py). For a fitted checkpoint, also
supply `--training path/to/training.json` from the completed Colab run. Staging
checks the expert report against the actual GGUF. How it works reads the checked
expert counts, intermediate expert width when compacted, file sizes and context
from this pack.

To stage a task LoRA, add `--adapter`, `--adapter-sha256`,
`--adapter-trunk-sha256`, `--adapter-task code` and an optional
`--adapter-scale 0.8` to the staging command. The LoRA's export provenance must
identify the exact serving trunk SHA-256. Kiwi Mind loads the registered adapter
at zero scale through the runtime's `--lora-scaled` flag and selects its scale per
request using the existing task router. If adapter startup fails or that flag is
unavailable, the managed engine starts with the verified trunk.
[llama.cpp per-request LoRA settings](https://github.com/ggml-org/llama.cpp/blob/301db633702f1d2c107d4feaa4268f07c0b508c5/tools/server/README.md)

## Train and export on Colab

Build the [KiwiBro curriculum](../scripts/mind/agent_corpus.py) with
`python3 -m scripts.mind.agent_corpus build corpus --verify-fixtures`. The training
input is `corpus/train.jsonl` with its referenced images. Keep the held-out conversations and
next-action probes outside the training input.
Training rows with `thinking: true` use the native reasoning template and must supply
brief evidence-based reasoning on every assistant turn. Other rows use the compact
non-thinking template.

The L4 recipe uses the [pinned Gemma checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52)
in `reference/`, a BF16-capable Colab CUDA runtime, Transformers 5.23.0, Accelerate,
Safetensors and PEFT 0.21.3. The fit stores frozen expert weights in FP8 on the GPU and
uses BF16 for their computation. It projects the answer into the vocabulary 16 tokens
at a time to bound memory during backward.

```bash
python3 -m scripts.mind.calibrate_gemma4 corpus/train.jsonl router-profile.json \
  --checkpoint reference --device auto --gpu-budget 18GiB --per-family 2
python3 -m scripts.mind.prune_gemma4 reference router-profile.json compact \
  --retain 80 --top-k 6
python3 -m scripts.mind.fit_gemma4 compact corpus/train.jsonl adapted \
  --mode l4-lora --steps 28 --per-family 2 --max-tokens 3072 --lr 3e-5
python3 -m scripts.mind.merge_gemma4 compact adapted merged
python3 llama.cpp/convert_hf_to_gguf.py merged --outtype bf16 --outfile merged-bf16.gguf
llama-quantize merged-bf16.gguf kiwimind-q4_k_m.gguf Q4_K_M
sha256sum kiwimind-q4_k_m.gguf
```

The 28-step fit checks that all 14 workflow families can train and export. To extend that
fit, pass `--init-adapter adapted` with a fresh output directory, step count and seed.
Each continuation starts with the prior adapter weights and a fresh optimizer.
The merge uses the compact BF16 checkpoint and the fitted adapter. Use llama.cpp
build b12443 (commit `301db6337`) for a matching converter, quantizer and replay server.
If PEFT rejects the TorchAO version bundled in the Colab image, run
`python -m pip uninstall -y torchao` in that runtime and retry the CPU BF16 merge.
Retrain the routing and recovery helpers on the new model's traces with the
[Colab calibration workflow](../scripts/mind/STEERING_CALIBRATION.md).

### 300-step Colab run

The 300-step candidate continued the 28-step adapter for 112 steps on the
original curriculum. It then trained for 96 steps and 64 steps on 84 sampled
original episodes and 40 new episodes. Twenty of the supplementary episodes
included native reasoning targets. The model retained 80 experts per layer,
six active per token and a width of 704 per expert. Build the mixed input
with the [supplement recipe](../scripts/mind/AGENT_SUPPLEMENT.md#colab-mixture).

Run these commands on a Colab L4 after the pilot above. Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as part of the Colab fit
environment.

```bash
python3 -m scripts.mind.fit_gemma4 compact corpus/train.jsonl continued \
  --mode l4-lora --init-adapter adapted --steps 112 --per-family 16 \
  --max-tokens 3072 --seed 4081 --lr 2e-5
python3 -m scripts.mind.merge_gemma4 compact continued continued-merged
python3 -m scripts.mind.fit_gemma4 continued-merged corpus/mixed-train.jsonl mixed-96 \
  --mode l4-lora --steps 96 --per-family 96 --max-tokens 3072 --seed 5511 --lr 1.5e-5
python3 -m scripts.mind.fit_gemma4 continued-merged corpus/mixed-train.jsonl mixed-160 \
  --mode l4-lora --init-adapter mixed-96 --steps 64 --per-family 96 \
  --max-tokens 3072 --seed 5513 --lr 1e-5
python3 -m scripts.mind.merge_gemma4 continued-merged mixed-160 mixed-merged
python3 llama.cpp/convert_hf_to_gguf.py mixed-merged --outtype bf16 --outfile mixed-bf16.gguf
llama-quantize mixed-bf16.gguf kiwi-mind-gemma4-mixed-q4_k_m.gguf Q4_K_M
llama-quantize --tensor-type '^blk\.[0-9]+\.ffn_down_exps\.weight$=q4_0' \
  mixed-bf16.gguf kiwi-mind-gemma4-mixed-q4-down.gguf Q4_K_M
```

Compare both exports and the stock base with the same projector, held-out
probes, reasoning settings and token budget. The next-action replay and the
[native app acceptance suite](MIND-FINALIST-ACCEPTANCE.md) measure different
parts of the workflow.

### Matched replay results

The published 300-step standard pack includes the text weights, matching vision
projector, SHA-256 manifest, architecture report and training receipt. This
Hugging Face revision is a candidate build under evaluation.
[Published checkpoint](https://huggingface.co/yunusemrejr/kiwimind/tree/95b697df83df010340724363ffad535b6ce93c1f/candidate)

The pinned Colab restore bundle contains the intermediate adapter weights,
router calibration and export sources for rebuilding the 300-step checkpoint.
Its SHA-256 is `f464a43c6a9ebbbf40d911f7e3ddf5cf1fd294cbf77f9897dcc633972443c040`.
[Colab restore bundle](https://huggingface.co/yunusemrejr/kiwimind/blob/26ff7bb6fc8dd1386086247918a3e2585a3d8175/experiments/training/restore-300.tar.gz)

The three versions ran serially on the same Colab L4 and b12443 CUDA build with
one 24,576-token slot and the same projector. Each mode scored 26 held-out
next actions using full answers. Turn time includes prefill and generation.

| Version | Reasoning Off | Median turn | Reasoning On | Median turn |
| --- | ---: | ---: | ---: | ---: |
| Stock Gemma, UD-IQ3_XXS | 5/26 | 1.16 s | 10/26 | 3.23 s |
| 300 steps, standard Q4_K_M | 6/26 | 0.99 s | 6/26 | 1.58 s |
| 300 steps, expert down weights in Q4_0 | 7/26 | 0.99 s | 6/26 | 2.08 s |

All 156 requests completed. The compressed exports had shorter medians and
missed four more cases than stock with reasoning On. On the Ryzen laptop, the
standard export passed exact recall after 18,490 estimated tokens of history.
After the repeated-call guard was added, the table-recovery scenario passed
in 295.3 seconds on CPU. The repository diagnosis initially timed out
trying nonexistent paths. A verified-directory recovery step brought it to
129.9 seconds on a separate run, with the correct file still unidentified.
[Laptop measurements](../scripts/mind/NATIVE_CPU.md#long-chat-on-the-desktop)

With immediate post-write inspection, the laptop saved desktop and phone
browser evidence for both the dashboard and the 3D hatch. The dashboard turn
finished in 274 seconds. The independent interaction check rejected its
initial status element. The hatch rendered after 285 seconds and the turn
reached the 300-second limit before acceptance could check its controls.
[Native acceptance checks](MIND-FINALIST-ACCEPTANCE.md)

The [interactive training generator](../scripts/mind/interactive_artifact_shard.py)
produces 16 separate synthetic cases and eight input screenshots. Chrome
verifies exact initial text, keyboard and pointer activation, focus, and
360-pixel layout. Eight cases also check that the 3D component moves around
the correct axis while its base stays still. These cases are separate from the
held-out dashboard and hatch acceptance tests.

### Repository task adapter

A separate 88-step Colab L4 run tuned a rank-16 adapter from the merged
300-step checkpoint. It covered repository discovery, scoped fixes, and command
success, failure and denial. It used 24 verified [repository traces](../scripts/mind/repo_discovery_shard.py),
two original episodes per family sampled at seed `6015`, and the 20 reasoning
episodes from the supplement. Append the reasoning episodes and repository traces
in that order to the original sample. The 72-row training input has SHA-256
`175231701ada72a87f98a091ad899b8cc3dc559209ff1fd33e76d90a05af355f`.

```bash
python3 -m scripts.mind.fit_gemma4 mixed-merged corpus/repo-mixed-train.jsonl repo-adapter \
  --mode l4-lora --steps 88 --per-family 96 --max-tokens 3072 --seed 6017 --lr 2e-5
python3 llama.cpp/convert_lora_to_gguf.py --base mixed-merged --outtype f16 \
  --outfile kiwi-repo-lora-f16.gguf repo-adapter
```

The GGUF adapter has SHA-256
`259d93f0e610c3526147afee3182ce163674a63b6dacab77e2dcdf4bc12b5729`.
The 300 steps are in the trunk and the additional 88 are in the optional adapter.

The verified GGUF loaded on the pinned Colab L4 runtime at scale zero. On
26 held-out next actions per arm, setting scale `0.8` recovered one of two
repository cases with reasoning Off. The overall score remained 6/26 and
median time went from 0.94 to 0.95 seconds. With reasoning On the repository
score stayed 1/2, overall score went from 7/26 to 4/26, and median time went
from 1.61 to 2.08 seconds. These arms applied the requested scale to every
probe. Automatic routing still needs whole-repository validation on the laptop.
[Published adapter and classifier artifacts](https://huggingface.co/yunusemrejr/kiwimind/tree/392fb41cd581b08520c65f047d28e269c28d0e8e/experiments)

### Visual and 3D task adapter

A second rank-16 adapter trained for 96 steps on a Colab L4 from the merged
300-step checkpoint. The fit used two original episodes per family at seed
`6113`, the 20 reasoning episodes, and the 16 Chrome-checked interactive
episodes. Place the image files beside the resulting JSONL under `images/`.
The 64-row mixture has SHA-256
`23e8de5032b87bd5537a8c47d6cf3a363cb61d92d61d32001de83a0b6f8ee20c`.

```bash
python3 -m scripts.mind.fit_gemma4 mixed-merged corpus/visual-mixed-train.jsonl visual-adapter \
  --mode l4-lora --steps 96 --per-family 96 --max-tokens 3072 --seed 6119 --lr 2e-5
python3 llama.cpp/convert_lora_to_gguf.py --base mixed-merged --outtype f16 \
  --outfile kiwi-visual-lora-f16.gguf visual-adapter
```

The export trained on dashboard startup and repair, exact live status and
keyboard focus, mobile rendering, and fixed-base 3D mechanisms. It has SHA-256
`f15389b8f7c39e98c53d9652f882b2e6251303a0e635ef1505629aa67957a993`.
The repository and visual adapters attach independently to the 300-step trunk.

The Colab L4 replay used the same 26 held-out next actions per arm with 24,576
tokens of context. Neither artifact case passed in any arm. The single vision
answer passed in all four. With reasoning Off, one repository observation
improved and one native-code observation regressed. With reasoning On, one
architecture observation regressed. The visual adapter remains experimental.

| Reasoning | Passes at scale 0 → 0.8 | Median response at scale 0 → 0.8 |
| --- | --- | --- |
| Off | 6/26 → 6/26 | 0.94 s → 1.02 s |
| On | 7/26 → 6/26 | 1.62 s → 2.03 s |

[Visual adapter, training receipt, and paired replay](https://huggingface.co/yunusemrejr/kiwimind/tree/b2791d427512803e58ff9c07e186865d1bde41ee/experiments/visual-lora)

### Dense Gemma 4 E4B experiment

The E4B experiment retains the 42-layer decoder, 2,560-wide hidden stream,
per-layer embeddings, shared attention keys and values, tokenizer and
matching vision projector. The Colab L4 calibration gathered feed-forward
activations from eight text, eight tool-use, and eight image episodes.
Within each layer, the retained channels determine the gate and up rows
and the down columns. This changes the width from
10,240 to 8,192 and removes 660,602,880 weights from the dense layers.

Colab completed 112 assistant-only rank-8 training steps on the compact
checkpoint, with 18 text, 62 tool, and 32 image targets. These accounted for
6,931 supervised tokens. The training receipt is tied to the 582-episode
mixture `9d7a45b1e2a4e860375f50ef1b825075fc9e54c31fb6221816ccd23a9f64d117`.
The four commands below run with Transformers `5.23.0` and PEFT `0.21.3`
on Colab. The source directory is the pinned E4B snapshot directory under
`snapshots/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`. Use an environment
without the incompatible optional torchao `0.10.0` package.

```bash
python3 -m scripts.mind.recover_e4b calibrate "$e4b_source" "$e4b_train" calibration
python3 -m scripts.mind.compact_e4b "$e4b_source" compact-e4b \
  --base-revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c \
  --activation-scores calibration/scores.npy
python3 -m scripts.mind.recover_e4b fit compact-e4b "$e4b_train" e4b-adapter --steps 112
python3 -m scripts.mind.recover_e4b merge compact-e4b e4b-adapter merged-e4b
```

[Pinned E4B base checkpoint](https://huggingface.co/google/gemma-4-E4B-it/tree/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c)

The exported Q4_K_M trunk is 4,902,294,208 bytes with SHA-256
`e8d713c5833ddbfef0d28abc3185c688867eb46655299dff422c55d50c16bb9b`.
The matching F16 projector brings the pack to 5.89 GB. **How it works**
reports the checked width, layer count and training steps from this pack.

The matched Colab replay used the same 26 held-out next actions, pinned
b12443 CUDA, one 24,576-token slot and the same projector in all arms.

| Reasoning | Stock → compact passes | Stock → compact median response |
| --- | --- | --- |
| Off | 5/26 → 7/26 | 0.86 s → 0.68 s |
| On | 6/26 → 2/26 | 3.82 s → 2.01 s |

The Off arm gained one browser, one native-code and one architecture case
and lost one research case. These are GPU next-action checks. The compact
pack remains experimental while the laptop workflow checks continue.
[Verified E4B pack](https://huggingface.co/yunusemrejr/kiwimind/tree/0f67f744e973c3992c17dc13106becbcb49e16b3/candidate) · [Model card and raw comparison](https://huggingface.co/yunusemrejr/kiwimind/blob/3cda458fa8720d0f9d9e50c41dbe3bde3c259aea/README.md)

Colab also fitted the workflow classifier using the reviewed authored
workflow examples. On the separate 26-request classifier test it scored
21 exact labels, compared with seven for the rules. The candidate remains
disabled pending an app-level routing benefit. The recovery fit abstained
because all 16 eligible calibration cases already passed with the E4B
export. The other eight cases were abstention controls.
[Workflow classifier and recovery receipt](https://huggingface.co/yunusemrejr/kiwimind/tree/346732571e2df99adb664a21cc71d91d9c317f51/experiments/e4b-classifiers)

The same activated-channel checkpoint before the 112-step fit scored 7/26
with reasoning Off and 8/26 with reasoning On in the matched Colab replay.
This comparison keeps the same expert-free 8,192-wide dense layers and
isolates the effect of fitting and merging. A separate 29.6 MB F16 LoRA
export binds to that exact compact trunk for request-level evaluation.
The matched adapter replay kept one server loaded and selected the scale
per request. Scale 0 reproduced the compaction-only scores. The adapter
remains disabled because scale 0.8 reduced accuracy in both modes.

| Reasoning | Scale 0 → 0.8 passes | Scale 0 → 0.8 median response |
| --- | --- | --- |
| Off | 7/26 → 3/26 | 0.74 s → 0.73 s |
| On | 8/26 → 4/26 | 3.37 s → 2.61 s |

The compact trunk and shared image projector total 5.93 GB. The separate
adapter is included for replay with its scale set to zero in the pack.
[Verified compaction-only pack](https://huggingface.co/yunusemrejr/kiwimind/tree/91f41a13a7519bfbd631e1937a70be83de411443/variants/e4b-activation-compacted)

The experimental CPU pack can include the official E4B Q8_0 draft head.
Its `mtp.file` points to the checked GGUF in `files`, and
`mtp.trunk_sha256` binds the head to the installed trunk. Together the trunk,
projector and head occupy 6.02 GB. The managed b12443 CPU server can draft
up to two tokens per step in one slot and verify them with the trunk.
An adapter that can activate uses ordinary decoding. A draft startup
failure retries with the same trunk and projector.
[Verified CPU draft pack](https://huggingface.co/yunusemrejr/kiwimind/tree/bfbcc8ddc1d4902ddd71ee4a6d14b373c76ca31c/variants/e4b-cpu-mtp) · [Measured draft decoding](../scripts/mind/NATIVE_CPU.md#compact-e4b-multi-token-prediction-on-cpu)

## Verify before changing the local launcher

- Open **How it works → Kiwi Mind** and check the model, quantization and expert counts
  against the candidate manifest.
- Send a plain chat, a request to read an actual workspace file and a request to describe
  an attached image. Check the resulting tool call and output in the chat.
- Exercise a durable memory in a fresh public chat and verify that a Private chat leaves the
  memory list alone.
- Compare task success, first-token time and completed-turn time on the same device and
  runtime, with the same task set and context settings. Record cold and warm runs separately.

The app's download target is `candidate`, with files under `candidate/` in the configured
pack repository. In candidate mode the default `primary` download selects the same pack.
[API contract](API.md#kiwi-mind)
