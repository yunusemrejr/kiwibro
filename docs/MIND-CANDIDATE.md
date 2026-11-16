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

For a compacted Gemma pack, pass `--architecture path/to/architecture.json` from
[the expert export](../scripts/mind/prune_gemma4.py). For a fitted checkpoint, also
supply `--training path/to/training.json` from the completed Colab run. Staging
checks the expert report against the actual GGUF. How it works reads the checked
expert counts, intermediate expert width when compacted, file sizes and context
from this pack.

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
Retrain the routing and recovery helpers on the new model's traces with the
[Colab calibration workflow](../scripts/mind/STEERING_CALIBRATION.md).

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
