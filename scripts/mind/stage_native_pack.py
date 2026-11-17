#!/usr/bin/env python3
"""Stage a checked Gemma 4 native pack without duplicating its GGUF files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from kiwibro.adapters import ALWAYS, TASKS
from kiwibro.gguf import read_metadata, summarize
from kiwibro.mind.native_generic import registered_lora
from kiwibro.mind.pack import PackError, TRUNK3
from scripts.mind.prune_gemma4 import BASE, BASE_REVISION

EXPERT_FIELDS = ("experts_before", "experts_after", "top_k_before", "top_k_after")
ADAPTER_FILE = "task-adapter.gguf"


def _file(source: Path, expected: str | None) -> dict[str, int | str]:
    before = source.stat()
    with source.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{source.name} changed while it was being checked.")
    if expected is not None and expected.lower() != digest:
        raise ValueError(f"Unexpected SHA-256 for {source.name}.")
    return {"bytes": after.st_size, "sha256": digest}


def stage(
    model: Path, projector: Path, output: Path, *, version: str, context: int,
    architecture: dict | None = None, training: dict | None = None,
    model_digest: str | None = None, projector_digest: str | None = None,
    adapter: Path | None = None, adapter_digest: str | None = None,
    adapter_trunk_digest: str | None = None, adapter_tasks: tuple[str, ...] = (), adapter_scale: float | None = None,
) -> dict:
    if adapter is None and (adapter_digest is not None or adapter_trunk_digest is not None
                            or adapter_tasks or adapter_scale is not None):
        raise ValueError("LoRA registration requires an explicit adapter file.")
    if adapter is not None and (not adapter_trunk_digest or not adapter_tasks):
        raise ValueError("LoRA registration requires its training/export trunk SHA-256 and task labels.")
    model = model.resolve(strict=True)
    projector = projector.resolve(strict=True)
    meta = read_metadata(model)
    hints = summarize(meta)
    vision_meta = read_metadata(projector)
    if hints["arch"] != "gemma4" or not hints["tools"] or not hints["quantization"]:
        raise ValueError("The native pack needs Gemma 4 GGUF metadata and a tool-capable chat template.")
    if not 512 <= context <= (hints["context_length"] or 0):
        raise ValueError("Context is outside the model's declared window.")
    if not summarize(vision_meta)["projector"] or (
        type(meta.get("gemma4.embedding_length")) is not int
        or meta["gemma4.embedding_length"] != vision_meta.get("clip.vision.projection_dim")
    ):
        raise ValueError("Projector output dimensions do not match the model's embedding width.")

    experts, active = meta.get("gemma4.expert_count"), meta.get("gemma4.expert_used_count")
    if type(experts) is not int or type(active) is not int or not 1 <= active <= experts:
        raise ValueError("Invalid or missing expert metadata in the GGUF.")
    bank = dict(zip(EXPERT_FIELDS, (experts, experts, active, active), strict=True))
    if architecture is not None:
        if architecture.get("base_revision") != BASE_REVISION or architecture.get("base_model") != BASE:
            raise ValueError("Expert report does not name the pinned base checkpoint.")
        bank = {key: architecture.get(key) for key in EXPERT_FIELDS}
        if (
            any(type(value) is not int or value < 1 for value in bank.values())
            or bank["experts_after"] != experts or bank["top_k_after"] != active
            or not active <= bank["top_k_before"] <= bank["experts_before"]
            or experts > bank["experts_before"]
        ):
            raise ValueError("Expert report disagrees with the exported GGUF.")
        if (channels := architecture.get("channel_compaction")) is not None:
            width = meta.get("gemma4.expert_feed_forward_length")
            if not isinstance(channels, dict):
                raise ValueError("Invalid expert channel report.")
            before, after = channels.get("width_before"), channels.get("width_after")
            if (
                type(width) is not int or type(before) is not int or type(after) is not int
                or not 1 <= after < before or after != width or channels.get("experts") != experts
                or channels.get("top_k_experts") != active
            ):
                raise ValueError("Expert channel report disagrees with the exported GGUF.")
            bank.update(width_before=before, width_after=after)
    if training is not None:
        steps = training.get("steps")
        total_steps = training.get("total_steps", steps)
        if (
            architecture is None or training.get("base_revision") != BASE_REVISION
            or type(steps) is not int or type(total_steps) is not int
            or not 1 <= steps <= total_steps
        ):
            raise ValueError("Training metadata is inconsistent with the compact checkpoint.")

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    files = {TRUNK3: _file(model, model_digest), "vision.gguf": _file(projector, projector_digest)}
    manifest = {
        "name": "kiwimind", "version": version, "engine": "llama", "format": "native-gguf",
        "base": BASE, "base_revision": BASE_REVISION, "arch": hints["arch"], "ctx": context,
        "quant": hints["quantization"], "model": TRUNK3, "vision": {"file": "vision.gguf"},
        "capabilities": {"tools": True, "thinking": hints["thinking"]}, "expert_bank": bank, "files": files,
    }
    if adapter is not None:
        if adapter_trunk_digest != files[TRUNK3]["sha256"]:
            raise ValueError("The LoRA binding does not match the trunk's exact SHA-256.")
        adapter = adapter.resolve(strict=True)
        files[ADAPTER_FILE] = _file(adapter, adapter_digest)
        try:
            entry = registered_lora(adapter, read_metadata(adapter), arch=hints["arch"], base=BASE,
                                    tasks=list(adapter_tasks), scale=1.0 if adapter_scale is None else adapter_scale)
        except PackError as exc:
            raise ValueError(str(exc)) from exc
        manifest["lora"] = {"file": ADAPTER_FILE, "trunk_sha256": adapter_trunk_digest,
                            "tasks": list(entry.tasks), "scale": entry.scale}
    if training is not None:
        manifest["training"] = {key: training[key] for key in ("mode", "steps", "total_steps", "corpus_sha256")
                                if key in training}

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".native-pack-") as folder:
        candidate = Path(folder) / "pack"
        candidate.mkdir()
        (candidate / TRUNK3).symlink_to(model)
        (candidate / "vision.gguf").symlink_to(projector)
        if adapter is not None:
            (candidate / ADAPTER_FILE).symlink_to(adapter)
        (candidate / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        candidate.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("projector", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--projector-sha256")
    parser.add_argument("--adapter", type=Path, help="Explicit native task LoRA GGUF to stage.")
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--adapter-trunk-sha256", help="Exact serving trunk SHA-256 from adapter provenance.")
    parser.add_argument("--adapter-task", action="append", choices=(*TASKS, ALWAYS), default=[])
    parser.add_argument("--adapter-scale", type=float)
    args = parser.parse_args()
    architecture = json.loads(args.architecture.read_text()) if args.architecture else None
    training = json.loads(args.training.read_text()) if args.training else None
    print(json.dumps(stage(args.model, args.projector, args.output, version=args.version, context=args.context,
                           architecture=architecture, training=training, model_digest=args.model_sha256,
                           projector_digest=args.projector_sha256, adapter=args.adapter,
                           adapter_digest=args.adapter_sha256, adapter_trunk_digest=args.adapter_trunk_sha256,
                           adapter_tasks=tuple(args.adapter_task), adapter_scale=args.adapter_scale),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
