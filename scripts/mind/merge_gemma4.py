#!/usr/bin/env python3
"""Merge one Colab L4 adapter into its compact BF16 checkpoint on the CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from scripts.mind.prune_gemma4 import BASE_REVISION


def merge(checkpoint: Path, adapter: Path, output: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    architecture = (checkpoint / "architecture.json").read_bytes()
    if architecture != (adapter / "architecture.json").read_bytes():
        raise ValueError("The adapter and checkpoint describe different expert banks.")
    report = json.loads((adapter / "training.json").read_text())
    if report.get("base_revision") != BASE_REVISION or report.get("mode") != "l4-lora":
        raise ValueError("Expected a tracked L4 adapter for the pinned Gemma 4 revision.")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".merge-gemma-") as folder:
        stage = Path(folder) / "merged"
        model.save_pretrained(stage, max_shard_size="4GB")
        AutoProcessor.from_pretrained(checkpoint).save_pretrained(stage)
        (stage / "architecture.json").write_bytes(architecture)
        shutil.copyfile(adapter / "training.json", stage / "training.json")
        stage.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    merge(args.checkpoint, args.adapter, args.output)


if __name__ == "__main__":
    main()
