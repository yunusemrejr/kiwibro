"""A tiny random Kiwi Mind pack for tests: the same files and manifest the training export writes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ["chat", "web", "code", "terminal", "computer", "tools", "write"]
HEAD_NAMES = [*SKILLS, "hard"]
FIRST_LAYER = 2


def load_formats() -> Any:
    path = ROOT / "scripts" / "mind" / "formats.py"
    spec = importlib.util.spec_from_file_location("kiwimind_formats_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def node_path() -> str | None:
    from kiwibro.mind.host import find_node

    return find_node()[0]


def make_pack(data_dir: Path, *, ctx: int = 2048, version: str = "test-1") -> Path:
    """Write ``<data_dir>/mind`` and return it."""
    formats = load_formats()
    config = formats.TINY
    pack = Path(data_dir) / "mind"
    shutil.rmtree(pack, ignore_errors=True)
    (pack / "skills").mkdir(parents=True)
    formats.write_random_trunk(pack / "kiwimind-trunk.fkm", config, mode="q4", emb="q8", seed=3, max_ctx=ctx)
    for number, name in enumerate(SKILLS):
        entries = formats.random_skill_entries(config, FIRST_LAYER, 4, seed=40 + number)
        formats.write_skill(pack / "skills" / f"{name}.kms", name, FIRST_LAYER, config["n_layers"], entries)
    formats.write_head(pack / "head.kmh", *formats.random_head(config, HEAD_NAMES, seed=9), HEAD_NAMES)
    (pack / "tokenizer.json").write_text(json.dumps(formats.tiny_tokenizer(config["vocab"])))
    files = {}
    for path in sorted(pack.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(pack))] = {
                "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    manifest = {"name": "kiwimind", "version": version, "base": "Qwen/Qwen3-0.6B", "split": FIRST_LAYER, "rank": 4,
                "skills": SKILLS, "head": HEAD_NAMES, "ctx": ctx, "files": files}
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return pack
