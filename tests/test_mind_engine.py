"""Runs tests/mind_engine.mjs (the Kiwi Mind engine, tokenizer and host on a tiny random model) under pytest.

The Node script does the work and prints one line per check; this wrapper skips when Node.js is missing and
otherwise needs exit code 0. It does not import the kiwibro package.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "mind_engine.mjs"


def _node() -> str | None:
    found = shutil.which("node")
    if found:
        return found
    for candidate in sorted(Path.home().glob(".nvm/versions/node/*/bin/node"), reverse=True):
        return str(candidate)
    return None


def test_mind_engine_checks_pass() -> None:
    node = _node()
    if node is None:
        pytest.skip("node is not installed")
    result = subprocess.run([node, str(SCRIPT)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
