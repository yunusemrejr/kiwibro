"""pytest adapter for the standalone e2e suite.

``python3 -m pytest tests/e2e -q`` runs the whole fake-LM end-to-end suite.
It needs no LM Studio and no network: the app talks to tests/e2e/fake_lm.py.
Set ``KIWIBRO_E2E=0`` to skip (e.g. on a machine where binding loopback ports is
forbidden).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

E2E_DIR = Path(__file__).resolve().parent


@pytest.mark.skipif(
    os.environ.get("KIWIBRO_E2E", "1") == "0",
    reason="KIWIBRO_E2E=0 disables the end-to-end suite",
)
def test_e2e_suite(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    process = subprocess.run(
        [sys.executable, str(E2E_DIR / "run_e2e.py"), "--quiet", "--out", str(evidence)],
        cwd=str(E2E_DIR.parents[1]),
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = (process.stdout or "") + (process.stderr or "")
    print(output[-30000:])
    assert process.returncode == 0, "end-to-end suite failed; see output above"
