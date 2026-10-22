"""Self-update: a running server moves onto new code by itself, only when idle."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kiwibro import selfupdate

REPO = Path(__file__).resolve().parent.parent


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "kiwibro").mkdir(parents=True)
    (root / "static" / "js").mkdir(parents=True)
    (root / "templates").mkdir()
    (root / "tests").mkdir()
    (root / "app.py").write_text("print('app')\n")
    (root / "kiwibro" / "core.py").write_text("X = 1\n")
    (root / "static" / "js" / "app.js").write_text("let a = 1;\n")
    (root / "templates" / "index.html").write_text("<html></html>\n")
    (root / "tests" / "test_x.py").write_text("def test(): pass\n")
    return root


def test_fingerprint_follows_served_source_only(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    base = selfupdate.fingerprint(root)
    assert base == selfupdate.fingerprint(root)

    (root / "tests" / "test_x.py").write_text("def test(): assert 1\n")
    (root / "kiwibro" / "__pycache__").mkdir()
    (root / "kiwibro" / "__pycache__" / "core.cpython-314.pyc").write_bytes(b"\0")
    os.utime(root / "app.py")  # touched, same content
    assert selfupdate.fingerprint(root) == base

    (root / "static" / "js" / "app.js").write_text("let a = 2;\n")
    changed = selfupdate.fingerprint(root)
    assert changed != base
    (root / "static" / "js" / "app.js").write_text("let a = 1;\n")
    assert selfupdate.fingerprint(root) == base


def test_fingerprint_script_matches_module_without_dependencies() -> None:
    """The launcher runs the file with `python3 -I`, no package import, no flask."""
    out = subprocess.run(
        [sys.executable, "-I", str(REPO / "kiwibro" / "selfupdate.py"), "fingerprint", str(REPO)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == selfupdate.fingerprint(REPO)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _updater(root: Path, *, busy: list[bool], preflight_ok: list[bool], execs: list[list[str]]):
    clock = _Clock()
    updater = selfupdate.Updater(
        root,
        lambda: busy[0],
        settle_s=10.0,
        exec_fn=execs.append,
        preflight_fn=lambda _root: (preflight_ok[0], "" if preflight_ok[0] else "SyntaxError"),
        clock=clock,
    )
    return updater, clock


def test_restarts_only_after_quiet_period_and_when_idle(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    busy, ok, execs = [True], [True], []
    updater, clock = _updater(root, busy=busy, preflight_ok=ok, execs=execs)
    start = updater.code_version

    assert updater.check() is False and not updater.update_pending
    (root / "kiwibro" / "core.py").write_text("X = 2\n")
    assert updater.check() is False  # change noticed
    assert updater.update_pending
    clock.now += 5
    assert updater.check() is False  # still settling
    clock.now += 10
    assert updater.check() is False and execs == []  # a reply is running
    busy[0] = False
    assert updater.check() is True
    assert len(execs) == 1 and execs[0][0] == sys.executable
    assert updater.code_version == start  # the new process reports the new version itself


def test_code_that_does_not_import_blocks_until_the_next_edit(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    busy, ok, execs = [False], [False], []
    updater, clock = _updater(root, busy=busy, preflight_ok=ok, execs=execs)

    (root / "kiwibro" / "core.py").write_text("X = (\n")
    updater.check()
    clock.now += 20
    assert updater.check() is False and execs == []
    assert updater.status()["update_blocked"] is True
    clock.now += 20
    assert updater.check() is False  # not retried for the same broken tree

    ok[0] = True
    (root / "kiwibro" / "core.py").write_text("X = 3\n")
    updater.check()
    assert updater.status()["update_blocked"] is False
    clock.now += 20
    assert updater.check() is True and len(execs) == 1


def test_reverted_change_needs_no_restart(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    busy, ok, execs = [False], [True], []
    updater, clock = _updater(root, busy=busy, preflight_ok=ok, execs=execs)
    (root / "app.py").write_text("print('edited')\n")
    updater.check()
    (root / "app.py").write_text("print('app')\n")
    updater.check()
    clock.now += 60
    assert updater.check() is False and execs == []
    assert not updater.update_pending


def test_preflight_imports_the_real_tree() -> None:
    ok, error = selfupdate.preflight(REPO)
    assert ok, error


def test_preflight_reports_a_broken_tree(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    (root / "kiwibro").mkdir(parents=True)
    (root / "kiwibro" / "__init__.py").write_text("def broken(:\n")
    ok, error = selfupdate.preflight(root)
    assert not ok and "SyntaxError" in error


def test_version_endpoint_reports_running_code(env) -> None:
    client = env.app.test_client()
    data = client.get("/api/version").get_json()
    assert data["ok"] is True and data["pid"] == os.getpid()
    assert data["busy"] is False and data["code_version"] is None  # no updater in tests

    updater = selfupdate.Updater(REPO, lambda: False)
    env.app.extensions["kiwibro_updater"] = updater
    data = client.get("/api/version").get_json()
    assert data["code_version"] == updater.code_version
    assert data["update_pending"] is False
    health = client.get("/health").get_json()
    assert health["code_version"] == updater.code_version


def test_ui_and_static_files_always_revalidate(env) -> None:
    client = env.app.test_client()
    assert "no-cache" in client.get("/").headers.get("Cache-Control", "")
    assert "no-cache" in client.get("/static/js/app.js").headers.get("Cache-Control", "")
