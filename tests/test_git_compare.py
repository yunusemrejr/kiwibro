"""git_info(action="compare"): the local copy against its remote, on real git in temp repos (offline)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiwibro import codebase  # noqa: E402
from kiwibro.sandbox import SandboxError, Workspace  # noqa: E402

IDENTITY = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *IDENTITY, *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=60
    )
    return done.stdout.strip()


def _commit(repo: Path, name: str, text: str, message: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)


def _run_compare(path: Path, remote: str | None = None) -> dict:
    return codebase.git_info(Workspace(path), "compare", path=".", remote=remote)


@pytest.fixture()
def repos(tmp_path: Path) -> tuple[Path, Path]:
    """Repository A with two commits, and B cloned from A (B's origin is A)."""
    origin = tmp_path / "a"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _commit(origin, "one.txt", "1\n", "first")
    _commit(origin, "two.txt", "2\n", "second")
    clone = tmp_path / "b"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True, timeout=60)
    return origin, clone


def test_same_commit(repos):
    _, clone = repos
    result = _run_compare(clone)
    lines = result["output"].splitlines()
    assert lines[0].startswith("Same commit: yes")
    assert result["same"] is True
    assert result["local"] == _git(clone, "rev-parse", "HEAD")
    assert result["remote_sha"] == result["local"]
    assert result["branch"] == "main"
    assert "Uncommitted changes in the local copy: none" in lines
    assert any(line.startswith("Local: ") for line in lines)
    assert any(line.startswith("Remote branches: ") and "main" in line for line in lines)


def test_local_commit_ahead(repos):
    _, clone = repos
    _commit(clone, "three.txt", "3\n", "local work")
    result = _run_compare(clone)
    lines = result["output"].splitlines()
    assert lines[0].startswith("Same commit: no")
    assert "The local copy is 1 commit ahead." in lines
    assert result["same"] is False


def test_remote_commit_not_fetched(repos):
    origin, clone = repos
    _commit(origin, "three.txt", "3\n", "remote work")
    result = _run_compare(clone)
    assert result["output"].splitlines()[0].startswith("Same commit: no")
    assert "not in the local copy" in result["output"]
    assert result["same"] is False


def test_uncommitted_file(repos):
    _, clone = repos
    (clone / "scratch.txt").write_text("x\n", encoding="utf-8")
    result = _run_compare(clone)
    assert "Uncommitted changes in the local copy: 1 file" in result["output"].splitlines()
    assert result["same"] is True


def test_local_branch_named_on_remote_is_compared(repos):
    origin, clone = repos
    _git(origin, "checkout", "-qb", "dev")
    _commit(origin, "dev.txt", "d\n", "dev work")
    _git(origin, "checkout", "-q", "main")
    _git(clone, "fetch", "-q", "origin")
    _git(clone, "checkout", "-q", "-b", "dev", "origin/dev")
    result = _run_compare(clone)
    assert result["output"].splitlines()[0].startswith("Same commit: yes")
    assert result["remote_branch"] == "dev"


def test_not_a_repository(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    result = _run_compare(plain)
    assert result["output"].startswith("Could not compare:")
    assert result["same"] is None


def test_remote_name_that_does_not_exist(repos):
    _, clone = repos
    result = _run_compare(clone, "nosuch")
    assert result["output"].startswith("Could not compare:")
    assert result["same"] is None


def test_credentials_in_address_are_refused_without_echo(repos):
    _, clone = repos
    with pytest.raises(SandboxError) as info:
        _run_compare(clone, "https://user:secret@github.com/o/r")
    assert "secret" not in str(info.value)


@pytest.mark.parametrize(
    ("text", "address", "branch"),
    [
        ("origin", "origin", None),
        ("https://github.com/owner/repo", "https://github.com/owner/repo", None),
        ("https://git.example.org:8443/team/repo.git", "https://git.example.org:8443/team/repo.git", None),
        ("owner/repo", "https://github.com/owner/repo", None),
        ("https://github.com/owner/repo/tree/main/src/app.py", "https://github.com/owner/repo", "main"),
        ("https://github.com/owner/repo/commit/abc123", "https://github.com/owner/repo", None),
    ],
)
def test_accepted_remote_addresses(text, address, branch):
    assert codebase._remote_address(text) == (address, branch)


@pytest.mark.parametrize(
    "text",
    [
        "--upload-pack=touch /tmp/x",
        "ext::sh -c id",
        "file:///etc",
        "/etc/passwd",
        "a b",
        "-o",
        "https://github.com/owner/repo; rm -rf x",
        "",
        "https://user:secret@github.com/o/r",
        "ssh://git@example.org:2222/team/repo.git",  # ssh only through the checkout's own remote, by name
        "git@github.com:owner/repo.git",
    ],
)
def test_refused_remote_addresses(text):
    with pytest.raises(SandboxError) as info:
        codebase._remote_address(text)
    assert "secret" not in str(info.value)
