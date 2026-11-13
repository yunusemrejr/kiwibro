"""Tests for kiwibro.safecopy: a private copy, a proof the user approves, and an undo that is rehearsed.

Every "user folder" here is a temporary folder with made-up content.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

from kiwibro import safecopy


def _real_git(real: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["git", "-C", str(real), *args], capture_output=True, text=True, env=env, check=False
    )


def _tracked(session: safecopy.Session) -> list[str]:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    proc = subprocess.run(
        [
            "git",
            f"--git-dir={session.repo}",
            f"--work-tree={session.work}",
            "-c",
            "core.quotePath=false",
            "ls-files",
            "-z",
        ],
        capture_output=True,
        env=env,
        check=True,
    )
    return [part for part in proc.stdout.decode("utf-8", "surrogateescape").split("\0") if part]


def _files(folder: Path) -> set[str]:
    found = set()
    for current, _folders, names in os.walk(folder, followlinks=False):
        for name in names:
            found.add(str(Path(current, name).relative_to(folder)))
    return found


def _digest(folder: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not path.is_symlink():
            hasher.update(path.relative_to(folder).as_posix().encode("utf-8", "surrogateescape"))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


@pytest.fixture
def real(tmp_path: Path) -> Path:
    folder = tmp_path / "real"
    folder.mkdir()
    (folder / "notes.txt").write_text("one\ntwo\n")
    (folder / "sub").mkdir()
    (folder / "sub" / "deep.txt").write_text("deep\n")
    return folder


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


# --- start -----------------------------------------------------------------------------------


def test_start_copies_files_modes_nested_folders_and_times(real: Path, root: Path) -> None:
    script = real / "sub" / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    session = safecopy.start(real, root, "s1")
    assert session.id == "s1"
    assert session.real == real.resolve()
    assert (session.work / "notes.txt").read_text() == "one\ntwo\n"
    assert (session.work / "sub" / "deep.txt").read_text() == "deep\n"
    assert stat.S_IMODE((session.work / "sub" / "run.sh").stat().st_mode) == 0o755
    assert int((session.work / "notes.txt").stat().st_mtime) == int((real / "notes.txt").stat().st_mtime)
    assert safecopy.load(root, "s1").real == real.resolve()


def test_git_and_cache_names_are_not_copied_or_tracked(real: Path, root: Path) -> None:
    (real / ".git").mkdir()
    (real / ".git" / "config").write_text("secret\n")
    (real / "__pycache__").mkdir()
    (real / "__pycache__" / "noise.pyc").write_text("noise\n")
    (real / "sub" / ".cache").mkdir()
    (real / "sub" / ".cache" / "item").write_text("noise\n")
    session = safecopy.start(real, root, "s1")
    assert not (session.work / ".git").exists()
    assert not (session.work / "__pycache__").exists()
    assert not (session.work / "sub" / ".cache").exists()
    tracked = _tracked(session)
    assert "notes.txt" in tracked
    assert not any(".git" in path or "__pycache__" in path or ".cache" in path for path in tracked)


def test_node_modules_is_linked_to_the_real_one_and_never_tracked(real: Path, root: Path) -> None:
    package = real / "node_modules" / "left-pad"
    package.mkdir(parents=True)
    (package / "index.js").write_text("module.exports = 1\n")
    session = safecopy.start(real, root, "s1")
    link = session.work / "node_modules"
    assert link.is_symlink()
    assert (link / "left-pad" / "index.js").read_text() == "module.exports = 1\n"
    assert not any("node_modules" in path for path in _tracked(session))


def test_an_oversized_file_is_left_out_and_reported(real: Path, root: Path) -> None:
    (real / "big.bin").write_bytes(b"x" * 200)
    session = safecopy.start(real, root, "s1", limits=safecopy.Limits(max_file_bytes=100))
    assert not (session.work / "big.bin").exists()
    assert [entry["path"] for entry in session.left_out] == ["big.bin"]
    assert "100" in session.left_out[0]["reason"]
    assert session.left_out == safecopy.load(root, "s1").left_out


def test_a_link_pointing_outside_the_folder_is_not_copied(real: Path, root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    os.symlink(str(outside), real / "escape.txt")
    session = safecopy.start(real, root, "s1")
    assert not (session.work / "escape.txt").exists()
    assert [entry["path"] for entry in session.left_out] == ["escape.txt"]


def test_too_many_files_raises_too_big_with_the_numbers(real: Path, root: Path) -> None:
    for number in range(5):
        (real / f"more{number}.txt").write_text("x\n")
    with pytest.raises(safecopy.SafeCopyError) as caught:
        safecopy.start(real, root, "s1", limits=safecopy.Limits(max_files=3))
    assert caught.value.code == "too_big"
    assert "3" in caught.value.message
    assert not (root / "s1").exists()


def test_too_many_bytes_raises_too_big_with_the_numbers(real: Path, root: Path) -> None:
    (real / "heavy.bin").write_bytes(b"x" * 5000)
    with pytest.raises(safecopy.SafeCopyError) as caught:
        safecopy.start(real, root, "s1", limits=safecopy.Limits(max_bytes=1000))
    assert caught.value.code == "too_big"
    assert "1000" in caught.value.message


def test_the_refused_folders_are_refused(real: Path, root: Path, tmp_path: Path) -> None:
    with pytest.raises(safecopy.SafeCopyError) as caught:
        safecopy.start(Path("/"), tmp_path / "root-for-fs", "s1")
    assert caught.value.code == "refused_folder"
    with pytest.raises(safecopy.SafeCopyError) as caught:
        safecopy.start(Path.home(), tmp_path / "root-for-home", "s1")
    assert caught.value.code == "refused_folder"
    inside = real / "sessions"
    inside.mkdir()
    with pytest.raises(safecopy.SafeCopyError) as caught:
        safecopy.start(real, inside, "s1")
    assert caught.value.code == "refused_folder"
    with pytest.raises(safecopy.SafeCopyError) as caught:
        safecopy.start(real, tmp_path, "s1")
    assert caught.value.code == "refused_folder"
    a_file = tmp_path / "not-a-folder.txt"
    a_file.write_text("x\n")
    with pytest.raises(safecopy.SafeCopyError) as caught:
        safecopy.start(a_file, root, "s1")
    assert caught.value.code == "not_a_folder"
    assert not (root / "s1").exists()


# --- changes ---------------------------------------------------------------------------------


def test_changes_report_kinds_and_line_counts_and_moves(real: Path, root: Path) -> None:
    (real / "move.txt").write_text("move me\n")
    session = safecopy.start(real, root, "s1")
    work = session.work
    (work / "notes.txt").write_text("one\nTWO\nthree\n")
    (work / "fresh.txt").write_text("fresh\n")
    (work / "sub" / "deep.txt").unlink()
    (work / "move.txt").rename(work / "moved.txt")
    changes = {change.path: change for change in session.changes()}
    assert changes["notes.txt"].kind == "modified"
    assert (changes["notes.txt"].added_lines, changes["notes.txt"].removed_lines) == (2, 1)
    assert changes["fresh.txt"].kind == "added"
    assert (changes["fresh.txt"].added_lines, changes["fresh.txt"].removed_lines) == (1, 0)
    assert changes["sub/deep.txt"].kind == "deleted"
    assert (changes["sub/deep.txt"].added_lines, changes["sub/deep.txt"].removed_lines) == (0, 1)
    assert changes["move.txt"].kind == "deleted"
    assert changes["moved.txt"].kind == "added"


def test_a_binary_file_is_flagged_and_shown_as_one_line(real: Path, root: Path) -> None:
    (real / "blob.bin").write_bytes(b"\x00\x01\x02")
    session = safecopy.start(real, root, "s1")
    (session.work / "blob.bin").write_bytes(b"\x00\x01\x03")
    change = {item.path: item for item in session.changes()}["blob.bin"]
    assert change.binary is True
    assert change.kind == "modified"
    proof = session.proof()
    assert "blob.bin: binary file" in proof.diff.splitlines()
    assert not any(line.startswith("@@") for line in proof.diff.splitlines())


def test_a_file_ignored_by_a_gitignore_inside_the_folder_is_still_tracked(real: Path, root: Path) -> None:
    (real / ".gitignore").write_text("*.log\n")
    (real / "app.log").write_text("first\n")
    session = safecopy.start(real, root, "s1")
    assert "app.log" in _tracked(session)
    (session.work / "app.log").write_text("second\n")
    proof = session.proof()
    assert [change.path for change in proof.changes] == ["app.log"]
    assert "+second" in proof.diff


# --- proof -----------------------------------------------------------------------------------


def test_proof_has_a_diff_no_conflicts_and_a_verified_revert(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    (session.work / "notes.txt").write_text("one\ntwo\nthree\n")
    (session.work / "added.txt").write_text("added\n")
    proof = session.proof()
    assert proof.conflicts == []
    assert proof.left_out == []
    assert "+three" in proof.diff
    assert proof.totals["files"] == 2
    assert proof.revert["verified"] is True
    assert proof.revert["files_checked"] == 1
    assert proof.before_commit
    data = proof.to_dict()
    assert data["changes"][0]["path"] == "added.txt"
    assert data["fingerprint"] == proof.fingerprint
    assert data["totals"] == proof.totals
    assert json.loads(json.dumps(data))["revert"]["verified"] is True


def test_removal_commands_quote_spaces_quotes_and_leading_dashes(real: Path, root: Path) -> None:
    names = ["with space.txt", "it's.txt", "-dash.txt"]
    for name in names:
        (real / name).write_text("x\n")
    session = safecopy.start(real, root, "s1")
    for name in names:
        (session.work / name).unlink()
    proof = session.proof()
    assert sorted(entry["path"] for entry in proof.removals) == sorted(names)
    for entry in proof.removals:
        assert shlex.split(entry["command"]) == ["rm", "--", str(real / entry["path"])]
        assert entry["command"].startswith("rm -- ")


def test_the_fingerprint_holds_still_and_moves_with_the_copy_and_the_real_folder(
    real: Path, root: Path
) -> None:
    session = safecopy.start(real, root, "s1")
    (session.work / "notes.txt").write_text("one\ntwo\nthree\n")
    first = session.proof().fingerprint
    assert session.proof().fingerprint == first
    (session.work / "notes.txt").write_text("one\nTWO\n")
    second = session.proof().fingerprint
    assert second != first
    (real / "notes.txt").write_text("user edit\n")
    third = session.proof().fingerprint
    assert third != second


def test_the_diff_is_cut_at_the_line_limits(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    for number in range(12):
        body = "".join(f"line {number} {step}\n" for step in range(500))
        (session.work / f"long{number}.txt").write_text(body)
    proof = session.proof()
    lines = proof.diff.splitlines()
    assert len(lines) <= safecopy.MAX_DIFF_LINES_TOTAL + 1
    assert any("more lines of this file not shown" in line for line in lines)
    assert any("more files not shown" in line for line in lines)


def test_nothing_to_apply_when_nothing_is_added_or_modified(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    with pytest.raises(safecopy.SafeCopyError) as caught:
        session.revert_last()
    assert caught.value.code == "not_found"
    with pytest.raises(safecopy.SafeCopyError) as caught:
        session.apply(session.proof().fingerprint)
    assert caught.value.code == "nothing_to_apply"
    (session.work / "notes.txt").unlink()
    with pytest.raises(safecopy.SafeCopyError) as caught:
        session.apply(session.proof().fingerprint)
    assert caught.value.code == "nothing_to_apply"
    assert (real / "notes.txt").exists()


def test_commit_reports_nothing_to_record_when_the_copy_is_unchanged(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    assert session.commit("again") is None
    (session.work / "notes.txt").write_text("changed\n")
    assert session.commit("edit") is not None
    assert session.commit("edit") is None


def test_an_undo_that_does_not_restore_the_old_bytes_makes_no_proof(
    real: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = safecopy.start(real, root, "s1")
    (session.work / "notes.txt").write_text("changed\n")
    monkeypatch.setattr(safecopy, "_revert_command", lambda *_args: "true")
    with pytest.raises(safecopy.SafeCopyError) as caught:
        session.proof()
    assert caught.value.code == "revert_unproven"
    assert (real / "notes.txt").read_text() == "one\ntwo\n"


# --- apply -----------------------------------------------------------------------------------


def test_apply_with_a_wrong_fingerprint_is_stale_and_writes_nothing(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    (session.work / "notes.txt").write_text("changed\n")
    (session.work / "added.txt").write_text("added\n")
    with pytest.raises(safecopy.SafeCopyError) as caught:
        session.apply("0" * 64)
    assert caught.value.code == "stale"
    assert (real / "notes.txt").read_text() == "one\ntwo\n"
    assert not (real / "added.txt").exists()


def test_apply_writes_added_and_modified_files_and_hands_removals_back(real: Path, root: Path) -> None:
    (real / "move.txt").write_text("move me\n")
    (real / "old.txt").write_text("old\n")
    session = safecopy.start(real, root, "s1")
    work = session.work
    (work / "notes.txt").write_text("one\nTWO\n")
    (work / "fresh.txt").write_text("fresh\n")
    (work / "made" / "deep").mkdir(parents=True)
    (work / "made" / "deep" / "nested.txt").write_text("nested\n")
    (work / "sub" / "deep.txt").unlink()
    (work / "old.txt").unlink()
    (work / "move.txt").rename(work / "moved.txt")
    (work / "fresh.txt").chmod(0o640)
    proof = session.proof()
    applied = session.apply(proof.fingerprint)
    assert applied.written == ["fresh.txt", "made/deep/nested.txt", "moved.txt", "notes.txt"]
    assert (real / "notes.txt").read_text() == "one\nTWO\n"
    assert (real / "fresh.txt").read_text() == "fresh\n"
    assert (real / "made" / "deep" / "nested.txt").read_text() == "nested\n"
    assert stat.S_IMODE((real / "fresh.txt").stat().st_mode) == 0o640
    assert (real / "moved.txt").read_text() == "move me\n"
    assert (real / "old.txt").exists()
    assert (real / "sub" / "deep.txt").exists()
    assert (real / "move.txt").exists()
    assert {entry["path"] for entry in applied.removals} == {"old.txt", "move.txt", "sub/deep.txt"}
    assert applied.applied_commit and applied.before_commit


def test_a_changed_real_file_is_a_conflict_and_blocks_apply(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    (session.work / "notes.txt").write_text("mine\n")
    (session.work / "added.txt").write_text("added\n")
    (real / "notes.txt").write_text("user edit\n")
    proof = session.proof()
    assert proof.conflicts == ["notes.txt"]
    with pytest.raises(safecopy.SafeCopyError) as caught:
        session.apply(proof.fingerprint)
    assert caught.value.code == "conflict"
    assert (real / "notes.txt").read_text() == "user edit\n"
    assert not (real / "added.txt").exists()


def test_an_added_path_that_already_exists_with_other_content_is_a_conflict(
    real: Path, root: Path
) -> None:
    session = safecopy.start(real, root, "s1")
    (session.work / "added.txt").write_text("from the copy\n")
    (real / "added.txt").write_text("from the user\n")
    proof = session.proof()
    assert "added.txt" in proof.conflicts
    with pytest.raises(safecopy.SafeCopyError) as caught:
        session.apply(proof.fingerprint)
    assert caught.value.code == "conflict"
    assert (real / "added.txt").read_text() == "from the user\n"


def test_a_write_that_fails_midway_records_what_was_written_so_undo_works(
    real: Path, root: Path
) -> None:
    (real / "locked").mkdir()
    (real / "locked" / "inner.txt").write_text("inner\n")
    (real / "a-first.txt").write_text("first\n")
    session = safecopy.start(real, root, "s1")
    (session.work / "a-first.txt").write_text("FIRST\n")
    (session.work / "locked" / "inner.txt").write_text("INNER\n")
    (session.work / "locked" / "extra.txt").write_text("extra\n")
    proof = session.proof()
    (real / "locked").chmod(0o500)
    try:
        with pytest.raises(safecopy.SafeCopyError) as caught:
            session.apply(proof.fingerprint)
    finally:
        (real / "locked").chmod(0o700)
    assert caught.value.code == "partial"
    assert "1 of 3" in caught.value.message
    assert (real / "a-first.txt").read_text() == "FIRST\n"
    assert (real / "locked" / "inner.txt").read_text() == "inner\n"
    assert not (real / "locked" / "extra.txt").exists()
    reverted = session.revert_last()
    assert reverted.restored == ["a-first.txt"]
    assert (real / "a-first.txt").read_text() == "first\n"
    assert reverted.kept == []
    assert {change.path for change in session.changes()} == {
        "a-first.txt",
        "locked/extra.txt",
        "locked/inner.txt",
    }
    assert session.proof().conflicts == []


# --- undo ------------------------------------------------------------------------------------


def test_the_returned_revert_command_brings_overwritten_files_back(real: Path, root: Path) -> None:
    (real / "-dash 'file'.txt").write_text("dash\n")
    before = {
        path: (real / path).read_bytes() for path in ["notes.txt", "sub/deep.txt", "-dash 'file'.txt"]
    }
    session = safecopy.start(real, root, "s1")
    work = session.work
    (work / "notes.txt").write_text("one\nTWO\nthree\n")
    (work / "sub" / "deep.txt").write_text("DEEPER\n")
    (work / "-dash 'file'.txt").write_text("DASH\n")
    (work / "fresh.txt").write_text("fresh\n")
    (work / "gone.txt").write_text("soon\n")
    session.commit("prepare")
    (work / "gone.txt").unlink()
    proof = session.proof()
    argv = shlex.split(proof.revert["command"])
    assert argv[:3] == ["git", f"--git-dir={session.repo}", f"--work-tree={session.real}"]
    assert argv[3] == "checkout" and argv[4] == proof.before_commit and argv[5] == "--"
    assert sorted(argv[6:]) == sorted(before)
    applied = session.apply(proof.fingerprint)
    assert (real / "notes.txt").read_text() == "one\nTWO\nthree\n"
    assert (real / "sub" / "deep.txt").read_text() == "DEEPER\n"
    assert (real / "-dash 'file'.txt").read_text() == "DASH\n"
    assert (real / "fresh.txt").read_text() == "fresh\n"
    completed = subprocess.run(proof.revert["command"], shell=True, capture_output=True)
    assert completed.returncode == 0
    for path, data in before.items():
        assert (real / path).read_bytes() == data
    assert (real / "fresh.txt").read_text() == "fresh\n"
    assert [entry["path"] for entry in proof.revert["added"]] == ["fresh.txt"]
    assert shlex.split(proof.revert["added"][0]["command"]) == ["rm", "--", str(real / "fresh.txt")]
    assert applied.revert == proof.revert


def test_revert_last_undoes_the_apply_and_refuses_after_a_real_edit(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    (session.work / "notes.txt").write_text("one\nTWO\nthree\n")
    (session.work / "fresh.txt").write_text("fresh\n")
    applied = session.apply(session.proof().fingerprint)
    assert applied.written == ["fresh.txt", "notes.txt"]
    (real / "notes.txt").write_text("user edit\n")
    with pytest.raises(safecopy.SafeCopyError) as caught:
        session.revert_last()
    assert caught.value.code == "conflict"
    assert (real / "notes.txt").read_text() == "user edit\n"
    (real / "notes.txt").write_text("one\nTWO\nthree\n")
    reverted = session.revert_last()
    assert reverted.restored == ["notes.txt"]
    assert (real / "notes.txt").read_text() == "one\ntwo\n"
    assert (real / "fresh.txt").read_text() == "fresh\n"
    assert [entry["path"] for entry in reverted.kept] == ["fresh.txt"]
    assert shlex.split(reverted.kept[0]["command"]) == ["rm", "--", str(real / "fresh.txt")]
    assert reverted.before_commit == applied.before_commit
    assert {change.path: change.kind for change in session.changes()} == {"notes.txt": "modified"}


# --- links -----------------------------------------------------------------------------------


def test_a_copy_symlink_pointing_outside_is_never_applied(real: Path, root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    session = safecopy.start(real, root, "s1")
    os.symlink(str(outside), session.work / "escape.txt")
    (session.work / "notes.txt").write_text("one\ntwo\nthree\n")
    proof = session.proof()
    assert [entry["path"] for entry in proof.left_out] == ["escape.txt"]
    applied = session.apply(proof.fingerprint)
    assert applied.written == ["notes.txt"]
    assert not (real / "escape.txt").exists()
    assert [entry["path"] for entry in applied.left_out] == ["escape.txt"]


def test_a_real_symlink_is_never_written_through(real: Path, root: Path) -> None:
    (real / "data.txt").write_text("data\n")
    os.symlink("data.txt", real / "link.txt")
    session = safecopy.start(real, root, "s1")
    assert (session.work / "link.txt").is_symlink()
    (session.work / "link.txt").unlink()
    (session.work / "link.txt").write_text("written into the copy\n")
    (session.work / "data.txt").write_text("data\nmore\n")
    proof = session.proof()
    assert [entry["path"] for entry in proof.left_out] == ["link.txt"]
    applied = session.apply(proof.fingerprint)
    assert (real / "link.txt").is_symlink()
    assert os.readlink(real / "link.txt") == "data.txt"
    assert (real / "data.txt").read_text() == "data\nmore\n"
    assert "link.txt" not in applied.written


def test_a_symlinked_folder_in_real_is_never_written_through(
    real: Path, root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(str(outside), real / "linked")
    session = safecopy.start(real, root, "s1")
    assert [entry["path"] for entry in session.left_out] == ["linked"]
    (session.work / "linked").mkdir()
    (session.work / "linked" / "new.txt").write_text("new\n")
    (session.work / "notes.txt").write_text("one\ntwo\nthree\n")
    proof = session.proof()
    applied = session.apply(proof.fingerprint)
    assert not (outside / "new.txt").exists()
    assert [entry["path"] for entry in applied.left_out] == ["linked/new.txt"]
    assert applied.written == ["notes.txt"]


def test_a_file_name_with_a_newline_goes_through_the_whole_cycle(real: Path, root: Path) -> None:
    name = "weird\nname.txt"
    (real / name).write_text("first\n")
    session = safecopy.start(real, root, "s1")
    assert (session.work / name).read_text() == "first\n"
    (session.work / name).write_text("second\n")
    proof = session.proof()
    assert [change.path for change in proof.changes] == [name]
    session.apply(proof.fingerprint)
    assert (real / name).read_text() == "second\n"
    (session.work / name).unlink()
    proof = session.proof()
    assert shlex.split(proof.removals[0]["command"]) == ["rm", "--", str(real / name)]


# --- the real folder as a git repository -----------------------------------------------------


def test_a_real_git_repository_keeps_its_git_folder_and_shows_the_change(real: Path, root: Path) -> None:
    _real_git(real, "init", "--quiet")
    _real_git(real, "add", "-A")
    _real_git(real, "-c", "user.name=User", "-c", "user.email=user@localhost", "commit", "--quiet", "-m", "start")
    session = safecopy.start(real, root, "s1")
    (session.work / "notes.txt").write_text("one\nTWO\n")
    (session.work / "fresh.txt").write_text("fresh\n")
    before = _digest(real / ".git")
    proof = session.proof()
    assert proof.revert["real_is_clean_git"] is True
    assert proof.revert["git_restore"].startswith(f"git -C {shlex.quote(str(session.real))} restore -- ")
    session.apply(proof.fingerprint)
    assert _digest(real / ".git") == before
    status = _real_git(real, "status", "--porcelain").stdout
    assert " M notes.txt" in status
    assert "?? fresh.txt" in status
    assert session.proof().revert["real_is_clean_git"] is False
    reverted = session.revert_last()
    assert reverted.restored == ["notes.txt"]
    assert _digest(real / ".git") == before
    assert (real / "notes.txt").read_text() == "one\ntwo\n"
    assert (real / "fresh.txt").exists()


# --- rounds, status, discard and cleanup -----------------------------------------------------


def test_a_second_round_proves_against_the_applied_state(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    (session.work / "notes.txt").write_text("one\nTWO\n")
    first = session.proof()
    session.apply(first.fingerprint)
    assert (real / "notes.txt").read_text() == "one\nTWO\n"
    second = session.proof()
    assert second.changes == []
    assert second.revert["files_checked"] == 0
    assert second.revert["added"] == []
    assert second.fingerprint != first.fingerprint
    (session.work / "notes.txt").write_text("one\nTHREE\n")
    (session.work / "other.txt").write_text("other\n")
    third = session.proof()
    assert [change.path for change in third.changes] == ["notes.txt", "other.txt"]
    applied = session.apply(third.fingerprint)
    assert applied.written == ["notes.txt", "other.txt"]
    assert (real / "notes.txt").read_text() == "one\nTHREE\n"
    assert (real / "other.txt").read_text() == "other\n"


def test_status_shows_the_folders_counts_and_last_apply(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    (session.work / "notes.txt").write_text("one\nTWO\n")
    status = session.status()
    assert status["real"] == str(session.real)
    assert status["work"] == str(session.work)
    assert status["changes"] == {"added": 0, "modified": 1, "deleted": 0, "total": 1}
    assert status["last_apply"] is None
    assert status["bytes"] > 0
    session.apply(session.proof().fingerprint)
    status = session.status()
    assert status["changes"]["total"] == 0
    assert status["last_apply"]["files"] == ["notes.txt"]
    assert status["last_apply"]["time"]


def test_discard_removes_only_the_session_folder(real: Path, root: Path) -> None:
    session = safecopy.start(real, root, "s1")
    session.discard()
    assert not session.session_dir.exists()
    assert (real / "notes.txt").read_text() == "one\ntwo\n"


def test_discard_refuses_a_folder_outside_root(real: Path, root: Path, tmp_path: Path) -> None:
    session = safecopy.start(real, root, "s1")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "meta.json").write_text("{}")
    session.session_dir = outside
    with pytest.raises(safecopy.SafeCopyError) as caught:
        session.discard()
    assert caught.value.code == "refused_folder"
    assert outside.exists()
    assert (real / "notes.txt").exists()


def test_load_refuses_a_name_that_leaves_root(root: Path) -> None:
    with pytest.raises(safecopy.SafeCopyError) as caught:
        safecopy.load(root, "../elsewhere")
    assert caught.value.code == "not_found"
    with pytest.raises(safecopy.SafeCopyError) as caught:
        safecopy.load(root, "never-made")
    assert caught.value.code == "not_found"


def test_cleanup_keeps_the_named_sessions_and_the_young_ones(real: Path, root: Path) -> None:
    first = safecopy.start(real, root, "keep")
    safecopy.start(real, root, "drop")
    safecopy.start(real, root, "young")
    (root / "not-a-session").mkdir()
    removed = safecopy.cleanup(root, keep_ids={"keep", "young"}, max_age_days=14)
    assert removed == []
    removed = safecopy.cleanup(root, keep_ids={"keep", "young"}, max_age_days=0)
    assert removed == ["drop"]
    assert (root / "keep").exists()
    assert not (root / "drop").exists()
    assert (root / "young").exists()
    assert (root / "not-a-session").exists()
    assert first.session_dir.exists()
    assert (real / "notes.txt").exists()


# --- the promise: nothing is ever removed from the real folder -------------------------------


def test_no_step_removes_a_file_from_the_real_folder(real: Path, root: Path) -> None:
    (real / "old.txt").write_text("old\n")
    (real / "with space.txt").write_text("space\n")
    session = safecopy.start(real, root, "s1")
    work = session.work
    (work / "notes.txt").write_text("one\nTWO\n")
    (work / "fresh.txt").write_text("fresh\n")
    (work / "old.txt").unlink()
    (work / "sub" / "deep.txt").rename(work / "moved.txt")
    original = _files(real)
    proof = session.proof()
    assert _files(real) == original
    applied = session.apply(proof.fingerprint)
    assert _files(real) == original | {path for path in applied.written if path not in original}
    reverted = session.revert_last()
    assert _files(real) == original | {path for path in applied.written if path not in original}
    assert (real / "old.txt").read_text() == "old\n"
    assert (real / "sub" / "deep.txt").exists()
    assert (real / "moved.txt").exists()
    assert sorted(reverted.restored) == ["notes.txt"]
