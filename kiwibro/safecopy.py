"""Safe copy sessions: the agent works on a copy and only an approved, proven change reaches the real folder.

A session lives in ``<root>/<session_id>/`` with ``work/`` (the copy the agent edits), ``repo.git`` (a
git directory kept apart from the copy), ``meta.json`` and ``tmp/``. Every state of the copy is
committed in ``repo.git``, so a proof can state what changed, where and how to undo it. Nothing is
written to the real folder without a proof the user approved, and Kiwibro never removes anything
there: removals come back as commands the user runs.

Before a proof is handed out the undo step is rehearsed in ``tmp/``: the before versions go into a
scratch folder, the change is written into it exactly as apply would, the very revert command is run
against it, and every file it must restore is compared byte for byte with the real version. A
revert that does not bring the old bytes back raises ``revert_unproven`` and no proof exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SKIP_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".gradle",
        ".next",
        "target",
        ".cache",
    }
)
LINK_NAMES = frozenset({"node_modules", ".venv", "venv"})
GIT_TIMEOUT = 120.0
_LOCK_GUARD = threading.Lock()
_REPO_LOCKS: dict[str, threading.RLock] = {}
MAX_DIFF_LINES_PER_FILE = 400
MAX_DIFF_LINES_TOTAL = 4000
MAX_INLINE_PATHS = 200
GIT_OPTIONS = (
    "-c",
    "user.name=Kiwibro",
    "-c",
    "user.email=kiwibro@localhost",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.autocrlf=false",
    "-c",
    "core.quotePath=false",
    "-c",
    "gc.auto=0",
)
STRIPPED_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
)
KINDS = {"A": "added", "D": "deleted", "M": "modified", "T": "modified"}


class SafeCopyError(Exception):
    """A refusal or a failure a caller shows as a short sentence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Limits:
    """What a copy may hold; a folder that needs more is refused before anything is kept."""

    max_files: int = 20000
    max_bytes: int = 2 * 1024**3
    max_file_bytes: int = 64 * 1024**2


@dataclass(frozen=True)
class Change:
    """One path that differs between the baseline and the copy."""

    path: str
    kind: str
    added_lines: int
    removed_lines: int
    binary: bool


@dataclass(frozen=True)
class Proof:
    """What the user reads before anything reaches their folder."""

    changes: list[Change]
    totals: dict[str, int]
    diff: str
    conflicts: list[str]
    left_out: list[dict[str, str]]
    removals: list[dict[str, str]]
    before_commit: str
    revert: dict[str, Any]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON for the page that shows the proof."""
        return {
            "changes": [
                {
                    "path": change.path,
                    "kind": change.kind,
                    "added_lines": change.added_lines,
                    "removed_lines": change.removed_lines,
                    "binary": change.binary,
                }
                for change in self.changes
            ],
            "totals": dict(self.totals),
            "diff": self.diff,
            "conflicts": list(self.conflicts),
            "left_out": [dict(entry) for entry in self.left_out],
            "removals": [dict(entry) for entry in self.removals],
            "before_commit": self.before_commit,
            "revert": dict(self.revert),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class Applied:
    """What reached the real folder, and what is left for the user to run."""

    written: list[str]
    removals: list[dict[str, str]]
    revert: dict[str, Any]
    before_commit: str
    applied_commit: str
    left_out: list[dict[str, str]]


@dataclass(frozen=True)
class Reverted:
    """What came back after an undo, and the commands for the files that stay."""

    restored: list[str]
    kept: list[dict[str, str]]
    before_commit: str


def start(real: str | Path, root: str | Path, session_id: str, limits: Limits = Limits()) -> Session:
    """Copy a folder into a new session and record its first state."""
    source = Path(real)
    if not source.is_dir():
        raise SafeCopyError("not_a_folder", "That is not a folder, so nothing was copied.")
    real_path = source.resolve()
    _check_session_id(session_id)
    root_path = Path(root).resolve()
    if real_path.parent == real_path:
        raise SafeCopyError("refused_folder", "The whole filesystem cannot go into a safe copy.")
    if real_path == Path.home().resolve():
        raise SafeCopyError("refused_folder", "Your home folder itself cannot go into a safe copy.")
    if _within(root_path, real_path) or _within(real_path, root_path):
        raise SafeCopyError("refused_folder", "The session folder and the copied folder cannot hold each other.")
    session_dir = root_path / session_id
    if session_dir.exists():
        raise SafeCopyError("conflict", "A session with this name is already there.")
    try:
        session_dir.mkdir(parents=True)
    except OSError as exc:
        raise SafeCopyError("not_a_folder", f"The session folder could not be made. {exc}") from exc
    work = session_dir / "work"
    left_out: list[dict[str, str]] = []
    counted = {"files": 0, "bytes": 0}
    try:
        work.mkdir()
        (session_dir / "tmp").mkdir()
        _copy_folder(real_path, work, real_path, limits, left_out, counted)
    except SafeCopyError:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise SafeCopyError("not_a_folder", f"The folder could not be copied. {exc}") from exc
    repo = session_dir / "repo.git"
    try:
        _git(None, None, ["init", "--quiet", "--bare", str(repo)])
        _write_json(
            session_dir / "meta.json",
            {
                "session_id": session_id,
                "real": str(real_path),
                "created": time.time(),
                "baseline": None,
                "limits": {
                    "max_files": limits.max_files,
                    "max_bytes": limits.max_bytes,
                    "max_file_bytes": limits.max_file_bytes,
                },
                "left_out": left_out,
                "applies": [],
            },
        )
        session = Session(root_path, session_id)
        meta = session._read_meta()
        meta["baseline"] = session.commit("start")
        session._save_meta(meta)
    except BaseException:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    return session


def load(root: str | Path, session_id: str) -> Session:
    """Open a session made earlier by start()."""
    _check_session_id(session_id)
    root_path = Path(root).resolve()
    if not (root_path / session_id / "meta.json").is_file():
        raise SafeCopyError("not_found", "That safe copy session is not there.")
    session = Session(root_path, session_id)
    if not session.work.is_dir() or not session.repo.is_dir():
        raise SafeCopyError("not_found", "That safe copy session is not complete.")
    return session


def cleanup(root: str | Path, keep_ids: Iterable[str], max_age_days: float = 14) -> list[str]:
    """Discard sessions nobody named that are older than the given days, to bound disk use."""
    root_path = Path(root).resolve()
    kept = set(keep_ids)
    cutoff = time.time() - max_age_days * 86400
    removed: list[str] = []
    if not root_path.is_dir():
        return removed
    for child in sorted(root_path.iterdir()):
        if child.name in kept or child.is_symlink() or not child.is_dir():
            continue
        try:
            session = load(root_path, child.name)
        except SafeCopyError:
            continue
        if session.created >= cutoff:
            continue
        try:
            session.discard()
        except SafeCopyError:
            continue
        removed.append(session.id)
    return removed


class Session:
    """One safe copy: the copy the agent edits, its history, and what was applied to the real folder."""

    def __init__(self, root: Path, session_id: str) -> None:
        self.root = Path(root)
        self.id = session_id
        self.session_dir = self.root / session_id
        self.work = self.session_dir / "work"
        self.repo = self.session_dir / "repo.git"
        self.tmp = self.session_dir / "tmp"
        meta = self._read_meta()
        self.real = Path(meta["real"])
        self.created = float(meta.get("created", 0.0))

    @property
    def left_out(self) -> list[dict[str, str]]:
        """What the copy could not hold, with the reason, as start() found it."""
        return list(self._read_meta().get("left_out", []))

    def commit(self, message: str) -> str | None:
        """Record the copy as it is now; None when it is already recorded."""
        self._stage()
        head = self._head()
        if head is not None:
            changed = self._run(["diff", "--cached", "--name-only", "-z"]).stdout
            if not changed:
                return None
        self._run(["commit", "--quiet", "--allow-empty", "-m", message])
        return self._head()

    def changes(self) -> list[Change]:
        """What differs between the baseline and the copy, after recording pending edits."""
        self.commit("edit")
        baseline = self._read_meta().get("baseline")
        head = self._head()
        if not baseline or not head or baseline == head:
            return []
        status = self._run(["diff", "--name-status", "-z", "--no-renames", baseline, head, "--"]).stdout
        numbers = self._run(["diff", "--numstat", "-z", "--no-renames", baseline, head, "--"]).stdout
        counted = _parse_numstat(numbers)
        result = []
        for path, kind in _parse_name_status(status):
            added, removed, binary = counted.get(path, (0, 0, False))
            result.append(
                Change(path=path, kind=kind, added_lines=added, removed_lines=removed, binary=binary)
            )
        result.sort(key=lambda change: change.path)
        return result

    def proof(self) -> Proof:
        """Build the proof, after rehearsing the undo; raises revert_unproven when the undo fails."""
        changes = self.changes()
        meta = self._read_meta()
        baseline = meta.get("baseline")
        head = self._head()
        if not baseline or not head:
            raise SafeCopyError("git", "The copy has no record of its first state.")
        baseline_blobs = {path: oid for path, (_mode, oid) in self._blob_map(baseline).items()}
        copy_blobs = {path: oid for path, (_mode, oid) in self._blob_map(head).items()}
        left_out, conflicts = self._examine(changes, baseline_blobs, copy_blobs)
        left_paths = {entry["path"] for entry in left_out}
        added_paths = [c.path for c in changes if c.kind == "added" and c.path not in left_paths]
        modified_paths = [c.path for c in changes if c.kind == "modified" and c.path not in left_paths]
        existing_added = [
            path for path in added_paths if not (self.real / path).is_symlink() and (self.real / path).is_file()
        ]
        overwritten = sorted(set(modified_paths) | set(existing_added))
        before_commit, restore_paths = self._record_before(overwritten)
        write_paths = sorted(set(added_paths) | set(modified_paths))
        self._rehearse(before_commit, restore_paths, write_paths)
        fresh_added = [path for path in added_paths if path not in set(existing_added)]
        revert = self._revert_info(before_commit, restore_paths, fresh_added)
        removals = [
            {"path": change.path, "command": _rm_command(self.real / change.path)}
            for change in changes
            if change.kind == "deleted"
        ]
        totals = {
            "files": len(changes),
            "added": sum(1 for change in changes if change.kind == "added"),
            "modified": sum(1 for change in changes if change.kind == "modified"),
            "deleted": sum(1 for change in changes if change.kind == "deleted"),
            "added_lines": sum(change.added_lines for change in changes),
            "removed_lines": sum(change.removed_lines for change in changes),
        }
        return Proof(
            changes=changes,
            totals=totals,
            diff=self._diff_text(baseline, head, changes),
            conflicts=conflicts,
            left_out=left_out,
            removals=removals,
            before_commit=before_commit,
            revert=revert,
            fingerprint=_fingerprint(baseline, head, before_commit, changes),
        )

    def apply(self, fingerprint: str) -> Applied:
        """Write the proven change into the real folder; never removes and never renames anything."""
        proof = self.proof()
        if proof.fingerprint != fingerprint:
            raise SafeCopyError(
                "stale", "The files changed after the proof was made. Make a new proof and read it again."
            )
        if proof.conflicts:
            raise SafeCopyError(
                "conflict", "Files in your folder changed after the copy was made. Nothing was written."
            )
        left_paths = {entry["path"] for entry in proof.left_out}
        todo = [
            change.path
            for change in proof.changes
            if change.kind in {"added", "modified"} and change.path not in left_paths
        ]
        if not todo:
            raise SafeCopyError("nothing_to_apply", "There is nothing to write to your folder.")
        head = self._head()
        if not head:
            raise SafeCopyError("git", "The copy has no record, so nothing was written.")
        written: list[str] = []
        extra_left_out: list[dict[str, str]] = []
        try:
            for path in todo:
                reason = _refuse_target(self.real, path)
                if reason:
                    extra_left_out.append({"path": path, "reason": reason})
                    continue
                _write_copy_file(self.work, self.real, path)
                written.append(path)
        except OSError as exc:
            if not written:
                raise SafeCopyError(
                    "partial", "The write failed before any file was written. Nothing in your folder changed."
                ) from exc
            self._record_apply(proof, written, extra_left_out, head, partial=True)
            raise SafeCopyError(
                "partial",
                f"{len(written)} of {len(todo)} files were written before the write failed. "
                "Undo puts them back.",
            ) from exc
        if not written:
            raise SafeCopyError(
                "conflict", "Files in your folder got in the way while writing. Nothing was written."
            )
        left_out = self._record_apply(proof, written, extra_left_out, head, partial=False)
        return Applied(
            written=sorted(written),
            removals=proof.removals,
            revert=proof.revert,
            before_commit=proof.before_commit,
            applied_commit=head,
            left_out=left_out,
        )

    def _record_apply(
        self,
        proof: Proof,
        written: list[str],
        extra_left_out: list[dict[str, str]],
        head: str,
        *,
        partial: bool,
    ) -> list[dict[str, str]]:
        """Write down an apply so it can be undone; a half-done one keeps the old baseline."""
        left_out = proof.left_out + extra_left_out
        overwritten = list(proof.revert.get("paths", []))
        added = [entry["path"] for entry in proof.revert.get("added", [])]
        restored = overwritten
        missing: list[str] = []
        if partial:
            added = [path for path in added if path in set(written)]
            missing = [entry["path"] for entry in proof.revert.get("added", []) if entry["path"] not in set(written)]
            restored = [path for path in overwritten if path in set(written)]
        meta = self._read_meta()
        meta.setdefault("applies", []).append(
            {
                "time": time.time(),
                "fingerprint": proof.fingerprint,
                "written": sorted(written),
                "restored": restored,
                "revert_paths": overwritten,
                "missing_added": missing,
                "added": added,
                "before_commit": proof.before_commit,
                "applied_commit": head,
                "left_out": left_out,
                "removals": proof.removals,
                "partial": partial,
            }
        )
        if not partial:
            meta["baseline"] = head
        self._save_meta(meta)
        self._run(["update-ref", f"refs/kiwibro/apply/{proof.before_commit}", proof.before_commit], check=False)
        return left_out

    def revert_last(self) -> Reverted:
        """Put the last apply's files back; the files it added stay and come back as commands."""
        meta = self._read_meta()
        applies = meta.get("applies", [])
        if not applies:
            raise SafeCopyError("not_found", "Nothing has been written to your folder in this session.")
        record = applies[-1]
        if not self._still_as_applied(record):
            raise SafeCopyError(
                "conflict", "A file in your folder changed after it was written. Nothing was undone."
            )
        restored = list(record.get("restored", []))
        command = _revert_command(self.repo, self.real, record["before_commit"], restored)
        self._run_command(command)
        before_blobs = self._blob_map(record["before_commit"])
        for path in restored:
            if path not in before_blobs or self._real_blob(path) != before_blobs[path][1]:
                raise SafeCopyError(
                    "revert_unproven", "The undo did not bring the old files back. Look at your folder."
                )
        meta["baseline"] = self._baseline_after_revert(record)
        applies.pop()
        meta.setdefault("reverted", []).append(
            {"time": time.time(), "restored": restored, "kept": list(record.get("added", []))}
        )
        self._save_meta(meta)
        kept = [
            {"path": path, "command": _rm_command(self.real / path)} for path in record.get("added", [])
        ]
        return Reverted(restored=restored, kept=kept, before_commit=record["before_commit"])

    def last_apply(self) -> dict[str, Any] | None:
        """The newest apply this session recorded, as stored; None when nothing was applied."""
        applies = self._read_meta().get("applies", [])
        return dict(applies[-1]) if applies else None

    def can_undo(self) -> bool:
        """Whether revert_last would go through: an apply is there and its files are still as written."""
        applies = self._read_meta().get("applies", [])
        return self._still_as_applied(applies[-1]) if applies else False

    def _still_as_applied(self, record: dict[str, Any]) -> bool:
        """Whether every file the apply wrote still holds exactly what it wrote."""
        applied_commit = record.get("applied_commit")
        applied_blobs = self._blob_map(applied_commit) if applied_commit else {}
        for path in list(record.get("written", [])):
            source = self.work / path
            target = self.real / path
            if source.is_symlink():
                if not target.is_symlink() or os.readlink(target) != os.readlink(source):
                    return False
                continue
            if target.is_symlink() or not target.is_file():
                return False
            if path not in applied_blobs or self._real_blob(path) != applied_blobs[path][1]:
                return False
        return True

    def status(self) -> dict[str, Any]:
        """A few facts for a UI: where the two folders are, what waits, the last apply, disk use."""
        changes = self.changes()
        counts = {"added": 0, "modified": 0, "deleted": 0}
        for change in changes:
            counts[change.kind] = counts.get(change.kind, 0) + 1
        applies = self._read_meta().get("applies", [])
        last_apply = None
        if applies:
            last = applies[-1]
            last_apply = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last.get("time", 0.0))),
                "files": list(last.get("written", [])),
            }
        return {
            "id": self.id,
            "real": str(self.real),
            "work": str(self.work),
            "changes": {**counts, "total": len(changes)},
            "last_apply": last_apply,
            "bytes": _disk_bytes(self.session_dir),
        }

    def discard(self) -> None:
        """Remove this session's own folder; the real folder is never touched."""
        root = self.root.resolve()
        session_dir = self.session_dir.resolve()
        if not _within(session_dir, root) or session_dir == root:
            raise SafeCopyError(
                "refused_folder", "That folder is not a session inside the session folder, so nothing was removed."
            )
        if not (session_dir / "meta.json").is_file():
            raise SafeCopyError("not_found", "That safe copy session is not there.")
        real = self.real.resolve()
        if _within(real, session_dir) or _within(session_dir, real):
            raise SafeCopyError(
                "refused_folder", "That session folder holds the folder it copies, so nothing was removed."
            )
        shutil.rmtree(session_dir)

    def _read_meta(self) -> dict[str, Any]:
        path = self.session_dir / "meta.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SafeCopyError("not_found", "That safe copy session is not there.") from exc
        except (OSError, ValueError) as exc:
            raise SafeCopyError("not_found", "That safe copy session cannot be read.") from exc

    def _save_meta(self, meta: dict[str, Any]) -> None:
        _write_json(self.session_dir / "meta.json", meta)

    def _run(
        self,
        args: list[str],
        *,
        work_tree: Path | None = None,
        index: Path | None = None,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        return _git(
            self.repo,
            self.work if work_tree is None else work_tree,
            args,
            index=index,
            input_bytes=input_bytes,
            check=check,
        )

    def _head(self) -> str | None:
        proc = self._run(["rev-parse", "--verify", "HEAD"], check=False)
        if proc.returncode != 0:
            return None
        return _text(proc.stdout).strip() or None

    def _blob_map(self, commit: str) -> dict[str, tuple[str, str]]:
        return _parse_tree(self._run(["ls-tree", "-z", "-r", commit]).stdout)

    def _stage(self) -> None:
        self._run(["add", "-A", "-f", "--", "."])
        drop = [path for mode, path in _parse_index(self._run(["ls-files", "-s", "-z"]).stdout) if mode == "160000"]
        for path in self._tracked_names():
            drop.append(path)
        if drop:
            payload = b"".join(_bytes(path) + b"\0" for path in drop)
            self._run(["update-index", "--force-remove", "-z", "--stdin"], input_bytes=payload)

    def _tracked_names(self) -> list[str]:
        tracked = [_text(part) for part in _split_z(self._run(["ls-files", "-z"]).stdout)]
        return [
            path
            for path in tracked
            if any(part in SKIP_NAMES or part in LINK_NAMES for part in path.split("/"))
        ]

    def _real_blob(self, path: str) -> str:
        proc = self._run(["hash-object", "--", str(self.real / path)])
        return _text(proc.stdout).strip()

    def _examine(
        self,
        changes: list[Change],
        baseline_blobs: dict[str, str],
        copy_blobs: dict[str, str],
    ) -> tuple[list[dict[str, str]], list[str]]:
        left_out: list[dict[str, str]] = []
        conflicts: list[str] = []
        for change in changes:
            path = change.path
            if change.kind in {"added", "modified"}:
                reason = _refuse_target(self.real, path)
                if reason:
                    left_out.append({"path": path, "reason": reason})
                    continue
                if _link_points_outside(self.work / path, self.work):
                    left_out.append(
                        {
                            "path": path,
                            "reason": "This is a link that points outside the folder, so it is not applied.",
                        }
                    )
                    continue
            target = self.real / path
            if change.kind == "added":
                if target.is_symlink():
                    left_out.append(
                        {"path": path, "reason": "This is a link in your folder and is never written through."}
                    )
                    continue
                if not target.exists():
                    continue
                if not target.is_file() or self._real_blob(path) != copy_blobs.get(path):
                    conflicts.append(path)
                continue
            if target.is_symlink() or not target.is_file():
                conflicts.append(path)
                continue
            if self._real_blob(path) != baseline_blobs.get(path):
                conflicts.append(path)
        return left_out, conflicts

    def _record_before(self, paths: list[str]) -> tuple[str, list[str]]:
        folder = self.tmp / "before"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        present: list[str] = []
        for path in paths:
            source = self.real / path
            if source.is_symlink() or not source.is_file():
                continue
            target = folder / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            present.append(path)
        index = self.tmp / "before.index"
        index.unlink(missing_ok=True)
        self._run(["read-tree", "--empty"], work_tree=folder, index=index)
        if present:
            if len(present) > MAX_INLINE_PATHS:
                payload = b"".join(_bytes(path) + b"\0" for path in present)
                self._run(
                    ["add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"],
                    work_tree=folder,
                    index=index,
                    input_bytes=payload,
                )
            else:
                self._run(["add", "-f", "--", *present], work_tree=folder, index=index)
        tree = _text(self._run(["write-tree"], work_tree=folder, index=index).stdout).strip()
        current = self._ref("refs/kiwibro/before")
        if current:
            current_tree = _text(self._run(["rev-parse", f"{current}^{{tree}}"], check=False).stdout).strip()
            if current_tree == tree:
                return current, present
        commit = _text(self._run(["commit-tree", tree, "-m", "before"]).stdout).strip()
        self._run(["update-ref", "refs/kiwibro/before", commit])
        return commit, present

    def _ref(self, name: str) -> str | None:
        proc = self._run(["rev-parse", "--verify", name], check=False)
        if proc.returncode != 0:
            return None
        return _text(proc.stdout).strip() or None

    def _rehearse(self, before_commit: str, restore_paths: list[str], write_paths: list[str]) -> None:
        self.tmp.mkdir(parents=True, exist_ok=True)
        rehearsal = self.tmp / "rehearsal"
        probe = self.tmp / "probe"
        shutil.rmtree(rehearsal, ignore_errors=True)
        shutil.rmtree(probe, ignore_errors=True)
        rehearsal.mkdir()
        probe.mkdir()
        index = self.tmp / "rehearsal.index"
        index.unlink(missing_ok=True)
        if restore_paths:
            self._run(["checkout", before_commit, "--", *restore_paths], work_tree=rehearsal, index=index)
        for path in sorted(set(restore_paths) | set(write_paths)):
            _copy_attribute_files(self.real, rehearsal, path)
        for path in restore_paths:
            target = probe / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.tmp / "before" / path, target)
        for path in write_paths:
            _write_copy_file(self.work, rehearsal, path)
        self._run_command(_revert_command(self.repo, rehearsal, before_commit, restore_paths))
        for path in restore_paths:
            if not _same_bytes(probe / path, rehearsal / path):
                raise SafeCopyError(
                    "revert_unproven", "The undo did not bring the old files back, so no proof was made."
                )

    def _run_command(self, command: str) -> None:
        self.tmp.mkdir(parents=True, exist_ok=True)
        index = self.tmp / "command.index"
        index.unlink(missing_ok=True)
        try:
            with _repo_lock(self.repo):  # same repository as the other git calls: one at a time
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    env=_git_env(index),
                    timeout=GIT_TIMEOUT,
                )
        except subprocess.TimeoutExpired as exc:
            raise SafeCopyError("git", "The undo command took too long and was stopped.") from exc
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace").strip()
            raise SafeCopyError("git", f"The undo command did not finish. {detail}".strip())

    def _revert_info(
        self, before_commit: str, restore_paths: list[str], fresh_added: list[str]
    ) -> dict[str, Any]:
        clean = self._real_git_state()
        info: dict[str, Any] = {
            "command": _revert_command(self.repo, self.real, before_commit, restore_paths),
            "paths": list(restore_paths),
            "added": [{"path": path, "command": _rm_command(self.real / path)} for path in fresh_added],
            "verified": True,
            "files_checked": len(restore_paths),
            "real_is_clean_git": clean,
        }
        if clean and restore_paths:
            restores = " ".join(shlex.quote(path) for path in restore_paths)
            info["git_restore"] = f"git -C {shlex.quote(str(self.real))} restore -- {restores}"
        return info

    def _real_git_state(self) -> bool:
        proc = _git(None, None, ["-C", str(self.real), "rev-parse", "--show-toplevel"], check=False)
        if proc.returncode != 0 or _text(proc.stdout).strip() != str(self.real):
            return False
        status = _git(None, None, ["-C", str(self.real), "status", "--porcelain"], check=False)
        return status.returncode == 0 and not status.stdout.strip()

    def _diff_text(self, baseline: str, head: str, changes: list[Change]) -> str:
        if not changes:
            return ""
        text = _text(
            self._run(
                ["diff", "--no-renames", "--no-color", "--no-ext-diff", "--unified=3", baseline, head, "--"]
            ).stdout
        )
        sections = _split_diff(text)
        if len(sections) != len(changes):
            sections = []
            for change in changes:
                single = self._run(
                    [
                        "diff",
                        "--no-renames",
                        "--no-color",
                        "--no-ext-diff",
                        "--unified=3",
                        baseline,
                        head,
                        "--",
                        change.path,
                    ]
                ).stdout
                sections.append(_text(single))
        pieces: list[str] = []
        used = 0
        for position, change in enumerate(changes):
            if change.binary:
                pieces.append(f"{change.path}: binary file")
                used += 1
                if used >= MAX_DIFF_LINES_TOTAL:
                    rest = len(changes) - position - 1
                    if rest:
                        pieces.append(f"... {rest} more files not shown")
                    break
                continue
            lines = sections[position].splitlines()
            allowed = min(MAX_DIFF_LINES_PER_FILE, MAX_DIFF_LINES_TOTAL - used)
            if allowed <= 1:
                pieces.append(f"... {len(changes) - position} more files not shown")
                break
            if len(lines) > allowed:
                shown = lines[: allowed - 1]
                pieces.extend(shown)
                pieces.append(f"... {len(lines) - len(shown)} more lines of this file not shown")
                used += allowed
            else:
                pieces.extend(lines)
                used += len(lines)
        return "\n".join(pieces)

    def _baseline_after_revert(self, record: dict[str, Any]) -> str:
        applied = record.get("applied_commit")
        before = record["before_commit"]
        restored = list(record.get("revert_paths", record.get("restored", [])))
        if not applied:
            return before
        self.tmp.mkdir(parents=True, exist_ok=True)
        index = self.tmp / "baseline.index"
        index.unlink(missing_ok=True)
        self._run(["read-tree", applied], index=index)
        before_blobs = self._blob_map(before)
        for path in restored:
            if path not in before_blobs:
                continue
            mode, oid = before_blobs[path]
            self._run(["update-index", "--add", "--cacheinfo", mode, oid, path], index=index)
        missing = record.get("missing_added", [])
        if missing:
            payload = b"".join(_bytes(path) + b"\0" for path in missing)
            self._run(["update-index", "--force-remove", "-z", "--stdin"], index=index, input_bytes=payload)
        tree = _text(self._run(["write-tree"], index=index).stdout).strip()
        return _text(self._run(["commit-tree", tree, "-m", "after undo"]).stdout).strip()


def _git(
    git_dir: Path | None,
    work_tree: Path | None,
    args: list[str],
    *,
    index: Path | None = None,
    input_bytes: bytes | None = None,
    check: bool = True,
    timeout: float = GIT_TIMEOUT,
) -> subprocess.CompletedProcess[bytes]:
    lock = _repo_lock(git_dir)
    with lock:  # the turn thread records a step while the page asks for a proof: one git at a time
        return _git_call(
            git_dir, work_tree, args, index=index, input_bytes=input_bytes, check=check, timeout=timeout
        )


def _repo_lock(git_dir: Path | None) -> threading.RLock:
    """One re-entrant lock per session repository, so two threads never fight over its index."""
    if git_dir is None:
        return threading.RLock()
    key = str(git_dir)
    with _LOCK_GUARD:
        lock = _REPO_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _REPO_LOCKS[key] = lock
        return lock


def _git_call(
    git_dir: Path | None,
    work_tree: Path | None,
    args: list[str],
    *,
    index: Path | None = None,
    input_bytes: bytes | None = None,
    check: bool = True,
    timeout: float = GIT_TIMEOUT,
) -> subprocess.CompletedProcess[bytes]:
    command = ["git"]
    if git_dir is not None:
        command.append(f"--git-dir={git_dir}")
    if work_tree is not None:
        command.append(f"--work-tree={work_tree}")
    command.extend(GIT_OPTIONS)
    command.extend(args)
    try:
        proc = subprocess.run(
            command,
            input=input_bytes,
            capture_output=True,
            env=_git_env(index),
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise SafeCopyError("git", "Git is not installed, so nothing could be recorded.") from exc
    except subprocess.TimeoutExpired as exc:
        raise SafeCopyError("git", "Git took too long, so nothing could be recorded.") from exc
    if check and proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise SafeCopyError("git", f"Git could not record the change. {detail}".strip())
    return proc


def _git_env(index: Path | None = None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in STRIPPED_ENV
        and key != "GIT_CONFIG_COUNT"
        and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    return env


def _copy_folder(
    source: Path,
    target: Path,
    real_root: Path,
    limits: Limits,
    left_out: list[dict[str, str]],
    counted: dict[str, int],
) -> None:
    with os.scandir(source) as scanner:
        entries = sorted(scanner, key=lambda entry: entry.name)
    for entry in entries:
        name = entry.name
        relative = _relative(entry.path, real_root)
        if name in SKIP_NAMES:
            continue
        if name in LINK_NAMES:
            os.symlink(str(real_root / relative), target / name)
            continue
        if entry.is_symlink():
            if _link_points_outside(Path(entry.path), real_root):
                left_out.append(
                    {
                        "path": relative,
                        "reason": "This is a link that points outside the folder, so it was not copied.",
                    }
                )
            else:
                os.symlink(os.readlink(entry.path), target / name)
                _keep_link_times(Path(entry.path), target / name)
            continue
        if entry.is_dir(follow_symlinks=False):
            (target / name).mkdir()
            _copy_folder(Path(entry.path), target / name, real_root, limits, left_out, counted)
            continue
        if entry.is_file(follow_symlinks=False):
            size = entry.stat(follow_symlinks=False).st_size
            if size > limits.max_file_bytes:
                left_out.append(
                    {
                        "path": relative,
                        "reason": (
                            f"This file is bigger than the {limits.max_file_bytes} byte limit, "
                            "so it was not copied."
                        ),
                    }
                )
                continue
            counted["files"] += 1
            counted["bytes"] += size
            if counted["files"] > limits.max_files or counted["bytes"] > limits.max_bytes:
                raise SafeCopyError(
                    "too_big",
                    f"This folder is too big: {counted['files']} files and {counted['bytes']} bytes, "
                    f"the limit is {limits.max_files} files and {limits.max_bytes} bytes.",
                )
            shutil.copy2(entry.path, target / name, follow_symlinks=False)
            continue
        left_out.append(
            {"path": relative, "reason": "This is not a normal file or folder, so it was not copied."}
        )


def _refuse_target(real: Path, path: str) -> str | None:
    parts = path.split("/")
    if path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        return "This path does not stay inside the folder, so it is not applied."
    if ".git" in parts or any(part in SKIP_NAMES for part in parts):
        return "This path is never applied."
    parent = real
    for part in parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            return "There is a link in your folder in the way, so nothing is written there."
        if parent.exists() and not parent.is_dir():
            return "A file is in your folder in the way, so nothing is written there."
    if (real / path).is_symlink():
        return "This is a link in your folder and is never written through."
    return None


def _write_copy_file(copy_root: Path, target_root: Path, path: str) -> None:
    source = copy_root / path
    target = target_root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        temporary = target.parent / f".kiwibro-{uuid.uuid4().hex}.tmp"
        os.symlink(os.readlink(source), temporary)
        try:
            os.replace(temporary, target)
        except OSError:
            os.unlink(temporary)
            raise
        return
    mode = stat.S_IMODE(os.stat(source, follow_symlinks=False).st_mode)
    handle, name = tempfile.mkstemp(dir=target.parent, prefix=".kiwibro-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out, open(source, "rb") as incoming:
            shutil.copyfileobj(incoming, out, 1024 * 1024)
        os.chmod(name, mode)
        os.replace(name, target)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _copy_attribute_files(source_root: Path, target_root: Path, path: str) -> None:
    parts = Path(path).parts[:-1]
    for depth in range(len(parts), -1, -1):
        relative = Path(*parts[:depth]) / ".gitattributes"
        source = source_root / relative
        target = target_root / relative
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _revert_command(repo: Path, work_tree: Path, before_commit: str, paths: list[str]) -> str:
    if not _is_commit(before_commit):
        raise SafeCopyError("git", "The undo point is not a valid commit, so nothing was done.")
    for path in paths:
        if path.startswith("/") or ".." in path.split("/"):
            raise SafeCopyError("git", "A path does not stay inside the folder, so nothing was done.")
    if not paths:
        return "true"
    quoted = " ".join(shlex.quote(path) for path in paths)
    return (
        f"git --git-dir={shlex.quote(str(repo))} --work-tree={shlex.quote(str(work_tree))} "
        f"checkout {before_commit} -- {quoted}"
    )


def _is_commit(value: str) -> bool:
    return len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value)


def _rm_command(path: Path) -> str:
    return f"rm -- {shlex.quote(str(path))}"


def _fingerprint(baseline: str, head: str, before_commit: str, changes: list[Change]) -> str:
    payload = json.dumps(
        {
            "baseline": baseline,
            "copy": head,
            "before": before_commit,
            "changes": [[change.path, change.kind] for change in changes],
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _same_bytes(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with open(left, "rb") as first, open(right, "rb") as second:
            while True:
                chunk = first.read(1024 * 1024)
                if chunk != second.read(1024 * 1024):
                    return False
                if not chunk:
                    return True
    except OSError:
        return False


def _disk_bytes(folder: Path) -> int:
    total = 0
    for current, _folders, files in os.walk(folder, followlinks=False):
        for name in files:
            try:
                info = os.lstat(os.path.join(current, name))
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _relative(path: str, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _link_points_outside(link: Path, root: Path) -> bool:
    if not link.is_symlink():
        return False
    return not _within(Path(os.path.realpath(link)), root)


def _keep_link_times(source: Path, target: Path) -> None:
    try:
        info = os.lstat(source)
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns), follow_symlinks=False)
    except OSError:
        pass


def _within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _check_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise SafeCopyError("not_found", "That session name cannot be used.")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _split_z(data: bytes) -> list[bytes]:
    return [part for part in data.split(b"\0") if part]


def _parse_name_status(data: bytes) -> list[tuple[str, str]]:
    fields = _split_z(data)
    pairs: list[tuple[str, str]] = []
    for index in range(0, len(fields) - 1, 2):
        letter = _text(fields[index])[:1]
        pairs.append((_text(fields[index + 1]), KINDS.get(letter, "modified")))
    return pairs


def _parse_numstat(data: bytes) -> dict[str, tuple[int, int, bool]]:
    numbers: dict[str, tuple[int, int, bool]] = {}
    for record in _split_z(data):
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            continue
        added, removed, path = fields
        binary = added == b"-" or removed == b"-"
        numbers[_text(path)] = (
            0 if binary else int(added),
            0 if binary else int(removed),
            binary,
        )
    return numbers


def _parse_tree(data: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in _split_z(data):
        header, _tab, path = record.partition(b"\t")
        fields = header.split(b" ")
        if len(fields) != 3 or fields[1] == b"commit":
            continue
        entries[_text(path)] = (_text(fields[0]), _text(fields[2]))
    return entries


def _parse_index(data: bytes) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for record in _split_z(data):
        header, _tab, path = record.partition(b"\t")
        fields = header.split(b" ")
        if len(fields) < 2:
            continue
        entries.append((_text(fields[0]), _text(path)))
    return entries


def _split_diff(text: str) -> list[str]:
    sections: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                sections.append("\n".join(current))
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        sections.append("\n".join(current))
    return sections


def _text(data: bytes) -> str:
    return data.decode("utf-8", "surrogateescape")


def _bytes(text: str) -> bytes:
    return text.encode("utf-8", "surrogateescape")
