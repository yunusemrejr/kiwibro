"""Automatic file checkpoints: every agent file change can be undone.

Before a file-changing tool runs, the files it may touch are snapshotted into a
content-addressed, zlib-compressed object store under the data directory, with
one journal row per file in a small SQLite database. ``undo_changes`` restores
the newest checkpoint of a conversation (optionally for one path). A restore
snapshots the current state first, so nothing is ever lost by undoing. The
store is bounded in bytes and rows and pruned oldest-first.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .extensions import ExtensionTool, data_dir_for, schema, tool_result

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES_PER_CALL = 500
MAX_STORE_BYTES = 512 * 1024 * 1024
MAX_ROWS = 20_000
UNDO_TOOL = "undo_changes"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY,
    batch TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    path TEXT NOT NULL,
    display TEXT NOT NULL,
    existed INTEGER NOT NULL,
    sha256 TEXT,
    size INTEGER NOT NULL DEFAULT 0,
    mode INTEGER,
    created_at REAL NOT NULL,
    restored INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS snapshots_conversation ON snapshots(conversation_id, id);
CREATE INDEX IF NOT EXISTS snapshots_sha ON snapshots(sha256);
"""


class CheckpointStore:
    """Journal + object store rooted at ``<data_dir>/checkpoints``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.root / "journal.db", timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- objects -------------------------------------------------------------------

    def _object_path(self, sha: str) -> Path:
        return self.objects / sha[:2] / sha

    def _store_blob(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        target = self._object_path(sha)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{sha}.{uuid.uuid4().hex}")
            temporary.write_bytes(zlib.compress(data, 6))
            os.replace(temporary, target)
        return sha

    def _load_blob(self, sha: str) -> bytes:
        return zlib.decompress(self._object_path(sha).read_bytes())

    # --- snapshots -----------------------------------------------------------------

    def _rows(self, files: list[tuple[Path, str]], conversation_id: str, tool: str) -> list[tuple[Any, ...]]:
        batch, now, rows = uuid.uuid4().hex[:12], time.time(), []
        for path, display in files[:MAX_FILES_PER_CALL]:
            existed, sha, size, mode = 0, None, 0, None
            try:
                if path.is_file():
                    stat = path.stat()
                    existed, size, mode = 1, stat.st_size, stat.st_mode & 0o7777
                    if size <= MAX_FILE_BYTES:
                        sha = self._store_blob(path.read_bytes())
            except OSError:
                log.warning("could not snapshot %s", display, exc_info=True)
                continue
            rows.append((batch, conversation_id, tool, str(path), display, existed, sha, size, mode, now))
        return rows

    def _insert(self, rows: list[tuple[Any, ...]]) -> str | None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO snapshots (batch, conversation_id, tool, path, display, existed, sha256, size, mode, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return rows[0][0] if rows else None

    def snapshot(self, files: list[tuple[Path, str]], *, conversation_id: str, tool: str) -> str | None:
        """Record the current state of ``(absolute path, display path)`` pairs. Returns the batch id."""
        if not files:
            return None
        with self._lock:
            batch = self._insert(self._rows(files, conversation_id, tool))
        self.prune()
        return batch

    def history(self, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT batch, tool, MIN(created_at) AS created_at, COUNT(*) AS files, GROUP_CONCAT(display, ', ') "
                "AS paths, MAX(restored) AS restored FROM snapshots WHERE conversation_id = ? GROUP BY batch "
                "ORDER BY MAX(id) DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def undo(self, conversation_id: str, *, display: str | None = None, steps: int = 1) -> list[dict[str, Any]]:
        """Restore the newest unrestored change(s); returns one report per restored file."""
        reports: list[dict[str, Any]] = []
        with self._lock:
            for _ in range(max(1, steps)):
                rows = self._latest(conversation_id, display)
                if not rows:
                    break
                reports += self._restore(rows, conversation_id)
        return reports

    def _latest(self, conversation_id: str, display: str | None) -> list[sqlite3.Row]:
        with self._connect() as conn:
            if display is not None:
                row = conn.execute(
                    "SELECT * FROM snapshots WHERE conversation_id = ? AND display = ? AND restored = 0 "
                    "AND tool != ? ORDER BY id DESC LIMIT 1",
                    (conversation_id, display, UNDO_TOOL),
                ).fetchone()
                return [row] if row else []
            latest = conn.execute(
                "SELECT batch FROM snapshots WHERE conversation_id = ? AND restored = 0 AND tool != ? "
                "ORDER BY id DESC LIMIT 1",
                (conversation_id, UNDO_TOOL),
            ).fetchone()
            if latest is None:
                return []
            return conn.execute(
                "SELECT * FROM snapshots WHERE batch = ? AND restored = 0 ORDER BY id", (latest["batch"],)
            ).fetchall()

    def _restore(self, rows: list[sqlite3.Row], conversation_id: str) -> list[dict[str, Any]]:
        # Snapshot the pre-undo state first so an undo never loses anything.
        self._insert(self._rows([(Path(row["path"]), row["display"]) for row in rows], conversation_id, UNDO_TOOL))
        reports = []
        for row in rows:
            path, display = Path(row["path"]), row["display"]
            try:
                if not row["existed"]:
                    removed = path.is_file()
                    if removed:
                        path.unlink()
                    action = "removed (it did not exist before)" if removed else "already absent"
                    reports.append({"path": display, "action": action})
                elif row["sha256"] is None:
                    reports.append({"path": display, "action": "skipped: file was too large to checkpoint"})
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(f".{path.name}.kiwibro-restore")
                    temporary.write_bytes(self._load_blob(row["sha256"]))
                    if row["mode"] is not None:
                        os.chmod(temporary, row["mode"])
                    os.replace(temporary, path)
                    reports.append({"path": display, "action": f"restored ({row['size']:,} bytes)"})
            except OSError as exc:
                reports.append({"path": display, "action": f"failed: {exc.strerror or exc}"})
                continue
            with self._connect() as conn:
                conn.execute("UPDATE snapshots SET restored = 1 WHERE id = ?", (row["id"],))
        for report, row in zip(reports, rows):
            report["tool"] = row["tool"]
            report["when"] = time.strftime("%H:%M:%S", time.localtime(row["created_at"]))
        return reports

    # --- bounds --------------------------------------------------------------------

    def prune(self) -> int:
        """Drop the oldest batches beyond the row/byte budget and their orphaned objects."""
        with self._lock, self._connect() as conn:
            rows, size = conn.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM snapshots").fetchone()
            removed = 0
            while rows > MAX_ROWS or size > MAX_STORE_BYTES:
                oldest = conn.execute("SELECT batch FROM snapshots ORDER BY id LIMIT 1").fetchone()
                if oldest is None:
                    break
                doomed = conn.execute(
                    "SELECT sha256, size FROM snapshots WHERE batch = ?", (oldest["batch"],)
                ).fetchall()
                conn.execute("DELETE FROM snapshots WHERE batch = ?", (oldest["batch"],))
                rows -= len(doomed)
                size -= sum(row["size"] for row in doomed)
                removed += len(doomed)
                for row in doomed:
                    sha = row["sha256"]
                    if sha and conn.execute("SELECT 1 FROM snapshots WHERE sha256 = ? LIMIT 1", (sha,)).fetchone() is None:
                        self._object_path(sha).unlink(missing_ok=True)
            return removed


_STORES: dict[Path, CheckpointStore] = {}
_STORES_LOCK = threading.Lock()


def store_for(data_dir: Path) -> CheckpointStore:
    root = (Path(data_dir) / "checkpoints").resolve()
    with _STORES_LOCK:
        store = _STORES.get(root)
        if store is None:
            store = _STORES[root] = CheckpointStore(root)
        return store


# --- which files a tool call may touch -------------------------------------------------


def _resolve(workspace: Any, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return workspace.resolve(raw)
    except Exception:  # outside the workspace: the tool itself will refuse
        return None


def _expand(workspace: Any, path: Path | None) -> list[tuple[Path, str]]:
    if path is None:
        return []
    if path.is_dir():
        files = []
        for directory, _dirs, names in os.walk(path):
            for name in names:
                files.append(Path(directory) / name)
                if len(files) >= MAX_FILES_PER_CALL:
                    break
            if len(files) >= MAX_FILES_PER_CALL:
                break
        return [(file, workspace.public(file)) for file in files]
    return [(path, workspace.public(path))]


def _landing(workspace: Any, source: Path, destination: Path) -> list[tuple[Path, str]]:
    """Destination files a move/copy will create or overwrite (one per source file)."""
    if source.is_dir():
        targets = [destination / path.relative_to(source) for path, _display in _expand(workspace, source)]
    else:
        targets = [destination]
    return [(target, workspace.public(target)) for target in targets]


def files_for(name: str, args: dict[str, Any], workspace: Any) -> list[tuple[Path, str]]:
    """Files a file-changing tool call may modify, as (absolute, display) pairs."""
    if name in ("write_file", "replace_in_file", "apply_patch"):
        return _expand(workspace, _resolve(workspace, args.get("path")))
    if name == "file_manage":
        action = args.get("action")
        source = _resolve(workspace, args.get("path"))
        files = _expand(workspace, source) if action in ("move", "delete") else []
        destination = _resolve(workspace, args.get("destination")) if action in ("move", "copy") else None
        if source is not None and destination is not None:
            if destination.is_dir():
                destination = destination / source.name
            files += _landing(workspace, source, destination)
        return files
    return []


def snapshot_call(data_dir: Path, workspace: Any, name: str, args: dict[str, Any], conversation_id: str) -> str | None:
    """Checkpoint everything a tool call may change. Never raises."""
    try:
        files = files_for(name, args, workspace)
        if not files:
            return None
        return store_for(data_dir).snapshot(files, conversation_id=conversation_id, tool=name)
    except Exception:  # checkpoints must never block the edit itself
        log.warning("checkpoint before %s failed", name, exc_info=True)
        return None


# --- tool ----------------------------------------------------------------------------------


def _validate_undo(args: dict[str, Any]) -> dict[str, Any]:
    path = args.get("path")
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("path must be a non-empty workspace-relative path")
    steps = args.get("steps", 1)
    try:
        steps = int(steps)
    except (TypeError, ValueError) as exc:
        raise ValueError("steps must be an integer") from exc
    if not 1 <= steps <= 10:
        raise ValueError("steps must be between 1 and 10")
    return {"path": path.strip() if isinstance(path, str) else None, "steps": steps}


def _execute_undo(args: dict[str, Any], ctx: Any) -> Any:
    store = store_for(data_dir_for(ctx))
    display = None
    if args.get("path"):
        resolved = _resolve(ctx.workspace, args["path"])
        if resolved is None:
            raise ValueError(f"path is not inside the workspace: {args['path']}")
        display = ctx.workspace.public(resolved)
    reports = store.undo(ctx.conversation_id, display=display, steps=args["steps"])
    if not reports:
        scope = f" for {display}" if display else ""
        return tool_result(True, f"Nothing to undo{scope}: no file changes from this chat are left to restore.")
    lines = [f"- {item['path']}: {item['action']} (before {item['tool']} at {item['when']})" for item in reports]
    content = "Undid the most recent file changes:\n" + "\n".join(lines)
    return tool_result(True, content, meta={"restored": len(reports)})


UNDO_CHANGES = ExtensionTool(
    name=UNDO_TOOL,
    description=(
        "Undo your recent file changes in this chat. Kiwibro checkpoints every file write, edit, patch, "
        "move and delete automatically; this restores the files exactly as they were before the last change. "
        "Use it right away when an edit went wrong. Optional: one path only, or several steps back."
    ),
    parameters=schema(
        {
            "path": {"type": "string", "description": "Only undo changes to this workspace file"},
            "steps": {"type": "integer", "description": "How many changes to undo (1-10, default 1)", "minimum": 1, "maximum": 10},
        }
    ),
    risk="write",
    phase="writing",
    label="Undoing changes",
    summary=lambda args: f"Undo the last {args.get('steps', 1)} change(s)" + (f" to {args['path']}" if args.get("path") else ""),
    validate=_validate_undo,
    execute=_execute_undo,
    permission="auto",
    category="coding",
)

TOOLS = (UNDO_CHANGES,)
