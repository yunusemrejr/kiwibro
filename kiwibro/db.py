"""SQLite persistence for Kiwibro: schema, forward migrations, CRUD helpers."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import config

log = logging.getLogger(__name__)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Any, fallback: Any = None) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


# --- migrations -------------------------------------------------------------

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New chat',
    title_auto INTEGER NOT NULL DEFAULT 1,
    model TEXT,
    summary TEXT NOT NULL DEFAULT '',
    summary_through_seq INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '',
    tool_calls TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    model TEXT,
    tokens INTEGER,
    meta TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(conversation_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
    ON messages(conversation_id, seq);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'fact',
    content TEXT NOT NULL,
    normalized TEXT NOT NULL,
    tags TEXT,
    source_conversation_id TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED, content, normalized, tokenize='unicode61'
);
CREATE TABLE IF NOT EXISTS permissions (
    tool TEXT PRIMARY KEY,
    mode TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    model TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""

_MIGRATIONS: list[tuple[int, str]] = [(1, _SCHEMA_V1)]


def fts_query(query: str) -> str | None:
    """Turn a user query into a safe FTS5 MATCH expression (OR of quoted tokens)."""
    tokens = [t for t in re.findall(r"[^\W_]+", query or "", flags=re.UNICODE) if len(t) >= 1]
    if not tokens:
        return None
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    quoted = ['"' + token.replace('"', '""') + '"' for token in seen[:12]]
    return " OR ".join(quoted)


class Database:
    """Thread-safe (per-call connections) SQLite wrapper."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._init_lock = threading.Lock()
        self._initialized = False

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            conn.close()

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
                )
                applied = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
                for version, script in _MIGRATIONS:
                    if version in applied:
                        continue
                    conn.executescript(script)
                    conn.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (version, config.now_iso()),
                    )
            self._initialized = True

    # --- conversations -----------------------------------------------------

    @staticmethod
    def _conversation(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "title_auto": bool(row["title_auto"]),
            "model": row["model"],
            "summary": row["summary"] or "",
            "summary_through_seq": row["summary_through_seq"],
            "pinned": bool(row["pinned"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_conversation(self, title: str = "New chat", model: str | None = None) -> dict[str, Any]:
        now = config.now_iso()
        cid = _new_id("c")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO conversations(id, title, title_auto, model, created_at, updated_at) "
                "VALUES (?, ?, 1, ?, ?, ?)",
                (cid, title or "New chat", model, now, now),
            )
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
        return self._conversation(row)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return self._conversation(row) if row else None

    def list_conversations(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
                "FROM conversations c ORDER BY c.pinned DESC, c.updated_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            item = self._conversation(row)
            item["message_count"] = row["message_count"]
            result.append(item)
        return result

    def update_conversation(self, conversation_id: str, touch: bool = True, **fields: Any) -> dict[str, Any] | None:
        allowed = {"title", "title_auto", "model", "summary", "summary_through_seq", "pinned"}
        sets: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key in ("title_auto", "pinned"):
                value = 1 if value else 0
            sets.append(f"{key} = ?")
            values.append(value)
        if touch:
            sets.append("updated_at = ?")
            values.append(config.now_iso())
        if not sets:
            return self.get_conversation(conversation_id)
        values.append(conversation_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?", values)
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return self._conversation(row) if row else None

    def touch_conversation(self, conversation_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (config.now_iso(), conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cursor.rowcount > 0

    def clear_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        now = config.now_iso()
        with self.connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute(
                "UPDATE conversations SET summary = '', summary_through_seq = 0, updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return self._conversation(row) if row else None

    def conversation_message_count(self, conversation_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
        return int(row["n"])

    # --- messages ----------------------------------------------------------

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "seq": row["seq"],
            "role": row["role"],
            "content": row["content"] or "",
            "reasoning": row["reasoning"] or "",
            "tool_calls": _json_loads(row["tool_calls"], None),
            "tool_call_id": row["tool_call_id"],
            "tool_name": row["tool_name"],
            "model": row["model"],
            "tokens": row["tokens"],
            "meta": _json_loads(row["meta"], {}),
            "created_at": row["created_at"],
        }

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str = "",
        *,
        reasoning: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        model: str | None = None,
        tokens: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = config.now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS seq FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                seq = int(row["seq"]) + 1
                mid = _new_id("m")
                conn.execute(
                    "INSERT INTO messages(id, conversation_id, seq, role, content, reasoning, "
                    "tool_calls, tool_call_id, tool_name, model, tokens, meta, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        mid,
                        conversation_id,
                        seq,
                        role,
                        content or "",
                        reasoning or "",
                        _json_dumps(tool_calls),
                        tool_call_id,
                        tool_name,
                        model,
                        tokens,
                        _json_dumps(meta),
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone()
        return self._message(row)

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return self._message(row) if row else None

    def list_messages(
        self, conversation_id: str, after_seq: int = 0, limit: int | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE conversation_id = ? AND seq > ? ORDER BY seq ASC"
        params: list[Any] = [conversation_id, after_seq]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._message(row) for row in rows]

    def recent_messages(self, conversation_id: str, limit: int = 8) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [self._message(row) for row in reversed(rows)]

    def update_message(self, message_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {"content", "reasoning", "tokens", "meta", "tool_calls", "tool_name"}
        sets: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key in ("meta", "tool_calls"):
                value = _json_dumps(value)
            sets.append(f"{key} = ?")
            values.append(value)
        if not sets:
            return self.get_message(message_id)
        values.append(message_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE messages SET {', '.join(sets)} WHERE id = ?", values)
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return self._message(row) if row else None

    # --- memories ----------------------------------------------------------

    @staticmethod
    def _memory(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "content": row["content"],
            "normalized": row["normalized"],
            "tags": _json_loads(row["tags"], []) or [],
            "source_conversation_id": row["source_conversation_id"],
            "confidence": float(row["confidence"]),
            "pinned": bool(row["pinned"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_used_at": row["last_used_at"],
        }

    @staticmethod
    def _fts_insert(conn: sqlite3.Connection, memory_id: str, content: str, normalized: str) -> None:
        conn.execute(
            "INSERT INTO memories_fts(memory_id, content, normalized) VALUES (?, ?, ?)",
            (memory_id, content, normalized),
        )

    @staticmethod
    def _fts_delete(conn: sqlite3.Connection, memory_id: str) -> None:
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))

    def create_memory(
        self,
        kind: str,
        content: str,
        normalized: str,
        *,
        tags: list[str] | None = None,
        source_conversation_id: str | None = None,
        confidence: float = 1.0,
        pinned: bool = False,
    ) -> dict[str, Any]:
        now = config.now_iso()
        mid = _new_id("mem")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO memories(id, kind, content, normalized, tags, source_conversation_id, "
                    "confidence, pinned, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        mid,
                        kind,
                        content,
                        normalized,
                        _json_dumps(tags or []),
                        source_conversation_id,
                        float(confidence),
                        1 if pinned else 0,
                        now,
                        now,
                    ),
                )
                self._fts_insert(conn, mid, content, normalized)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        return self._memory(row)

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._memory(row) if row else None

    def list_memories(
        self,
        *,
        kind: str | None = None,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT m.* FROM memories m"
        params: list[Any] = []
        match = fts_query(query) if query else None
        if match:
            sql += " JOIN memories_fts ON memories_fts.memory_id = m.id WHERE memories_fts MATCH ?"
            params.append(match)
        elif query:
            sql += " WHERE (m.content LIKE ? OR m.normalized LIKE ?)"
            like = f"%{query}%"
            params.extend([like, like])
        else:
            sql += " WHERE 1=1"
        if kind:
            sql += " AND m.kind = ?"
            params.append(kind)
        sql += " ORDER BY m.pinned DESC, m.updated_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        try:
            with self.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            log.warning("memory search failed (%s); using LIKE fallback", exc)
            return self.list_memories(kind=kind, query=None, limit=limit)
        return [self._memory(row) for row in rows]

    def search_memories(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        return self.list_memories(query=query, limit=limit)

    def count_memories(self, kind: str | None = None, query: str | None = None) -> int:
        return len(self.list_memories(kind=kind, query=query))

    def all_memories(self, limit: int = config.MEMORY_CAP) -> list[dict[str, Any]]:
        return self.list_memories(limit=limit)

    def pinned_memories(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE pinned = 1 ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._memory(row) for row in rows]

    def recently_used_memories(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE last_used_at IS NOT NULL "
                "ORDER BY last_used_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._memory(row) for row in rows]

    def update_memory(self, memory_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {"kind", "content", "normalized", "tags", "confidence", "pinned"}
        sets: list[str] = []
        values: list[Any] = []
        content_changed = False
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "tags":
                value = _json_dumps(value)
            if key == "pinned":
                value = 1 if value else 0
            if key == "confidence":
                value = float(value)
            if key == "content":
                content_changed = True
            sets.append(f"{key} = ?")
            values.append(value)
        if not sets:
            return self.get_memory(memory_id)
        sets.append("updated_at = ?")
        values.append(config.now_iso())
        values.append(memory_id)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", values)
                row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
                if row is not None and content_changed:
                    self._fts_delete(conn, memory_id)
                    self._fts_insert(conn, memory_id, row["content"], row["normalized"])
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self._memory(row) if row else None

    def delete_memory(self, memory_id: str) -> bool:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                self._fts_delete(conn, memory_id)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return cursor.rowcount > 0

    def delete_all_memories(self) -> int:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
                conn.execute("DELETE FROM memories")
                conn.execute("DELETE FROM memories_fts")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return int(row["n"])

    def mark_memories_used(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = config.now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(
                    "UPDATE memories SET last_used_at = ? WHERE id = ?",
                    [(now, mid) for mid in memory_ids],
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def prune_memories(self, cap: int = config.MEMORY_CAP) -> int:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                total = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
                excess = int(total) - cap
                if excess <= 0:
                    conn.execute("COMMIT")
                    return 0
                rows = conn.execute(
                    "SELECT id FROM memories WHERE pinned = 0 "
                    "ORDER BY confidence ASC, updated_at ASC LIMIT ?",
                    (excess,),
                ).fetchall()
                ids = [row["id"] for row in rows]
                for mid in ids:
                    conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
                    self._fts_delete(conn, mid)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return len(ids)

    # --- permissions & settings -------------------------------------------

    def get_permissions(self) -> dict[str, str]:
        merged = dict(config.DEFAULT_PERMISSIONS)
        with self.connect() as conn:
            rows = conn.execute("SELECT tool, mode FROM permissions").fetchall()
        for row in rows:
            if row["tool"] in merged and row["mode"] in config.PERMISSION_MODES:
                merged[row["tool"]] = row["mode"]
        return merged

    def set_permission(self, tool: str, mode: str) -> None:
        if tool not in config.PERMISSION_TOOLS or mode not in config.PERMISSION_MODES:
            raise ValueError("invalid permission")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO permissions(tool, mode) VALUES (?, ?) "
                "ON CONFLICT(tool) DO UPDATE SET mode = excluded.mode",
                (tool, mode),
            )

    def set_permissions(self, permissions: dict[str, str]) -> None:
        for tool, mode in permissions.items():
            self.set_permission(tool, mode)

    def get_settings(self, defaults: dict[str, Any]) -> dict[str, Any]:
        merged = dict(defaults)
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        for row in rows:
            try:
                value = json.loads(row["value"])
            except (TypeError, ValueError):
                continue
            merged[row["key"]] = value
        merged["permissions"] = self.get_permissions()
        return merged

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def patch_settings(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if key == "permissions":
                self.set_permissions(value)
            else:
                self.set_setting(key, value)

    # --- usage -------------------------------------------------------------

    def add_usage(
        self,
        conversation_id: str | None,
        model: str | None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO usage(conversation_id, model, prompt_tokens, completion_tokens, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, model, int(prompt_tokens), int(completion_tokens), config.now_iso()),
            )

    def usage_totals(self) -> dict[str, int]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS turns, COALESCE(SUM(prompt_tokens), 0) AS prompt, "
                "COALESCE(SUM(completion_tokens), 0) AS completion FROM usage"
            ).fetchone()
        return {
            "turns": int(row["turns"]),
            "prompt_tokens": int(row["prompt"]),
            "completion_tokens": int(row["completion"]),
        }
