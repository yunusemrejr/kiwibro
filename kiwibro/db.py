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
from typing import Any, Iterable, Iterator

from . import config, terms

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

def _migrate_v2(conn: sqlite3.Connection) -> None:
    """Schema v2: episodic recall + segmented summaries + token cache.

    - ``messages_fts`` indexes every message's content so compacted history
      stays searchable (episodic recall) instead of being lost to the prompt.
    - ``summary_segments`` stores one bounded summary per seq range, so
      summarisation never recurses into summary-of-summary decay.
    - ``messages.est_tokens`` caches per-message prompt cost for O(window)
      context planning; ``conversations.extracted_through_seq`` is the
      memory-extraction watermark; ``memories.use_count`` feeds ranking.
    Column adds are guarded so a crash mid-migration can safely re-run.
    """

    def has_column(table: str, column: str) -> bool:
        return any(row["name"] == column for row in conn.execute(f"PRAGMA table_info({table})"))

    if not has_column("messages", "est_tokens"):
        conn.execute("ALTER TABLE messages ADD COLUMN est_tokens INTEGER")
    if not has_column("conversations", "extracted_through_seq"):
        conn.execute(
            "ALTER TABLE conversations ADD COLUMN extracted_through_seq INTEGER NOT NULL DEFAULT 0"
        )
    if not has_column("memories", "use_count"):
        conn.execute("ALTER TABLE memories ADD COLUMN use_count INTEGER NOT NULL DEFAULT 0")
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            message_id UNINDEXED, conversation_id UNINDEXED, content, tokenize='unicode61'
        );
        CREATE TABLE IF NOT EXISTS summary_segments (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            from_seq INTEGER NOT NULL,
            through_seq INTEGER NOT NULL,
            summary TEXT NOT NULL,
            est_tokens INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_summary_segments_conv
            ON summary_segments(conversation_id, through_seq);
        """
    )
    # Backfill the FTS index once (guarded: only when it is still empty).
    indexed = conn.execute("SELECT COUNT(*) AS n FROM messages_fts").fetchone()["n"]
    if not indexed:
        conn.execute(
            "INSERT INTO messages_fts(message_id, conversation_id, content) "
            "SELECT id, conversation_id, content FROM messages WHERE content <> ''"
        )


def _migrate_v3(conn: sqlite3.Connection) -> None:
    """Schema v3: a conversation's own working folder (NULL = the Settings workspace)."""
    if not any(row["name"] == "workspace" for row in conn.execute("PRAGMA table_info(conversations)")):
        conn.execute("ALTER TABLE conversations ADD COLUMN workspace TEXT")


def _migrate_v4(conn: sqlite3.Connection) -> None:
    """Schema v4: where a memory came from — ``you`` (added or edited in the Memory
    sheet), ``chat`` (saved by the model when asked in a conversation) or ``learned``
    (background extraction). NULL for memories from older versions."""
    if not any(row["name"] == "origin" for row in conn.execute("PRAGMA table_info(memories)")):
        conn.execute("ALTER TABLE memories ADD COLUMN origin TEXT")


def _migrate_v5(conn: sqlite3.Connection) -> None:
    """Schema v5: a private chat — nothing from it is remembered or shown to other chats."""
    if not any(row["name"] == "private" for row in conn.execute("PRAGMA table_info(conversations)")):
        conn.execute("ALTER TABLE conversations ADD COLUMN private INTEGER NOT NULL DEFAULT 0")


def _migrate_v6(conn: sqlite3.Connection) -> None:
    """Schema v6: notes — longer pages the user or Kiwibro writes (title, markdown body, tags)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notes ("
        "id TEXT PRIMARY KEY, "
        "title TEXT NOT NULL DEFAULT '', "
        "body TEXT NOT NULL DEFAULT '', "
        "tags TEXT NOT NULL DEFAULT '[]', "
        "pinned INTEGER NOT NULL DEFAULT 0, "
        "origin TEXT NOT NULL DEFAULT 'you', "
        "source_conversation_id TEXT, "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_order ON notes(pinned DESC, updated_at DESC)")
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5("
        "note_id UNINDEXED, title, body, tags, tokenize='unicode61')"
    )


def _migrate_v7(conn: sqlite3.Connection) -> None:
    """Schema v7: settings of the retired external-server engine are dropped (Kiwibro runs
    llama.cpp itself now)."""
    conn.execute(
        f"DELETE FROM settings WHERE key IN ({', '.join('?' for _ in config.RETIRED_SETTINGS)})",
        tuple(sorted(config.RETIRED_SETTINGS)),
    )


# A migration is either a SQL script or a callable taking the connection
# (for steps that must inspect the live schema, e.g. guarded ALTERs).
def _migrate_v8(conn: sqlite3.Connection) -> None:
    """Schema v8: per-model behaviour traits (what Kiwi Sense had to correct, and how often)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS model_traits ("
        "model TEXT NOT NULL, kind TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0, "
        "last_at TEXT NOT NULL, PRIMARY KEY (model, kind))"
    )


def _migrate_v9(conn: sqlite3.Connection) -> None:
    """Schema v9: the session ledger — one row per step saying what was asked, which file
    changed, which command ran and which page was read. Written by code, so a long
    conversation keeps an exact record of itself after the prompt has moved on."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ledger ("
        "id TEXT PRIMARY KEY, "
        "conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, "
        "seq INTEGER NOT NULL, "
        "kind TEXT NOT NULL, "
        "key TEXT NOT NULL DEFAULT '', "
        "detail TEXT NOT NULL DEFAULT '', "
        "created_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_conversation ON ledger(conversation_id, kind)")


def _migrate_v10(conn: sqlite3.Connection) -> None:
    """Schema v10: the safe copy session of a conversation (JSON: session_id, real,
    previous_workspace). NULL means the chat works in the ordinary workspace."""
    if not any(row["name"] == "safe_copy" for row in conn.execute("PRAGMA table_info(conversations)")):
        conn.execute("ALTER TABLE conversations ADD COLUMN safe_copy TEXT")


_MIGRATIONS: list[tuple[int, Any]] = [
    (1, _SCHEMA_V1), (2, _migrate_v2), (3, _migrate_v3), (4, _migrate_v4), (5, _migrate_v5),
    (6, _migrate_v6), (7, _migrate_v7), (8, _migrate_v8), (9, _migrate_v9), (10, _migrate_v10),
]

MEMORY_ORIGINS = ("you", "chat", "learned")
NOTE_ORIGINS = ("you", "kiwibro")


def _est_tokens(text: str | None) -> int:
    """Per-message token estimate stored at write time.

    Mirrors ``kiwibro.context.estimate_tokens`` (~3.6 ASCII chars/token,
    ~0.9 tokens/non-ASCII char); duplicated here to avoid a db→context
    import cycle. A test asserts the two stay in agreement.
    """
    import math

    if not text:
        return 0
    if text.isascii():
        return max(1, math.ceil(len(text) / 3.6))
    ascii_chars = len(text.encode("ascii", "ignore"))
    return max(1, math.ceil(ascii_chars / 3.6 + (len(text) - ascii_chars) * 0.9))


def _message_est_tokens(content: str, tool_calls_json: str | None) -> int:
    return _est_tokens(content) + _est_tokens(tool_calls_json) + 4


def fts_query(query: str) -> str | None:
    """Turn a user query into a safe FTS5 MATCH expression (OR of quoted tokens).

    Function words are dropped when the query has any other word: in an OR match "the"
    or "is" makes nearly every message a hit, which filled recall with noise."""
    tokens = [t for t in re.findall(r"[^\W_]+", query or "", flags=re.UNICODE) if len(t) >= 1]
    meaningful = [t for t in tokens if t.lower() not in terms.STOPWORDS]
    tokens = meaningful or tokens
    if not tokens:
        return None
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    quoted = [_fts_term(token) for token in seen[:12]]
    return " OR ".join(quoted)


def _fts_term(token: str) -> str:
    """A quoted FTS5 term; words of 4+ letters also match as a prefix of their singular
    stem, so "theme" finds "themes" and "themes" finds "theme"."""
    if len(token) < 4:
        return '"' + token.replace('"', '""') + '"'
    lowered = token.lower()
    stem = token[:-1] if lowered.endswith("s") and not lowered.endswith("ss") else token
    return '"' + stem.replace('"', '""') + '"*'


class Database:
    """Thread-safe (per-call connections) SQLite wrapper."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._init_lock = threading.Lock()
        self._initialized = False
        self._local = threading.local()
        self._settings_version = 0
        self._settings_version_lock = threading.Lock()

    @property
    def settings_version(self) -> int:
        """Monotonic counter bumped on every settings/permissions write."""
        return self._settings_version

    def _bump_settings_version(self) -> None:
        with self._settings_version_lock:
            self._settings_version += 1

    def _thread_connection(self) -> sqlite3.Connection:
        conns = getattr(self._local, "conns", None)
        if conns is None:
            conns = self._local.conns = {}
        conn = conns.get(self.path)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except sqlite3.Error:
                conns.pop(self.path, None)
        conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conns[self.path] = conn
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        # One reused connection per thread (SQLite handles concurrency via WAL +
        # busy_timeout); avoids a connect + pragma round-trip on every call.
        conn = self._thread_connection()
        try:
            yield conn
        finally:
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass

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
                for version, migration in _MIGRATIONS:
                    if version in applied:
                        continue
                    if callable(migration):
                        migration(conn)
                    else:
                        conn.executescript(migration)
                    conn.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (version, config.now_iso()),
                    )
            self._initialized = True

    # --- conversations -----------------------------------------------------

    @staticmethod
    def _conversation(row: sqlite3.Row) -> dict[str, Any]:
        keys = row.keys()
        return {
            "id": row["id"],
            "title": row["title"],
            "title_auto": bool(row["title_auto"]),
            "model": row["model"],
            "summary": row["summary"] or "",
            "summary_through_seq": row["summary_through_seq"],
            "extracted_through_seq": row["extracted_through_seq"] if "extracted_through_seq" in keys else 0,
            "workspace": row["workspace"] if "workspace" in keys else None,
            "safe_copy": row["safe_copy"] if "safe_copy" in keys else None,
            "pinned": bool(row["pinned"]),
            "private": bool(row["private"]) if "private" in keys else False,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_conversation(
        self, title: str = "New chat", model: str | None = None, *, private: bool = False
    ) -> dict[str, Any]:
        now = config.now_iso()
        cid = _new_id("c")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO conversations(id, title, title_auto, model, private, created_at, updated_at) "
                "VALUES (?, ?, 1, ?, ?, ?, ?)",
                (cid, title or "New chat", model, 1 if private else 0, now, now),
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

    def recent_conversations(self, limit: int = 6) -> list[dict[str, Any]]:
        """The most recently active conversations that have messages, newest first."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.* FROM conversations c WHERE EXISTS "
                "(SELECT 1 FROM messages m WHERE m.conversation_id = c.id) "
                "ORDER BY c.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._conversation(row) for row in rows]

    def update_conversation(self, conversation_id: str, touch: bool = True, **fields: Any) -> dict[str, Any] | None:
        allowed = {
            "title", "title_auto", "model", "summary", "summary_through_seq",
            "extracted_through_seq", "pinned", "workspace", "private", "safe_copy",
        }
        sets: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key in ("title_auto", "pinned", "private"):
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
            conn.execute("BEGIN IMMEDIATE")
            try:
                # messages/summary_segments cascade; the FTS index has no FK.
                conn.execute("DELETE FROM messages_fts WHERE conversation_id = ?", (conversation_id,))
                cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return cursor.rowcount > 0

    def clear_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        now = config.now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                conn.execute("DELETE FROM messages_fts WHERE conversation_id = ?", (conversation_id,))
                conn.execute("DELETE FROM summary_segments WHERE conversation_id = ?", (conversation_id,))
                conn.execute("DELETE FROM ledger WHERE conversation_id = ?", (conversation_id,))
                conn.execute(
                    "UPDATE conversations SET summary = '', summary_through_seq = 0, "
                    "extracted_through_seq = 0, updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return self._conversation(row) if row else None

    def fork_conversation(
        self, source_id: str, *, through_seq: int | None = None, title: str | None = None
    ) -> dict[str, Any] | None:
        """Copy a conversation's messages up to ``through_seq`` (default all) into a new one, atomically."""
        now = config.now_iso()
        new_id = _new_id("c")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                source = conn.execute("SELECT * FROM conversations WHERE id = ?", (source_id,)).fetchone()
                if source is None:
                    conn.execute("ROLLBACK")
                    return None
                last_seq = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS seq FROM messages WHERE conversation_id = ?",
                    (source_id,),
                ).fetchone()["seq"]
                cut = int(last_seq if through_seq is None else through_seq)
                summary_through = int(source["summary_through_seq"])
                compacted = summary_through <= cut
                conn.execute(
                    "INSERT INTO conversations(id, title, title_auto, model, summary, summary_through_seq, "
                    "extracted_through_seq, workspace, private, created_at, updated_at) "
                    "VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        new_id,
                        title or f"{source['title']} (fork)",
                        source["model"],
                        (source["summary"] or "") if compacted else "",
                        summary_through if compacted else 0,
                        min(int(source["extracted_through_seq"]), cut),
                        source["workspace"],
                        source["private"],
                        now,
                        now,
                    ),
                )
                rows = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? AND seq <= ? ORDER BY seq ASC",
                    (source_id, cut),
                ).fetchall()
                message_rows: list[tuple[Any, ...]] = []
                fts_rows: list[tuple[str, str, str]] = []
                for row in rows:
                    mid = _new_id("m")
                    message_rows.append((
                        mid, new_id, row["seq"], row["role"], row["content"], row["reasoning"],
                        row["tool_calls"], row["tool_call_id"], row["tool_name"], row["model"],
                        row["tokens"], row["est_tokens"], row["meta"], row["created_at"],
                    ))
                    if (row["content"] or "").strip():
                        fts_rows.append((mid, new_id, row["content"]))
                conn.executemany(
                    "INSERT INTO messages(id, conversation_id, seq, role, content, reasoning, "
                    "tool_calls, tool_call_id, tool_name, model, tokens, est_tokens, meta, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    message_rows,
                )
                conn.executemany(
                    "INSERT INTO messages_fts(message_id, conversation_id, content) VALUES (?, ?, ?)",
                    fts_rows,
                )
                if compacted:
                    segments = conn.execute(
                        "SELECT * FROM summary_segments WHERE conversation_id = ? AND through_seq <= ? "
                        "ORDER BY from_seq ASC",
                        (source_id, cut),
                    ).fetchall()
                    conn.executemany(
                        "INSERT INTO summary_segments(id, conversation_id, from_seq, through_seq, "
                        "summary, est_tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [
                            (
                                _new_id("seg"), new_id, seg["from_seq"], seg["through_seq"],
                                seg["summary"], seg["est_tokens"], seg["created_at"],
                            )
                            for seg in segments
                        ],
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self.get_conversation(new_id)

    def conversation_message_count(self, conversation_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
        return int(row["n"])

    # --- messages ----------------------------------------------------------

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        keys = row.keys()
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
            "est_tokens": row["est_tokens"] if "est_tokens" in keys else None,
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
                tool_calls_json = _json_dumps(tool_calls)
                conn.execute(
                    "INSERT INTO messages(id, conversation_id, seq, role, content, reasoning, "
                    "tool_calls, tool_call_id, tool_name, model, tokens, est_tokens, meta, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        mid,
                        conversation_id,
                        seq,
                        role,
                        content or "",
                        reasoning or "",
                        tool_calls_json,
                        tool_call_id,
                        tool_name,
                        model,
                        tokens,
                        _message_est_tokens(content or "", tool_calls_json),
                        _json_dumps(meta),
                        now,
                    ),
                )
                if (content or "").strip():
                    conn.execute(
                        "INSERT INTO messages_fts(message_id, conversation_id, content) VALUES (?, ?, ?)",
                        (mid, conversation_id, content or ""),
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

    def list_messages_page(
        self, conversation_id: str, *, before_seq: int | None = None, limit: int = 120
    ) -> tuple[list[dict[str, Any]], bool]:
        """Newest ``limit`` messages before ``before_seq`` (keyset pagination).

        Returns (messages oldest→newest, has_more_older). Long conversations
        page in from the tail instead of shipping megabytes per open.
        """
        sql = "SELECT * FROM messages WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if before_seq is not None:
            sql += " AND seq < ?"
            params.append(before_seq)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(int(limit) + 1)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        has_more = len(rows) > limit
        page = [self._message(row) for row in reversed(rows[:limit])]
        return page, has_more

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
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(f"UPDATE messages SET {', '.join(sets)} WHERE id = ?", values)
                row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
                if row is not None and ("content" in fields or "tool_calls" in fields):
                    # Keep the token cache and episodic index in sync.
                    conn.execute(
                        "UPDATE messages SET est_tokens = ? WHERE id = ?",
                        (_message_est_tokens(row["content"] or "", row["tool_calls"]), message_id),
                    )
                    conn.execute("DELETE FROM messages_fts WHERE message_id = ?", (message_id,))
                    if (row["content"] or "").strip():
                        conn.execute(
                            "INSERT INTO messages_fts(message_id, conversation_id, content) VALUES (?, ?, ?)",
                            (message_id, row["conversation_id"], row["content"]),
                        )
                    row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self._message(row) if row else None

    # --- episodic search & planning (schema v2) ----------------------------

    def search_messages(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        limit: int = 8,
        before_seq: int | None = None,
        after_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        """bm25-ranked full-text search over message content (episodic recall).

        This is the retrieval core that lets compacted history come back as
        exact evidence: messages are never deleted by compaction, and this
        finds the relevant ones again in milliseconds via FTS5.
        """
        match = fts_query(query)
        if not match:
            return []
        sql = (
            "SELECT m.* FROM messages m "
            "JOIN messages_fts ON messages_fts.message_id = m.id "
            "WHERE messages_fts MATCH ?"
        )
        params: list[Any] = [match]
        if conversation_id:
            sql += " AND m.conversation_id = ?"
            params.append(conversation_id)
        if before_seq is not None:
            sql += " AND m.seq < ?"
            params.append(before_seq)
        if after_seq is not None:
            sql += " AND m.seq > ?"
            params.append(after_seq)
        sql += " ORDER BY bm25(messages_fts), m.seq DESC LIMIT ?"
        params.append(limit)
        try:
            with self.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive (pre-v2 DB)
            log.warning("episodic search unavailable: %s", exc)
            return []
        return [self._message(row) for row in rows]

    def search_conversations(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        """Chats whose messages match ``query``, best match first, with a short snippet of it."""
        match = fts_query(query)
        if not match:
            return []
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    "SELECT conversation_id, snippet(messages_fts, 2, '', '', '…', 12) AS snippet "
                    "FROM messages_fts WHERE messages_fts MATCH ? ORDER BY bm25(messages_fts) LIMIT 400",
                    (match,),
                ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive (pre-v2 DB)
            log.warning("conversation search unavailable: %s", exc)
            return []
        found: dict[str, str] = {}
        for row in rows:
            if row["conversation_id"] not in found:
                snippet = re.sub(r"\*\*|__|`+|(?<!\S)(?:-{3,}|#{1,6})(?!\S)", " ", row["snippet"] or "")
                found[row["conversation_id"]] = " ".join(snippet.split())
                if len(found) >= limit:
                    break
        return [{"id": cid, "snippet": snippet} for cid, snippet in found.items()]

    def message_token_rows(
        self, conversation_id: str, after_seq: int = 0
    ) -> list[dict[str, Any]]:
        """Lightweight (seq, role, est_tokens, has_tool_calls) rows for planning.

        Context planning walks these instead of loading full message bodies,
        so its cost stays tiny no matter how long the conversation grows.
        Rows written before schema v2 have ``est_tokens = NULL``; callers
        fall back to estimating those from content length.
        """
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT seq, role, est_tokens, (tool_calls IS NOT NULL) AS has_tool_calls, "
                "LENGTH(content) AS content_chars, LENGTH(COALESCE(tool_calls, '')) AS tool_chars "
                "FROM messages WHERE conversation_id = ? AND seq > ? ORDER BY seq ASC",
                (conversation_id, after_seq),
            ).fetchall()
        return [
            {
                "seq": row["seq"],
                "role": row["role"],
                "est_tokens": row["est_tokens"],
                "has_tool_calls": bool(row["has_tool_calls"]),
                "content_chars": row["content_chars"],
                "tool_chars": row["tool_chars"],
            }
            for row in rows
        ]

    def list_messages_range(
        self, conversation_id: str, from_seq: int, through_seq: int
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? AND seq >= ? AND seq <= ? "
                "ORDER BY seq ASC",
                (conversation_id, from_seq, through_seq),
            ).fetchall()
        return [self._message(row) for row in rows]

    def last_message_seq(self, conversation_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["seq"])

    # --- summary segments (schema v2) ---------------------------------------

    @staticmethod
    def _segment(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "from_seq": row["from_seq"],
            "through_seq": row["through_seq"],
            "summary": row["summary"],
            "est_tokens": row["est_tokens"],
            "created_at": row["created_at"],
        }

    def add_summary_segment(
        self,
        conversation_id: str,
        from_seq: int,
        through_seq: int,
        summary: str,
        est_tokens: int | None = None,
    ) -> dict[str, Any]:
        sid = _new_id("seg")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO summary_segments(id, conversation_id, from_seq, through_seq, "
                "summary, est_tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sid, conversation_id, from_seq, through_seq, summary, est_tokens, config.now_iso()),
            )
            row = conn.execute("SELECT * FROM summary_segments WHERE id = ?", (sid,)).fetchone()
        return self._segment(row)

    def list_summary_segments(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM summary_segments WHERE conversation_id = ? ORDER BY from_seq ASC",
                (conversation_id,),
            ).fetchall()
        return [self._segment(row) for row in rows]

    def segments_through_seq(self, conversation_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(through_seq), 0) AS seq FROM summary_segments "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["seq"])

    # --- ledger (schema v9) -------------------------------------------------

    @staticmethod
    def _ledger_entry(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "seq": row["seq"],
            "kind": row["kind"],
            "key": row["key"],
            "detail": row["detail"],
            "created_at": row["created_at"],
        }

    def add_ledger_entry(
        self, conversation_id: str, seq: int, kind: str, key: str, detail: str = ""
    ) -> dict[str, Any]:
        """One ledger row for step ``seq``; the row id is what a later delete works from."""
        rid = _new_id("led")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO ledger(id, conversation_id, seq, kind, key, detail, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rid, conversation_id, int(seq), kind, key, detail, config.now_iso()),
            )
            row = conn.execute("SELECT * FROM ledger WHERE id = ?", (rid,)).fetchone()
        return self._ledger_entry(row)

    def list_ledger(
        self, conversation_id: str, through_seq: int | None = None
    ) -> list[dict[str, Any]]:
        """A conversation's ledger rows oldest step first; ``through_seq`` keeps only the
        steps that have scrolled out of the prompt."""
        sql = "SELECT * FROM ledger WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if through_seq is not None:
            sql += " AND seq <= ?"
            params.append(int(through_seq))
        sql += " ORDER BY seq ASC, rowid ASC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._ledger_entry(row) for row in rows]

    def delete_ledger(self, conversation_id: str) -> int:
        with self.connect() as conn:
            return conn.execute(
                "DELETE FROM ledger WHERE conversation_id = ?", (conversation_id,)
            ).rowcount

    # --- memories ----------------------------------------------------------

    @staticmethod
    def _memory(row: sqlite3.Row) -> dict[str, Any]:
        keys = row.keys()
        return {
            "id": row["id"],
            "kind": row["kind"],
            "content": row["content"],
            "normalized": row["normalized"],
            "tags": _json_loads(row["tags"], []) or [],
            "source_conversation_id": row["source_conversation_id"],
            "confidence": float(row["confidence"]),
            "pinned": bool(row["pinned"]),
            "use_count": int(row["use_count"]) if "use_count" in keys and row["use_count"] is not None else 0,
            "origin": row["origin"] if "origin" in keys else None,
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
        origin: str | None = None,
    ) -> dict[str, Any]:
        now = config.now_iso()
        mid = _new_id("mem")
        origin = origin if origin in MEMORY_ORIGINS else None
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO memories(id, kind, content, normalized, tags, source_conversation_id, "
                    "confidence, pinned, created_at, updated_at, origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        origin,
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
        if match:
            # Relevance first (bm25: lower is better), pinned still leads.
            sql += " ORDER BY m.pinned DESC, bm25(memories_fts), m.updated_at DESC"
        else:
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
        sql = "SELECT COUNT(*) AS n FROM memories m"
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
        try:
            with self.connect() as conn:
                row = conn.execute(sql, params).fetchone()
        except sqlite3.Error:  # pragma: no cover - defensive, mirrors list_memories
            return self.count_memories(kind=kind, query=None)
        return int(row["n"])

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
        allowed = {"kind", "content", "normalized", "tags", "confidence", "pinned", "origin"}
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
                    "UPDATE memories SET last_used_at = ?, use_count = use_count + 1 WHERE id = ?",
                    [(now, mid) for mid in memory_ids],
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def prune_memories(self, cap: int = config.MEMORY_CAP, keep: Iterable[str] = ()) -> int:
        """Delete the least valuable unpinned memories down to ``cap``; ``keep`` ids
        (just saved) are never chosen, or a new memory would evict itself at the cap."""
        keep = list(keep)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                total = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
                excess = int(total) - cap
                if excess <= 0:
                    conn.execute("COMMIT")
                    return 0
                exclude = f" AND id NOT IN ({','.join('?' * len(keep))})" if keep else ""
                rows = conn.execute(
                    # The user's own memories are never evicted to make room for learned ones.
                    "SELECT id FROM memories WHERE pinned = 0 AND COALESCE(origin, '') <> 'you'" + exclude + " "
                    "ORDER BY confidence ASC, use_count ASC, "
                    "COALESCE(last_used_at, updated_at) ASC LIMIT ?",
                    (*keep, excess),
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

    # --- notes -------------------------------------------------------------

    @staticmethod
    def _note(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "body": row["body"],
            "tags": _json_loads(row["tags"], []) or [],
            "pinned": bool(row["pinned"]),
            "origin": row["origin"],
            "source_conversation_id": row["source_conversation_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _note_fts_write(conn: sqlite3.Connection, note_id: str, title: str, body: str, tags: list[str]) -> None:
        conn.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
        conn.execute(
            "INSERT INTO notes_fts(note_id, title, body, tags) VALUES (?, ?, ?, ?)",
            (note_id, title, body, " ".join(tags)),
        )

    def create_note(
        self,
        title: str,
        body: str,
        tags: list[str] | None = None,
        *,
        origin: str = "you",
        source_conversation_id: str | None = None,
        pinned: bool = False,
    ) -> dict[str, Any]:
        if origin not in NOTE_ORIGINS:
            raise ValueError(f"unknown note origin: {origin}")
        now = config.now_iso()
        nid = _new_id("note")
        tags = list(tags or [])
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO notes(id, title, body, tags, pinned, origin, source_conversation_id, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (nid, title, body, _json_dumps(tags), 1 if pinned else 0, origin, source_conversation_id, now, now),
                )
                self._note_fts_write(conn, nid, title, body, tags)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (nid,)).fetchone()
        return self._note(row)

    def get_note(self, note_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return self._note(row) if row else None

    def find_note_by_title(self, title: str) -> dict[str, Any] | None:
        """Exact title match ignoring case; casefold so "İ" and "i" compare as SQLite's
        ASCII-only LOWER() would not. Titles are few, so the scan stays cheap."""
        wanted = " ".join((title or "").split()).casefold()
        if not wanted:
            return None
        with self.connect() as conn:
            rows = conn.execute("SELECT id, title FROM notes ORDER BY updated_at DESC").fetchall()
        for row in rows:
            if " ".join(row["title"].split()).casefold() == wanted:
                return self.get_note(row["id"])
        return None

    @staticmethod
    def _notes_from(query: str | None, tag: str | None) -> tuple[str, list[Any]]:
        """FROM/WHERE shared by list and count: FTS match (LIKE when the query has no
        words) and the tag filter."""
        sql = "FROM notes n"
        clauses: list[str] = []
        params: list[Any] = []
        match = fts_query(query) if query else None
        if match:
            sql += " JOIN notes_fts ON notes_fts.note_id = n.id"
            clauses.append("notes_fts MATCH ?")
            params.append(match)
        elif query:
            clauses.append("(n.title LIKE ? OR n.body LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like])
        tag = re.sub(r"[^\w-]+", "", (tag or "").strip().lower().lstrip("#"))  # as notes.clean_tags
        if tag:
            clauses.append("EXISTS (SELECT 1 FROM json_each(n.tags) AS j WHERE j.value = ?)")
            params.append(tag)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return sql, params

    def list_notes(
        self,
        query: str | None = None,
        tag: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql, params = self._notes_from(query, tag)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT n.* {sql} ORDER BY n.pinned DESC, n.updated_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [self._note(row) for row in rows]

    def count_notes(self, query: str | None = None, tag: str | None = None) -> int:
        sql, params = self._notes_from(query, tag)
        with self.connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS n {sql}", params).fetchone()
        return int(row["n"])

    def search_notes(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Best full-text matches first; title hits outweigh body hits, tags sit between."""
        match = fts_query(query)
        if not match:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT n.* FROM notes n JOIN notes_fts ON notes_fts.note_id = n.id "
                "WHERE notes_fts MATCH ? ORDER BY bm25(notes_fts, 0.0, 4.0, 1.0, 2.0), n.updated_at DESC LIMIT ?",
                (match, limit),
            ).fetchall()
        return [self._note(row) for row in rows]

    def update_note(self, note_id: str, **fields: Any) -> dict[str, Any] | None:
        unknown = set(fields) - {"title", "body", "tags", "pinned"}
        if unknown:
            raise ValueError(f"unknown note fields: {', '.join(sorted(unknown))}")
        if not fields:
            return self.get_note(note_id)
        sets: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key == "tags":
                value = _json_dumps(list(value))
            elif key == "pinned":
                value = 1 if value else 0
            sets.append(f"{key} = ?")
            values.append(value)
        sets.append("updated_at = ?")
        values.extend([config.now_iso(), note_id])
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(f"UPDATE notes SET {', '.join(sets)} WHERE id = ?", values)
                row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
                if row is not None and any(key in fields for key in ("title", "body", "tags")):
                    self._note_fts_write(
                        conn, note_id, row["title"], row["body"], _json_loads(row["tags"], []) or [],
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self._note(row) if row else None

    def delete_note(self, note_id: str) -> bool:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
                conn.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return cursor.rowcount > 0

    def note_tags(self) -> list[tuple[str, int]]:
        """Every tag with the number of notes carrying it; most used first, then A to Z."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT j.value AS tag, COUNT(*) AS uses FROM notes n, json_each(n.tags) AS j "
                "GROUP BY j.value ORDER BY uses DESC, tag ASC"
            ).fetchall()
        return [(row["tag"], int(row["uses"])) for row in rows]

    # --- permissions & settings -------------------------------------------

    def get_permissions(self) -> dict[str, str]:
        merged = config.permission_defaults()
        with self.connect() as conn:
            rows = conn.execute("SELECT tool, mode FROM permissions").fetchall()
        for row in rows:
            if row["tool"] in merged and row["mode"] in config.PERMISSION_MODES:
                merged[row["tool"]] = row["mode"]
        return merged

    def set_permission(self, tool: str, mode: str) -> None:
        if tool not in config.permission_defaults() or mode not in config.PERMISSION_MODES:
            raise ValueError("invalid permission")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO permissions(tool, mode) VALUES (?, ?) "
                "ON CONFLICT(tool) DO UPDATE SET mode = excluded.mode",
                (tool, mode),
            )
        self._bump_settings_version()

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

    # --- behaviour traits ---

    def note_trait(self, model: str, kind: str) -> int:
        """Count one more correction of ``kind`` for ``model``; returns the new count."""
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO model_traits(model, kind, count, last_at) VALUES (?, ?, 1, ?) "
                "ON CONFLICT(model, kind) DO UPDATE SET count = count + 1, last_at = excluded.last_at",
                (model, kind, config.now_iso()),
            )
            row = conn.execute(
                "SELECT count FROM model_traits WHERE model = ? AND kind = ?", (model, kind)
            ).fetchone()
        return int(row["count"]) if row else 0

    def traits(self, model: str, since: str | None = None) -> dict[str, int]:
        """{kind: count} for ``model``; with ``since`` (ISO time) only kinds seen at or after it."""
        query, values = "SELECT kind, count FROM model_traits WHERE model = ?", [model]
        if since:
            query += " AND last_at >= ?"
            values.append(since)
        with self.connect() as conn:
            return {row["kind"]: int(row["count"]) for row in conn.execute(query, values)}

    def all_traits(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT model, kind, count, last_at FROM model_traits ORDER BY model, kind")
            return [dict(row) for row in rows]

    def clear_traits(self, model: str | None = None) -> int:
        with self.connect() as conn:
            if model:
                return conn.execute("DELETE FROM model_traits WHERE model = ?", (model,)).rowcount
            return conn.execute("DELETE FROM model_traits").rowcount

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value, ensure_ascii=False)),
            )
        self._bump_settings_version()

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
