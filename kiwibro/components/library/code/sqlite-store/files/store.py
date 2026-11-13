"""SQLite store for {{table}} records: schema, migrations and typed queries."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

TABLE = "{{table}}"
SCHEMA_VERSION = 1

if not TABLE.isidentifier():
    raise ValueError(f"table name {TABLE!r} is not a plain identifier")


@dataclass(frozen=True)
class Record:
    """One row of the table."""

    id: int
    title: str
    body: str
    created_at: str
    archived: bool


def _record(row: sqlite3.Row) -> Record:
    return Record(
        id=row["id"],
        title=row["title"],
        body=row["body"],
        created_at=row["created_at"],
        archived=bool(row["archived"]),
    )


class Store:
    """Open or create the database at path and bring its schema up to date."""

    def __init__(self, path: str | Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _migrate(self) -> None:
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"database is at schema version {version}, this code knows {SCHEMA_VERSION}"
            )
        # KIWIBRO: version 1 is your first schema; add later changes as version 2, 3, ...
        if version < 1:
            with self.connection:
                self.connection.execute(
                    f"""
                    CREATE TABLE {TABLE} (
                        id INTEGER PRIMARY KEY,
                        title TEXT NOT NULL CHECK (length(title) > 0),
                        body TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        archived INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                self.connection.execute(
                    f"CREATE INDEX {TABLE}_created_at ON {TABLE} (created_at)"
                )
                self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # KIWIBRO: replace these queries with the ones your records need
    def add(self, title: str, body: str = "") -> Record:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(body, str):
            raise TypeError(f"body must be str, got {type(body).__name__}")

        clean_title = title.strip()
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.connection:
            cursor = self.connection.execute(
                f"INSERT INTO {TABLE} (title, body, created_at) VALUES (?, ?, ?)",
                (clean_title, body, created_at),
            )
        return Record(cursor.lastrowid, clean_title, body, created_at, False)

    def get(self, record_id: int) -> Record | None:
        row = self.connection.execute(
            f"SELECT * FROM {TABLE} WHERE id = ?", (record_id,)
        ).fetchone()
        return None if row is None else _record(row)

    def list(self, include_archived: bool = False) -> list[Record]:
        sql = f"SELECT * FROM {TABLE}"
        if not include_archived:
            sql += " WHERE archived = 0"
        sql += " ORDER BY created_at, id"
        return [_record(row) for row in self.connection.execute(sql)]

    def search(self, text: str) -> list[Record]:
        if not text.strip():
            raise ValueError("search text must not be empty")
        pattern = f"%{text.strip()}%"
        rows = self.connection.execute(
            f"""
            SELECT * FROM {TABLE}
            WHERE archived = 0 AND (title LIKE ? OR body LIKE ?)
            ORDER BY created_at, id
            """,
            (pattern, pattern),
        )
        return [_record(row) for row in rows]

    def archive(self, record_id: int) -> bool:
        """Hide a record instead of deleting it; False when nothing changed."""
        with self.connection:
            cursor = self.connection.execute(
                f"UPDATE {TABLE} SET archived = 1 WHERE id = ? AND archived = 0",
                (record_id,),
            )
        return cursor.rowcount == 1
