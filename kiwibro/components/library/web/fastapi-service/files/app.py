"""{{name}} — a typed JSON API backed by SQLite."""

import os
import sqlite3

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

NAME = "{{name}}"
RESOURCE = "{{resource}}"
if not RESOURCE.isidentifier():
    raise ValueError("resource must be a valid identifier, for example 'items'")

# KIWIBRO: replace these columns with the project's own fields
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {RESOURCE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class ItemCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    body: str = Field(default="", max_length=2000)

    @field_validator("title", "body")
    @classmethod
    def clean_text(cls, value: str, info: ValidationInfo) -> str:
        value = value.strip()
        if info.field_name == "title" and not value:
            raise ValueError("title is required")
        return value


class ItemUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=120)
    body: str | None = Field(default=None, max_length=2000)
    archived: bool | None = None

    @field_validator("title", "body")
    @classmethod
    def clean_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if info.field_name == "title" and not value:
            raise ValueError("title must not be blank")
        return value


class Item(BaseModel):
    id: int
    title: str
    body: str
    archived: bool
    created_at: str


def create_app(db_path: str = "service.db") -> FastAPI:
    app = FastAPI(title=NAME)

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def get_db():
        connection = connect()
        try:
            yield connection
        finally:
            connection.close()

    setup = connect()
    try:
        setup.execute(SCHEMA)
        setup.commit()
    finally:
        setup.close()

    def as_item(row: sqlite3.Row) -> Item:
        return Item(
            id=row["id"], title=row["title"], body=row["body"],
            archived=bool(row["archived"]), created_at=row["created_at"],
        )

    def find(db: sqlite3.Connection, item_id: int):
        return db.execute(f"SELECT * FROM {RESOURCE} WHERE id = ?", (item_id,)).fetchone()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "name": NAME}

    @app.get(f"/{RESOURCE}", response_model=list[Item])
    def list_items(include_archived: bool = False, db: sqlite3.Connection = Depends(get_db)) -> list[Item]:
        query = f"SELECT * FROM {RESOURCE}"
        if not include_archived:
            query += " WHERE archived = 0"
        query += " ORDER BY id DESC"
        return [as_item(row) for row in db.execute(query).fetchall()]

    @app.post(f"/{RESOURCE}", response_model=Item, status_code=201)
    def create_item(item: ItemCreate, db: sqlite3.Connection = Depends(get_db)) -> Item:
        cursor = db.execute(
            f"INSERT INTO {RESOURCE} (title, body) VALUES (?, ?)",
            (item.title, item.body),
        )
        db.commit()
        return as_item(find(db, cursor.lastrowid))

    @app.get("/" + RESOURCE + "/{item_id}", response_model=Item)
    def get_item(item_id: int, db: sqlite3.Connection = Depends(get_db)) -> Item:
        row = find(db, item_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Item not found.")
        return as_item(row)

    # KIWIBRO: archive instead of deleting — PATCH {"archived": true} hides an item.
    @app.patch("/" + RESOURCE + "/{item_id}", response_model=Item)
    def update_item(item_id: int, changes: ItemUpdate, db: sqlite3.Connection = Depends(get_db)) -> Item:
        if find(db, item_id) is None:
            raise HTTPException(status_code=404, detail="Item not found.")
        update = changes.model_dump(exclude_unset=True)
        if not update:
            raise HTTPException(400, detail="Send at least one of title, body or archived.")
        assignments = ", ".join(f"{field} = ?" for field in update)
        values = [int(v) if f == "archived" else v for f, v in update.items()]
        db.execute(
            f"UPDATE {RESOURCE} SET {assignments} WHERE id = ?",
            (*values, item_id),
        )
        db.commit()
        return as_item(find(db, item_id))

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(os.environ.get("DB_PATH", "service.db")))
