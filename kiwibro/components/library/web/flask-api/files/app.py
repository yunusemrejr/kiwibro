"""{{name}} — a small JSON API backed by SQLite."""

import os
import sqlite3

from flask import Flask, g, jsonify, request

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

FIELDS = ("title", "body", "archived")


def create_app(db_path: str = "app.db") -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path

    def get_db() -> sqlite3.Connection:
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DB_PATH"])
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def close_db(_error=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    with app.app_context():
        db = get_db()
        db.execute(SCHEMA)
        db.commit()

    def as_json(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "body": row["body"],
            "archived": bool(row["archived"]),
            "created_at": row["created_at"],
        }

    def payload() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("Send a JSON object.")
        unknown = set(data) - set(FIELDS)
        if unknown:
            raise ValueError(f"Unknown field: {sorted(unknown)[0]}.")
        return data

    def text(data: dict, field: str, *, required: bool, limit: int) -> str:
        value = data.get(field, "")
        if not isinstance(value, str):
            raise ValueError(f"'{field}' must be text.")
        value = value.strip()
        if required and not value:
            raise ValueError(f"'{field}' is required.")
        if len(value) > limit:
            raise ValueError(f"'{field}' must be at most {limit} characters.")
        return value

    def find(db: sqlite3.Connection, item_id: int):
        return db.execute(f"SELECT * FROM {RESOURCE} WHERE id = ?", (item_id,)).fetchone()

    @app.get("/health")
    def health():
        return jsonify(status="ok", name=NAME)

    @app.get(f"/{RESOURCE}")
    def list_items():
        query = f"SELECT * FROM {RESOURCE}"
        if request.args.get("include_archived") != "1":
            query += " WHERE archived = 0"
        query += " ORDER BY id DESC"
        rows = get_db().execute(query).fetchall()
        return jsonify([as_json(row) for row in rows])

    @app.post(f"/{RESOURCE}")
    def create_item():
        try:
            data = payload()
            title = text(data, "title", required=True, limit=120)
            body = text(data, "body", required=False, limit=2000)
        except ValueError as problem:
            return jsonify(error=str(problem)), 400
        db = get_db()
        cursor = db.execute(
            f"INSERT INTO {RESOURCE} (title, body) VALUES (?, ?)", (title, body)
        )
        db.commit()
        return jsonify(as_json(find(db, cursor.lastrowid))), 201

    @app.get(f"/{RESOURCE}/<int:item_id>")
    def get_item(item_id: int):
        row = find(get_db(), item_id)
        if row is None:
            return jsonify(error="Item not found."), 404
        return jsonify(as_json(row))

    # KIWIBRO: archive instead of deleting — PATCH {"archived": true} hides an item.
    @app.patch(f"/{RESOURCE}/<int:item_id>")
    def update_item(item_id: int):
        db = get_db()
        row = find(db, item_id)
        if row is None:
            return jsonify(error="Item not found."), 404
        try:
            data = payload()
            changes = {}
            if "title" in data:
                changes["title"] = text(data, "title", required=True, limit=120)
            if "body" in data:
                changes["body"] = text(data, "body", required=False, limit=2000)
            if "archived" in data:
                if not isinstance(data["archived"], bool):
                    raise ValueError("'archived' must be true or false.")
                changes["archived"] = int(data["archived"])
            if not changes:
                raise ValueError("Send at least one of title, body or archived.")
        except ValueError as problem:
            return jsonify(error=str(problem)), 400
        assignments = ", ".join(f"{field} = ?" for field in changes)
        db.execute(
            f"UPDATE {RESOURCE} SET {assignments} WHERE id = ?",
            (*changes.values(), item_id),
        )
        db.commit()
        return jsonify(as_json(find(db, item_id)))

    return app


if __name__ == "__main__":
    create_app(os.environ.get("DB_PATH", "app.db")).run(debug=True)
