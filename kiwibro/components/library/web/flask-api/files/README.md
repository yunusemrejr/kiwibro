# {{name}}

A Flask JSON API with SQLite storage for {{resource}}.

Run: `python3 app.py` (serves `http://127.0.0.1:5000`)

Test: `pytest`

Routes: `GET /health`, `GET /{{resource}}`, `POST /{{resource}}`, `GET /{{resource}}/<id>`, `PATCH /{{resource}}/<id>`.

There is no DELETE route. Send `{"archived": true}` with PATCH to hide an item; add `?include_archived=1` to list everything.

The database file defaults to `app.db` next to `app.py`; set `DB_PATH` to change it. Every response is JSON; validation errors return 400 with an `error` field.
