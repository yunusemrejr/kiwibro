# {{name}}

A FastAPI service with typed models and SQLite storage for {{resource}}.

Run: `uvicorn app:create_app --factory --reload` (serves `http://127.0.0.1:8000`; docs at `/docs`)

Test: `pip install pytest httpx`, then `pytest`

Routes: `GET /health`, `GET /{{resource}}`, `POST /{{resource}}`, `GET /{{resource}}/<id>`, `PATCH /{{resource}}/<id>`.

There is no DELETE route. Send `{"archived": true}` with PATCH to hide an item; add `?include_archived=true` to list everything.

The database file defaults to `service.db`; set `DB_PATH` to change it. Invalid input returns FastAPI's 422 with field details.
