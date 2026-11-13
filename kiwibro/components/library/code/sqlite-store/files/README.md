# {{table}} store

A SQLite data layer for `{{table}}` records: one table, a schema version, typed
queries and no ORM. Rows are archived (`archived = 1`) instead of deleted.

    from store import Store
    store = Store("{{table}}.db")
    record = store.add("first note", "body text")
    store.list()              # unarchived records, oldest first
    store.search("note")      # title or body contains the text
    store.archive(record.id)  # hides it; returns True when it changed a row
    store.close()

Test with `python3 -m pytest -q`.
