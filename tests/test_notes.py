"""Notes: schema v6, storage and FTS, cleaning rules, Notebook behaviour, HTTP API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import kiwibro.db as db_module
from kiwibro.db import Database
from kiwibro.notes import (
    MAX_BODY,
    MAX_TAG_CHARS,
    MAX_TAGS,
    MAX_TITLE,
    UNTITLED,
    Notebook,
    clean_body,
    clean_tags,
    clean_title,
    derive_title,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "notes.db")
    database.initialize()
    return database


@pytest.fixture
def notebook(db: Database) -> Notebook:
    return Notebook(db)


def _tables(db: Database) -> set[str]:
    with db.connect() as conn:
        return {row["name"] for row in conn.execute("SELECT name FROM sqlite_master")}


def _versions(db: Database) -> set[int]:
    with db.connect() as conn:
        return {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}


def test_migration_creates_notes_on_fresh_database(db: Database) -> None:
    assert {"notes", "notes_fts"} <= _tables(db)
    assert 6 in _versions(db)


def test_migration_adds_notes_to_database_at_version_five(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "old.db"
    full = list(db_module._MIGRATIONS)
    monkeypatch.setattr(db_module, "_MIGRATIONS", full[:5])
    old = Database(path)
    old.initialize()
    old.create_memory("fact", "User likes tea", "user likes tea")
    assert "notes" not in _tables(old)
    assert _versions(old) == {1, 2, 3, 4, 5}

    monkeypatch.setattr(db_module, "_MIGRATIONS", full)
    upgraded = Database(path)
    upgraded.initialize()
    assert {"notes", "notes_fts"} <= _tables(upgraded)
    assert {1, 2, 3, 4, 5, 6} <= _versions(upgraded)
    assert len(upgraded.list_memories()) == 1


def test_list_puts_pinned_first_then_newest(db: Database) -> None:
    first = db.create_note("First", "one")
    second = db.create_note("Second", "two")
    pinned = db.create_note("Pinned", "three", pinned=True)
    assert [n["id"] for n in db.list_notes()] == [pinned["id"], second["id"], first["id"]]


def test_create_and_get_round_trip(db: Database) -> None:
    note = db.create_note(
        "Packing", "- passport", ["travel"], origin="kiwibro", source_conversation_id="chat_1", pinned=True,
    )
    assert note["id"].startswith("note_")
    assert note["tags"] == ["travel"]
    assert note["pinned"] is True
    assert note["origin"] == "kiwibro"
    assert note["source_conversation_id"] == "chat_1"
    assert db.get_note(note["id"]) == note
    assert db.get_note("note_missing") is None
    with pytest.raises(ValueError):
        db.create_note("Bad", "body", origin="chat")


def test_fts_finds_title_body_and_tag(db: Database) -> None:
    note = db.create_note("Quarterly review", "discussed the roadmap", ["planning"])
    for word in ("quarterly", "roadmap", "planning"):
        assert [n["id"] for n in db.search_notes(word)] == [note["id"]]
    assert db.list_notes(query="roadmap")[0]["id"] == note["id"]
    assert db.search_notes("unrelated") == []


def test_tag_filter_and_count(db: Database) -> None:
    food = db.create_note("Dinner", "lahmacun", ["food"])
    db.create_note("Garden", "tomatoes", ["plants"])
    assert [n["id"] for n in db.list_notes(tag="food")] == [food["id"]]
    assert db.list_notes(tag="FOOD")[0]["id"] == food["id"]
    assert db.count_notes(tag="food") == 1
    assert db.count_notes() == 2
    assert db.count_notes(query="tomatoes") == 1
    assert db.count_notes(query="tomatoes", tag="food") == 0


def test_unicode_tag_filter_matches_cleaned_tag(db: Database) -> None:
    note = db.create_note("Trip", "two nights", clean_tags(["İstanbul"]))
    assert note["tags"] == ["istanbul"]
    assert [n["id"] for n in db.list_notes(tag="İstanbul")] == [note["id"]]


def test_update_keeps_fts_in_sync(db: Database) -> None:
    note = db.create_note("Plan", "alpha beta", ["work"])
    updated = db.update_note(note["id"], body="gamma delta", tags=["home"])
    assert updated is not None and updated["body"] == "gamma delta"
    assert db.search_notes("alpha") == []
    assert [n["id"] for n in db.search_notes("gamma")] == [note["id"]]
    assert [n["id"] for n in db.search_notes("home")] == [note["id"]]
    assert db.search_notes("work") == []

    db.update_note(note["id"], title="Renamed", pinned=True)
    assert [n["id"] for n in db.search_notes("renamed")] == [note["id"]]
    assert db.get_note(note["id"])["pinned"] is True
    assert db.update_note("note_missing", title="x") is None
    with pytest.raises(ValueError):
        db.update_note(note["id"], origin="kiwibro")


def test_delete_removes_note_and_fts_row(db: Database) -> None:
    note = db.create_note("Gone", "soon")
    assert db.delete_note(note["id"]) is True
    assert db.delete_note(note["id"]) is False
    assert db.get_note(note["id"]) is None
    assert db.search_notes("soon") == []
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM notes_fts").fetchone()["n"] == 0


def test_note_tags_counts_most_used_first(db: Database) -> None:
    db.create_note("A", "a", ["food", "home"])
    db.create_note("B", "b", ["food"])
    db.create_note("C", "c", ["alpha"])
    assert db.note_tags() == [("food", 2), ("alpha", 1), ("home", 1)]


def test_clean_title_edge_cases() -> None:
    assert clean_title("") == ""
    assert clean_title("   # Heading   here  ") == "Heading here"
    assert clean_title('"Quoted"') == "Quoted"
    assert clean_title("“Curly”") == "Curly"
    assert clean_title("x" * 200) == "x" * MAX_TITLE
    with pytest.raises(ValueError):
        clean_title(42)


def test_derive_title_uses_first_line_and_cuts_at_word_boundary() -> None:
    assert derive_title("") == UNTITLED
    assert derive_title("\n \n# First line here\nsecond line") == "First line here"
    long_title = derive_title("word " * 30)
    assert long_title.endswith("…")
    assert len(long_title) <= 60
    assert all(part == "word" for part in long_title[:-1].split(" "))
    mid_word = derive_title(" ".join(["abcdefghij"] * 10))
    assert mid_word == " ".join(["abcdefghij"] * 5) + "…"


def test_clean_body_normalises_and_checks_size() -> None:
    assert clean_body("a  \r\nb\r\n\r\n") == "a\nb"
    assert clean_body("\n  \n  indented\n") == "  indented"
    assert clean_body("x" * MAX_BODY) == "x" * MAX_BODY
    with pytest.raises(ValueError, match="note is empty"):
        clean_body("")
    with pytest.raises(ValueError, match="note is empty"):
        clean_body("   \n\t\n")
    with pytest.raises(ValueError, match="note is too long"):
        clean_body("x" * (MAX_BODY + 1))


def test_clean_tags_rules() -> None:
    assert clean_tags("#Tag, tag  other") == ["tag", "other"]
    assert clean_tags(["Food", "#food", "  Two Words ", ""]) == ["food", "two", "words"]
    assert clean_tags("İstanbul") == ["istanbul"]
    assert clean_tags("a!b?c") == ["abc"]
    assert clean_tags(None) == []
    assert clean_tags("") == []
    assert len(clean_tags([f"t{i}" for i in range(20)])) == MAX_TAGS
    assert clean_tags("x" * 50) == ["x" * MAX_TAG_CHARS]
    with pytest.raises(ValueError):
        clean_tags(["ok", 3])
    with pytest.raises(ValueError):
        clean_tags(7)


def test_kiwibro_save_replaces_same_title_only(notebook: Notebook, db: Database) -> None:
    first = notebook.save("Version one", "Weekly plan", ["plan"], origin="kiwibro")
    second = notebook.save("Version two", "weekly PLAN", origin="kiwibro")
    assert second["id"] == first["id"]
    assert second["body"] == "Version two"
    assert second["tags"] == ["plan"]  # a rewrite that names no tags keeps them
    assert db.count_notes() == 1

    user_a = notebook.save("Mine", "Weekly plan", origin="you")
    user_b = notebook.save("Also mine", "Weekly plan", origin="you")
    assert len({user_a["id"], user_b["id"], first["id"]}) == 3
    assert db.count_notes() == 3


def test_kiwibro_save_never_overwrites_a_page_the_user_wrote(notebook: Notebook, db: Database) -> None:
    mine = notebook.save("My own words", "Trip ideas", origin="you")
    theirs = notebook.save("Generated list", "Trip ideas", origin="kiwibro")
    assert theirs["id"] != mine["id"]
    assert db.get_note(mine["id"])["body"] == "My own words"


def test_save_derives_title_and_validates(notebook: Notebook) -> None:
    note = notebook.save("# Shopping\n\nmilk", None, "groceries")
    assert note["title"] == "Shopping"
    assert note["tags"] == ["groceries"]
    assert note["origin"] == "you"
    with pytest.raises(ValueError):
        notebook.save("   ")


def test_append_adds_after_blank_line(notebook: Notebook, db: Database) -> None:
    note = notebook.save("Line one", "Journal")
    appended = notebook.append("Journal", "Line two")
    assert appended is not None and appended["id"] == note["id"]
    assert db.get_note(note["id"])["body"] == "Line one\n\nLine two"
    assert notebook.append("Nothing like this", "x") is None
    with pytest.raises(ValueError, match="too long"):
        notebook.append("Journal", "x" * MAX_BODY)


def test_find_by_id_title_and_fuzzy(notebook: Notebook) -> None:
    note = notebook.save("Trip planning for the spring", "Istanbul trip")
    assert notebook.find(note["id"])["id"] == note["id"]
    assert notebook.find("  ISTANBUL TRIP ")["id"] == note["id"]
    assert notebook.find("istanbul trip notes")["id"] == note["id"]
    assert notebook.find("dentist appointment") is None
    assert notebook.find("") is None


def test_find_ignores_body_only_matches(notebook: Notebook) -> None:
    notebook.save("passport and charger", "Packing")
    assert notebook.find("passport") is None


def test_relevant_puts_pinned_first_and_excludes_unrelated(notebook: Notebook, db: Database) -> None:
    db.create_note("Wifi password", "Router admin page is 192.168.1.1", pinned=True)
    notebook.save("Lahmacun and soup for guests", "Dinner ideas", ["food"])
    notebook.save("Tomatoes need full sun", "Garden")
    picked = [n["title"] for n in notebook.relevant("what should we cook for dinner tonight")]
    assert picked == ["Wifi password", "Dinner ideas"]
    assert [n["title"] for n in notebook.relevant("dinner", limit=1)] == ["Wifi password"]
    assert [n["title"] for n in notebook.relevant("tomatoes")] == ["Wifi password", "Garden"]
    assert [n["title"] for n in notebook.relevant("dentist appointment")] == ["Wifi password"]


def test_relevant_caps_pinned_and_avoids_duplicates(notebook: Notebook, db: Database) -> None:
    for title in ("Dinner rota", "Dinner menu", "Dinner budget"):
        db.create_note(title, "dinner", pinned=True)
    assert len(notebook.relevant("zzz")) == 2
    picked = notebook.relevant("dinner", limit=5)
    assert len({n["id"] for n in picked}) == len(picked) == 3


def test_render_block_shows_tags_and_stays_within_budget(notebook: Notebook) -> None:
    assert Notebook.render_block([]) == ""
    note = {"title": "Dinner ideas", "tags": ["food", "cooking"], "body": "Lahmacun\n\nwith   soup"}
    assert Notebook.render_block([note]) == "- Dinner ideas [food, cooking]: Lahmacun with soup"
    plain = {"title": "Plain", "tags": [], "body": "no tags here"}
    assert Notebook.render_block([plain]) == "- Plain: no tags here"

    long_notes = [{"title": f"Note {i}", "tags": [], "body": "word " * 800} for i in range(3)]
    block = Notebook.render_block(long_notes, max_chars=900)
    assert len(block) <= 900
    assert block.count("\n") == 2
    assert all(line.startswith("- Note") and "…" in line for line in block.split("\n"))

    many = [{"title": f"Many {i}", "tags": ["t"], "body": "x " * 500} for i in range(30)]
    assert len(Notebook.render_block(many, max_chars=900)) <= 900


def test_api_round_trip(env: Any) -> None:
    created = env.client.post(
        "/api/notes", json={"body": "Packing list\n\n- passport", "tags": "travel, #Trip"},
    )
    assert created.status_code == 200
    note = created.get_json()["note"]
    assert note["title"] == "Packing list"
    assert note["tags"] == ["travel", "trip"]
    assert note["pinned"] is False
    assert note["origin"] == "you"
    note_id = note["id"]

    assert env.client.get(f"/api/notes/{note_id}").get_json()["note"] == note

    patched = env.client.patch(f"/api/notes/{note_id}", json={"pinned": True, "body": "Packed"}).get_json()
    assert patched["ok"] is True
    assert patched["note"]["pinned"] is True
    assert patched["note"]["body"] == "Packed"

    listed = env.client.get("/api/notes").get_json()
    assert listed["ok"] is True
    assert listed["total"] == 1
    assert [n["id"] for n in listed["notes"]] == [note_id]
    assert listed["tags"] == [{"tag": "travel", "count": 1}, {"tag": "trip", "count": 1}]

    assert env.client.delete(f"/api/notes/{note_id}").get_json() == {"ok": True}
    assert env.client.get("/api/notes").get_json()["total"] == 0


def test_api_filters_by_query_and_tag(env: Any) -> None:
    env.client.post("/api/notes", json={"body": "Passport renewal due in May", "tags": ["admin"]})
    env.client.post("/api/notes", json={"body": "Lahmacun recipe", "tags": ["food"]})

    by_query = env.client.get("/api/notes?q=passport").get_json()
    assert by_query["total"] == 1
    assert by_query["notes"][0]["title"] == "Passport renewal due in May"

    by_tag = env.client.get("/api/notes?tag=food").get_json()
    assert by_tag["total"] == 1
    assert by_tag["notes"][0]["tags"] == ["food"]

    assert env.client.get("/api/notes?q=zzzz").get_json()["total"] == 0
    assert env.client.get("/api/notes?q=%20%20").get_json()["total"] == 2


def test_api_validation_and_not_found(env: Any) -> None:
    empty = env.client.post("/api/notes", json={"body": "   "})
    assert empty.status_code == 400
    assert empty.get_json()["error"]["code"] == "bad_request"
    assert env.client.post("/api/notes", data="not json").status_code == 400
    assert env.client.post("/api/notes", json={"body": "ok", "tags": 3}).status_code == 400

    note_id = env.client.post("/api/notes", json={"body": "Real"}).get_json()["note"]["id"]
    bad_pin = env.client.patch(f"/api/notes/{note_id}", json={"pinned": "yes"})
    assert bad_pin.status_code == 400
    assert env.client.patch(f"/api/notes/{note_id}", json={"body": ""}).status_code == 400

    missing = env.client.get("/api/notes/note_missing")
    assert missing.status_code == 404
    assert missing.get_json()["error"]["code"] == "not_found"
    assert env.client.patch("/api/notes/note_missing", json={"body": "x"}).status_code == 404
    assert env.client.delete("/api/notes/note_missing").status_code == 404
