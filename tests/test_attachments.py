"""Attachments: storage in the workspace, the fixed per-message note, API checks and the
end-to-end turn where a model that cannot see still gets the image's facts."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from kiwibro import media
from kiwibro.context import to_llm_message
from tests.test_backend import chat_events, env  # noqa: F401 - shared app fixture

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def test_save_attachment_is_confined_and_never_overwrites(tmp_path: Path):
    one = media.save_attachment(tmp_path, "../../etc/passwd", PNG, day="2026-10-22")
    two = media.save_attachment(tmp_path, "../../etc/passwd", PNG, day="2026-10-22")
    assert one["path"].startswith("attachments/2026-10-22/") and "/etc/" not in one["path"]
    assert one["path"] != two["path"] and one["path"].endswith("passwd.png")  # sniffed suffix
    assert (tmp_path / one["path"]).read_bytes() == PNG
    assert one == {"path": one["path"], "name": "passwd", "kind": "image", "size": len(PNG)}
    with pytest.raises(ValueError):
        media.save_attachment(tmp_path, "empty.txt", b"")
    with pytest.raises(ValueError):
        media.save_attachment(tmp_path, "big.bin", b"x" * (media.MAX_ATTACHMENT_BYTES + 1))


def test_attachment_note_is_fixed_text():
    items = [
        {"path": "attachments/d/a-photo.png", "kind": "image", "size": 3 * 2**20,
         "seen": {"model": "vlm-2b", "text": "A red circle."}},
        {"path": "attachments/d/b-memo.wav", "kind": "audio", "size": 2048,
         "seen": {"model": None, "text": "duration_seconds: 4.2"}},
        {"path": "attachments/d/c-report.pdf", "kind": "other", "size": 10},
    ]
    note = media.attachment_note(items)
    assert note.splitlines() == [
        "[The user attached image \u201ca-photo.png\u201d, saved as attachments/d/a-photo.png (3.0 MB). "
        "Description by vlm-2b:",
        "A red circle.]",
        "[The user attached audio \u201cb-memo.wav\u201d, saved as attachments/d/b-memo.wav (2 KB). "
        "Facts measured by code:",
        "duration_seconds: 4.2]",
        "[The user attached other \u201cc-report.pdf\u201d, saved as attachments/d/c-report.pdf (1 KB).]",
    ]
    assert media.attachment_note(None) == "" and media.attachment_note("junk") == ""
    message = {"role": "user", "content": "look", "meta": {"attachments": items}}
    assert to_llm_message(message) == to_llm_message(dict(message))  # same bytes every turn
    # the note comes first so the user's own words are what a small model reads last
    assert to_llm_message(message)["content"] == f"{note}\n\nlook"


def _upload(env, name: str, data: bytes):  # noqa: F811
    return env.client.post(
        "/api/attachments", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data"
    )


def test_upload_route_stores_in_workspace_and_rejects_bad_requests(env):  # noqa: F811
    response = _upload(env, "shot.png", PNG)
    assert response.status_code == 200, response.get_json()
    attachment = response.get_json()["attachment"]
    assert (env.workspace / attachment["path"]).is_file() and attachment["kind"] == "image"
    assert env.client.post("/api/attachments", data={}, content_type="multipart/form-data").status_code == 400
    assert _upload(env, "empty.txt", b"").status_code == 400


def test_chat_accepts_only_uploaded_attachments(env):  # noqa: F811
    outside = env.workspace / "notes.txt"
    outside.write_text("hi")
    for bad in ([{"path": "notes.txt"}], [{"path": "attachments/../../x"}], "nope",
                [{"path": "attachments/missing.png"}]):
        response = env.client.post("/api/chat", json={"message": "hi", "attachments": bad})
        assert response.status_code == 400, bad


def test_turn_describes_image_with_code_when_no_model_can_see(env):  # noqa: F811
    pytest.importorskip("PIL")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), (200, 30, 30)).save(buffer, "PNG")
    attachment = _upload(env, "red.png", buffer.getvalue()).get_json()["attachment"]
    body = {"message": "", "attachments": [{**attachment, "kind": "audio", "size": 1}]}  # client lies
    response = env.client.post("/api/chat", json=body, buffered=False)
    assert response.status_code == 200, response.get_data(as_text=True)
    from tests.test_backend import collect_sse

    events, _raw = collect_sse(response)
    names = [name for name, _data in events]
    assert names[0] == "meta" and names[-1] == "done"
    assert any(name == "status" and "Looking at red.png" in data.get("label", "") for name, data in events)
    conversation_id = events[0][1]["conversation_id"]
    user = next(m for m in env.svc.db.list_messages(conversation_id) if m["role"] == "user")
    assert user["content"] == "red.png"  # an attachment alone is captioned with its name
    stored = user["meta"]["attachments"][0]
    assert stored["kind"] == "image" and stored["size"] == len(buffer.getvalue())  # re-derived
    assert stored["seen"]["model"] is None and "#c81e1e" in stored["seen"]["text"]
    sent = json.dumps(env.fake.stream_calls[-1]["messages"])
    assert "Facts measured by code" in sent and attachment["path"] in sent
