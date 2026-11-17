"""Raster files reach vision by name, while the text reader keeps real text."""

from io import BytesIO

import pytest
from PIL import Image, features

from kiwibro import tools
from kiwibro.llm import ChatResult
from kiwibro.sandbox import MAX_READ_BYTES, Workspace
import test_backend
from test_backend import Env, collect_sse, tool_call_result

env = test_backend.env


def picture(kind: str = "PNG") -> bytes:
    output = BytesIO()
    Image.new("RGB", (3, 4), "#719abc").save(output, format=kind)
    return output.getvalue()


@pytest.mark.parametrize("kind", ["PNG", "JPEG", "GIF", "WEBP", "BMP", "TIFF"])
def test_raster_bytes_are_refused_even_with_an_ordinary_text_name(tmp_path, kind: str):
    if kind == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow was built without WebP encoding")
    workspace = Workspace(tmp_path)
    (workspace.root / "plain.txt").write_bytes(picture(kind))
    context = tools.ToolContext(workspace, None, None, "fixture")
    result = tools.execute_tool("read_file", {"path": "plain.txt"}, context)

    assert not result.ok
    assert "plain.txt is a raster image" in result.content
    assert "describe_image" in result.content and "attachment observation" in result.content
    assert "\ufffd" not in result.content


def test_the_reader_keeps_real_text_and_other_file_content(tmp_path):
    workspace = Workspace(tmp_path)
    (workspace.root / "notes.gif").write_bytes(b"GIF89a reference\r\nsecond line\r\n")
    (workspace.root / "data.bin").write_bytes(b"Other bytes \xff\r\n")
    assert workspace.read_file("notes.gif", line_numbers=True)["content"] == "1: GIF89a reference\n2: second line"
    assert workspace.read_file("data.bin")["content"] == "Other bytes \ufffd\n"


@pytest.mark.parametrize("large", [False, True])
def test_batch_reports_raster_paths_and_keeps_text_siblings_in_order(tmp_path, large):
    workspace = Workspace(tmp_path)
    workspace.write_file("one.txt", "First line\nFirst marker\n")
    workspace.write_file("two.txt", "Second line\nSecond marker\n")
    snapshot = picture()
    if large:
        output = BytesIO()
        Image.new("RGB", (300, 300), "#719abc").save(output, format="PNG", compress_level=0)
        snapshot = output.getvalue()
        assert len(snapshot) > MAX_READ_BYTES
    (workspace.root / "snapshot.png").write_bytes(snapshot)
    paths = ["one.txt", "snapshot.png", "two.txt"]
    context = tools.ToolContext(workspace, None, None, "fixture")
    result = tools.execute_tool("read_files", {"paths": paths, "start": 2, "end": 2}, context)
    assert not result.ok and result.meta == {"paths": paths, "lines": 4, "raster_images": ["snapshot.png"]}
    assert result.content.startswith("Raster images in this batch: snapshot.png. Use describe_image")
    assert "=== one.txt (lines 2-2 of 2) ===\nFirst marker" in result.content
    assert "=== two.txt (lines 2-2 of 2) ===\nSecond marker" in result.content
    assert result.content.index("First marker") < result.content.index("Second marker")
    assert "\ufffd" not in result.content


def test_uploaded_raster_read_sends_help_without_starting_vision(env: Env, monkeypatch: pytest.MonkeyPatch):
    from kiwibro import delegate

    upload = env.client.post("/api/attachments", data={"file": (BytesIO(picture()), "fixture.png")})
    assert upload.status_code == 200
    path = upload.get_json()["attachment"]["path"]

    def vision(*args, **kwargs):
        raise AssertionError("A text read must never invoke vision")

    monkeypatch.setattr(delegate, "describe_image", vision)
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": path}, "read_raster"),
        ChatResult(content="The file needs an image description."),
    ]
    response = env.client.post("/api/chat", json={"message": f"Open {path} and tell me what the file contains."},
                               buffered=False)
    assert response.status_code == 200
    events, raw = collect_sse(response)
    assert [(data["name"], data["ok"]) for kind, data in events if kind == "tool_result"] == [("read_file", False)]
    assert "describe_image" in raw and "\ufffd" not in raw
    evidence = [row["content"] for row in env.fake.stream_calls[-1]["messages"] if row["role"] == "tool"]
    assert len(evidence) == 1 and len(evidence[0]) < 300
