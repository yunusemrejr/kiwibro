"""Native workspace reads recover from observed missing paths through the app."""

from __future__ import annotations

import pytest

from kiwibro import tools
from kiwibro.llm import ChatResult
from kiwibro.mind import KIWIMIND
from kiwibro.sandbox import Workspace
from test_backend import Env, SSEStream, chat_events, env, tool_call_result  # noqa: F401


MISSING = "stack/dispatch/src/dispatch/service.py"
PRESENT = "stack/services/dispatch/service.py"
REQUEST = "In stack/, trace the dispatch queues and report the implementation marker. Keep the files untouched."


def native_workspace(env: Env, monkeypatch: pytest.MonkeyPatch, *, listing: str = "auto") -> None:
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: {
        "id": KIWIMIND, "engine": "native-gguf", "tools": True, "context": 8192,
    })
    permissions = {name: "deny" for name in tools.tool_names()}
    permissions.update(read_file="auto", read_files="auto", list_files=listing)
    env.settings_patch({"autonomy": "supervised", "tool_profile": "full", "max_tool_rounds": 8,
                        "permissions": permissions})
    file = env.workspace / PRESENT
    file.parent.mkdir(parents=True)
    file.write_text("QUEUE_IMPL = 'NE-427-T'\n")


def test_native_missing_batch_read_feeds_the_verified_ancestor(env: Env, monkeypatch: pytest.MonkeyPatch):
    native_workspace(env, monkeypatch)
    env.fake.stream_script = [
        tool_call_result("read_files", {"paths": [MISSING, "stack/audit/src/service.py"]}),
        tool_call_result("list_files", {"path": "stack/services/dispatch"}, call_id="call_2"),
        tool_call_result("read_files", {"paths": [PRESENT]}, call_id="call_3"),
        ChatResult(content=f"The implementation marker in {PRESENT} is NE-427-T."),
    ]

    events, _ = chat_events(env, REQUEST)
    results = [(body["name"], body["ok"]) for event, body in events if event == "tool_result"]
    assert results == [("read_files", False), ("list_files", True), ("list_files", True), ("read_files", True)]
    listing = next(body for event, body in events if event == "tool_call" and body["name"] == "list_files")
    assert listing["args"] == {"path": "stack"}
    observed = env.fake.stream_calls[1]["messages"]
    assert any(row["role"] == "tool" and f"path does not exist: {MISSING}" in row["content"] for row in observed)
    assert observed[-1]["role"] == "tool" and "dir\tstack/services/dispatch/" in observed[-1]["content"]
    assert f"file\t{PRESENT}\t" in env.fake.stream_calls[2]["messages"][-1]["content"]
    assert events[-1][0] == "done"


@pytest.mark.parametrize("listing", ["auto", "deny"])
def test_native_repeated_missing_path_stops_before_an_extra_generation(
        env: Env, monkeypatch: pytest.MonkeyPatch, listing: str,
):
    native_workspace(env, monkeypatch, listing=listing)
    reads = []
    read = Workspace.read_file

    def record_read(workspace, path, *args, **kwargs):
        reads.append(path)
        return read(workspace, path, *args, **kwargs)

    monkeypatch.setattr(Workspace, "read_file", record_read)
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": MISSING}),
        tool_call_result("read_files", {"paths": ["./" + MISSING], "start": 7}, call_id="call_2"),
        ChatResult(content="A third model round should not run."),
    ]

    events, _ = chat_events(env, REQUEST)
    assert reads == [MISSING] and len(env.fake.stream_calls) == 2
    listings = [body for event, body in events if event == "tool_call" and body["name"] == "list_files"]
    assert len(listings) == (1 if listing == "auto" else 0)
    assert any(event == "tool_result" and body["preview"] == "Repeated missing read stopped" for event, body in events)
    assert any(event == "delta" and "reported missing" in body["text"] for event, body in events)


def test_native_recovery_respects_listing_approval(env: Env, monkeypatch: pytest.MonkeyPatch):
    native_workspace(env, monkeypatch, listing="ask")
    listed = []
    original = Workspace.list_files

    def list_files(workspace, folder=".", *args, **kwargs):
        listed.append(folder)
        return original(workspace, folder, *args, **kwargs)

    monkeypatch.setattr(Workspace, "list_files", list_files)
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": MISSING}),
        ChatResult(content="The requested dispatch path is missing."),
    ]

    response = env.client.post("/api/chat", json={"message": REQUEST}, buffered=False)
    stream = SSEStream(response).start()
    approval = stream.wait_for("approval", timeout=4)
    assert approval["tool"] == "list_files" and approval["args"] == {"path": "stack"}
    assert listed == []
    denied = env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "deny"})
    assert denied.status_code == 200
    events = stream.finish(timeout=4)
    assert listed == []
    assert any(kind == "tool_result" and row["name"] == "list_files" and not row["ok"] for kind, row in events)


def test_native_invalid_read_range_does_not_trigger_directory_recovery(env: Env, monkeypatch: pytest.MonkeyPatch):
    native_workspace(env, monkeypatch)
    env.fake.stream_script = [
        tool_call_result("read_files", {"paths": [PRESENT], "start": [1]}),
        tool_call_result("read_files", {"paths": [PRESENT]}, call_id="call_2"),
        ChatResult(content="The implementation marker is NE-427-T."),
    ]
    events, _ = chat_events(env, REQUEST)
    results = [(body["name"], body["ok"]) for event, body in events if event == "tool_result"]
    assert results == [("read_files", False), ("read_files", True)]


def test_native_repeated_missing_read_does_not_waste_a_useful_parallel_call(
        env: Env, monkeypatch: pytest.MonkeyPatch,
):
    native_workspace(env, monkeypatch)
    repeat = tool_call_result("read_file", {"path": MISSING}, call_id="call_2")
    repeat.tool_calls.extend(tool_call_result("read_file", {"path": PRESENT}, call_id="call_3").tool_calls)
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": MISSING}),
        repeat,
        ChatResult(content="The implementation marker is NE-427-T."),
    ]
    events, _ = chat_events(env, REQUEST)
    results = [(body["name"], body["ok"]) for event, body in events if event == "tool_result"]
    assert results == [("read_file", False), ("list_files", True), ("read_file", False), ("read_file", True)]
    assert len(env.fake.stream_calls) == 3
