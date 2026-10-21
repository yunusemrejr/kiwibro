"""Comprehensive unit and client integration tests for Kiwibro multitask features:
Workspace tools, TerminalManager/TERMINALS, TaskBoard, BrowserEngine, and additive API routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kiwibro import create_app
from kiwibro.browser import BROWSER, BrowserEngine, html_to_markdown, inspect_page, render_headless
from kiwibro.sandbox import TERMINALS, SandboxError, TerminalManager, Workspace
from kiwibro.search import SearchError
from kiwibro.tasks import TASK_BOARD, TaskBoard


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws_dir = tmp_path / "test_workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)
    return Workspace(ws_dir)


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ws_dir = tmp_path / "api_ws"
    ws_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "api_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(
        {
            "TESTING": True,
            "KIWIBRO_DATA_DIR": str(data_dir),
            "KIWIBRO_WORKSPACE": str(ws_dir),
            "KIWIBRO_LMSTUDIO_URL": "http://127.0.0.1:9999",
        }
    )
    svc = app.extensions["kiwibro"]

    # Mock LLM calls so tests remain fully offline
    class FakeLLM:
        def models_status(self):
            return "fake-model", [
                {"id": "qwen2.5-coder-7b", "name": "qwen2.5-coder-7b", "state": "loaded"},
            ]

        def complete(self, *args, **kwargs):
            from kiwibro.llm import Completion
            return Completion("Compact summary of conversation.")

    monkeypatch.setattr(svc, "llm", FakeLLM())
    return app.test_client(), svc


# ============================================================================
# 1. Workspace Tests
# ============================================================================

def test_workspace_grep_files(workspace: Workspace):
    workspace.write_file("dir1/alpha.txt", "Alpha line\nHello world\nAnother line")
    workspace.write_file("dir2/beta.txt", "hello World\nSecond alpha")

    # Case-sensitive match
    results = workspace.grep_files("Hello", ".", case_sensitive=True)
    assert len(results["matches"]) == 1
    assert results["matches"][0]["path"] == "dir1/alpha.txt"
    assert results["matches"][0]["line"] == 2

    # Case-insensitive match
    ci_results = workspace.grep_files("hello", ".")
    assert len(ci_results["matches"]) == 2


def test_workspace_batch_read_file(workspace: Workspace):
    workspace.write_file("doc1.txt", "First doc line 1\nFirst doc line 2")
    workspace.write_file("doc2.txt", "Second doc line 1")

    # Batch read with line numbers
    res = workspace.read_file("", paths=["doc1.txt", "doc2.txt"], line_numbers=True)
    assert "paths" in res
    assert "doc1.txt" in res["paths"]
    assert "doc2.txt" in res["paths"]
    assert "1: First doc line 1" in res["content"]
    assert "1: Second doc line 1" in res["content"]


def test_workspace_multi_hunk_replace(workspace: Workspace):
    workspace.write_file("code.py", "apple = 1\nbanana = 2\napple = 3\n")
    edits = [
        {"old": "apple", "new": "orange", "replace_all": True},
        {"old": "banana", "new": "grape"},
    ]
    res = workspace.replace_in_file("code.py", edits=edits)
    assert res["replacements"] == 3
    updated = workspace.read_file("code.py")["content"]
    assert "orange = 1" in updated
    assert "grape = 2" in updated
    assert "orange = 3" in updated


def test_workspace_output_truncation(workspace: Workspace):
    long_text = "START_" + ("X" * 2000) + "_END"
    truncated, was_trunc = Workspace._truncate_output(long_text, max_bytes=200)
    assert was_trunc is True
    assert truncated.startswith("START_")
    assert truncated.endswith("_END")
    assert "… [truncated" in truncated


# ============================================================================
# 2. TerminalManager Tests
# ============================================================================

def test_terminal_start_background(workspace: Workspace):
    tm = TerminalManager()
    info = tm.start_background(workspace, "sleep 5", wait_ms=100)
    assert info["running"] is True
    sid = info["id"]
    try:
        status = tm.status(sid, workspace=workspace)
        assert status["running"] is True
    finally:
        tm.kill(sid, workspace=workspace)


def test_terminal_run_stateful_preserves_cd(workspace: Workspace):
    tm = TerminalManager()
    sid = "sess_stateful_test"
    workspace.write_file("subdir/marker.txt", "found me")

    # Change directory in session
    res1 = tm.run_stateful(workspace, "mkdir -p subdir && cd subdir", session_id=sid)
    assert res1["exit_code"] == 0

    # Subsequent command runs from preserved cwd
    res2 = tm.run_stateful(workspace, "cat marker.txt", session_id=sid)
    assert res2["exit_code"] == 0
    assert "found me" in res2["output"]


def test_terminal_send_input_and_status(workspace: Workspace):
    tm = TerminalManager()
    cmd = "python3 -u -c 'import sys; line=sys.stdin.readline(); print(\"ECHO:\" + line.strip(), flush=True)'"
    info = tm.start_background(workspace, cmd, wait_ms=100)
    sid = info["id"]
    try:
        res = tm.send_input(sid, "hello cat\n", wait_ms=400, workspace=workspace)
        assert "ECHO:hello cat" in res["output"]
    finally:
        tm.kill(sid, workspace=workspace)


def test_terminal_kill(workspace: Workspace):
    tm = TerminalManager()
    info = tm.start_background(workspace, "sleep 10", wait_ms=100)
    sid = info["id"]
    kill_res = tm.kill(sid, workspace=workspace)
    assert kill_res["running"] is False
    with pytest.raises(SandboxError):
        tm.kill("nonexistent_session_id", workspace=workspace)


def test_terminal_list_and_kill_all(workspace: Workspace):
    tm = TerminalManager()
    i1 = tm.start_background(workspace, "sleep 10", conversation_id="conv_terms", wait_ms=50)
    i2 = tm.start_background(workspace, "sleep 10", conversation_id="conv_terms", wait_ms=50)
    listed = tm.list_sessions(conversation_id="conv_terms", workspace=workspace)
    assert len(listed) >= 2
    killed = tm.kill_all(conversation_id="conv_terms")
    assert killed >= 2


# ============================================================================
# 3. TaskBoard & Quality Verification Tests
# ============================================================================

def test_taskboard_add_and_list(tmp_path: Path):
    tb = TaskBoard(persistence_path=tmp_path / "tasks.json")
    t1 = tb.add_task("High priority item", priority="high", notes="urgent")
    t2 = tb.add_task("Low priority item", priority="low")

    tasks = tb.list_tasks()
    assert len(tasks) == 2
    # High priority should sort before low priority
    assert tasks[0]["id"] == t1["id"]
    assert tasks[1]["id"] == t2["id"]


def test_taskboard_update(tmp_path: Path):
    tb = TaskBoard(persistence_path=tmp_path / "tasks.json")
    t = tb.add_task("Initial title", priority="medium")
    updated = tb.update_task(t["id"], status="in_progress", notes="updated note")
    assert updated["status"] == "in_progress"
    assert updated["notes"] == "updated note"

    with pytest.raises(ValueError):
        tb.update_task(t["id"], status="not_a_valid_status")
    with pytest.raises(ValueError):
        tb.update_task("missing_id", status="done")


def test_taskboard_clear_tasks(tmp_path: Path):
    tb = TaskBoard(persistence_path=tmp_path / "tasks.json")
    t1 = tb.add_task("Task 1")
    t2 = tb.add_task("Task 2")
    tb.update_task(t1["id"], status="done")

    cleared = tb.clear_tasks(only_done=True)
    assert cleared == 1
    remaining = tb.list_tasks()
    assert len(remaining) == 1
    assert remaining[0]["id"] == t2["id"]


def test_taskboard_execute_dispatcher(tmp_path: Path):
    tb = TaskBoard(persistence_path=tmp_path / "tasks.json")
    add_res = tb.execute("add", {"title": "Execute task", "priority": "high"})
    assert add_res["ok"] is True

    summary_res = tb.execute("summary", {})
    assert summary_res["ok"] is True
    assert "Task Summary" in summary_res["summary"]
    assert "priority: high" in summary_res["summary"]


def test_taskboard_evaluate_deliverable(tmp_path: Path):
    tb = TaskBoard(persistence_path=tmp_path / "tasks.json")
    valid_code = "def get_total(x, y):\n    return x + y\n"
    res_valid = tb.evaluate_deliverable(valid_code, requirements=["get_total"])
    assert res_valid["passed"] is True
    assert res_valid["gates"]["code_verified"] is True
    assert res_valid["gates"]["no_placeholders"] is True

    placeholder_code = "def process():\n    # ... rest of code\n    pass\n"
    res_placeholder = tb.evaluate_deliverable(placeholder_code)
    assert res_placeholder["passed"] is False
    assert res_placeholder["gates"]["no_placeholders"] is False

    syntax_err_code = "def broken(\n"
    res_syntax = tb.evaluate_deliverable(syntax_err_code)
    assert res_syntax["passed"] is False
    assert res_syntax["gates"]["code_verified"] is False


# ============================================================================
# 4. BrowserEngine Tests
# ============================================================================

def test_html_to_markdown_full_elements():
    html = """
    <html>
      <head><title>Test Documentation</title></head>
      <body>
        <h1>Main Title</h1>
        <p>Paragraph with a <a href="/docs/guide">Guide Link</a> and <strong>bold</strong> text.</p>
        <table>
          <tr><th>Param</th><th>Type</th></tr>
          <tr><td>timeout</td><td>int</td></tr>
        </table>
        <ul>
          <li>First item</li>
          <li>Second item</li>
        </ul>
        <pre><code class="language-python">def hello(): pass</code></pre>
        <blockquote>Important notice</blockquote>
        <form action="/search" method="GET">
          <input type="text" name="q" value="kiwibro" />
          <button type="submit">Search</button>
        </form>
      </body>
    </html>
    """
    res = html_to_markdown(html, base_url="https://docs.example.com")
    assert res["title"] == "Test Documentation"
    assert "# Main Title" in res["markdown"]
    assert "[Guide Link](https://docs.example.com/docs/guide)" in res["markdown"]
    assert "| Param | Type |" in res["markdown"]
    assert "* First item" in res["markdown"]
    assert "```python" in res["markdown"]
    assert "> Important notice" in res["markdown"]

    assert len(res["links"]) == 1
    assert res["links"][0]["url"] == "https://docs.example.com/docs/guide"
    assert len(res["forms"]) == 1
    assert res["forms"][0]["action"] == "https://docs.example.com/search"


def test_html_to_markdown_css_selector():
    html = """
    <div id="sidebar">Ignore this sidebar content</div>
    <div id="content"><h2>Main Content</h2><p>Read this part</p></div>
    """
    res = html_to_markdown(html, selector="#content")
    assert "Main Content" in res["markdown"]
    assert "Ignore this sidebar" not in res["markdown"]

    with pytest.raises(SearchError):
        html_to_markdown(html, selector="#nonexistent_selector")


def test_browser_multitab_lifecycle():
    class MockSearch:
        def _guarded_get(self, url, **kwargs):
            return url, "text/html", "utf-8", f"<html><head><title>{url}</title></head><body><h1>Content of {url}</h1></body></html>".encode("utf-8")

    engine = BrowserEngine(search_engine=MockSearch())
    tab1 = engine.open_tab("https://example.com/page1", conversation_id="conv_browser")
    tab2 = engine.open_tab("https://example.com/page2", conversation_id="conv_browser")
    assert tab1["ok"] and tab2["ok"]

    tabs = engine.list_tabs("conv_browser")
    assert len(tabs) == 2
    assert tabs[1]["active"] is True

    # Switch back to tab 1
    switched = engine.switch_tab(tab1["tab_id"], conversation_id="conv_browser")
    assert switched["ok"] is True
    assert switched["tab_id"] == tab1["tab_id"]

    # Close tab 2
    closed = engine.close_tab(tab2["tab_id"], conversation_id="conv_browser")
    assert closed is True
    assert len(engine.list_tabs("conv_browser")) == 1


def test_browser_ssrf_rejection():
    with pytest.raises(SearchError):
        inspect_page("http://127.0.0.1:8080/private")
    with pytest.raises(SearchError):
        inspect_page("http://192.168.1.1/admin")


def test_render_headless_ssrf_and_handling():
    # Headless rendering rejects SSRF safely
    res_private = render_headless("http://127.0.0.1:9090")
    assert res_private is None

    res_empty = render_headless("")
    assert res_empty is None


# ============================================================================
# 5. Flask API Endpoint Tests
# ============================================================================

def test_api_get_turns(app_client):
    client, _svc = app_client
    resp = client.get("/api/turns")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "active_count" in data
    assert "conversations" in data


def test_api_conversation_fork(app_client):
    client, svc = app_client
    conv = svc.db.create_conversation(title="Source Chat", model="fake-model")
    svc.db.add_message(conv["id"], role="user", content="Hello in source")
    svc.db.add_message(conv["id"], role="assistant", content="Reply in source")

    resp = client.post(f"/api/conversations/{conv['id']}/fork", json={"seq": 2})
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["ok"] is True
    assert "Source Chat (fork)" in data["conversation"]["title"]

    # Verify fork 404 on missing conversation
    resp_404 = client.post("/api/conversations/missing_id/fork")
    assert resp_404.status_code == 404


def test_api_conversation_compact(app_client):
    client, svc = app_client
    conv = svc.db.create_conversation(title="Compact Chat", model="fake-model")
    svc.db.add_message(conv["id"], role="user", content="Msg 1")
    svc.db.add_message(conv["id"], role="assistant", content="Msg 2")

    resp = client.post(f"/api/conversations/{conv['id']}/compact")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "compacted" in data


def test_api_terminals(app_client):
    client, _svc = app_client
    # List terminals
    resp = client.get("/api/terminals")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert isinstance(data["terminals"], list)

    # 404 kill on non-existent terminal
    resp_kill = client.post("/api/terminals/nonexistent_session/kill")
    assert resp_kill.status_code == 404


def test_api_tasks(app_client):
    client, _svc = app_client
    # Add task
    resp_add = client.post("/api/tasks", json={"action": "add", "title": "API Test Task", "priority": "high"})
    assert resp_add.status_code == 200
    data_add = resp_add.get_json()
    assert data_add["ok"] is True

    # List tasks
    resp_list = client.get("/api/tasks")
    assert resp_list.status_code == 200
    data_list = resp_list.get_json()
    assert data_list["ok"] is True
    assert any(t["title"] == "API Test Task" for t in data_list["tasks"])
