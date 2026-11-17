"""Opt-in warm interaction workload through a single KiwiBro application instance."""

from contextlib import contextmanager
import os
import time
from types import SimpleNamespace

import pytest

from kiwibro.llm import ChatResult
from scripts.mind.accept_finalist import INTERACTION_REPEATS, JOURNAL, append_journal, collect_journal, interaction_summary
from test_native_literal_read import native_read  # noqa: F401 - scripted native model identity
from test_successor_acceptance import (  # noqa: F401
    NATIVE_DATA, acceptance, call, offline_chrome, turn, verified_native_pack, write,
)


class InteractionBusy(AssertionError):
    """An earlier operation still owns the application after cancellation."""


def wait_for_idle(env):
    deadline = time.monotonic() + 10
    while env.svc.agent.active_turns():
        if time.monotonic() >= deadline:
            raise InteractionBusy("The application did not become idle after the previous turn")
        time.sleep(0.05)


@contextmanager
def measured(env, record, step, repeat):
    wait_for_idle(env)
    stats = {}
    failure = None
    try:
        yield stats
    except pytest.fail.Exception:
        if not stats.get("timed_out"):
            raise
        failure = "timeout"
    except AssertionError:
        terminal = stats.get("terminal")
        failure = terminal if terminal in {"error", "aborted", "missing"} else "assertion"
    else:
        stats["verified"] = True
    finally:
        if not stats.setdefault("verified", False):
            stats["failure_kind"] = failure or "interrupted"
        record({"step": step, "repeat": repeat, **stats})


@pytest.mark.skipif(not os.environ.get(INTERACTION_REPEATS), reason="Select --case interaction in accept_finalist.")
def test_warm_app_interaction(acceptance, offline_chrome, record_property):  # noqa: F811
    repeats = int(os.environ[INTERACTION_REPEATS])
    assert 1 <= repeats <= 20
    rows = []

    def record(row):
        append_journal({"interaction_turn": row})
        record_property("turn", row)
        rows.append(row)

    run_interaction(acceptance, repeats, record)
    assert sum(row["verified"] for row in rows) == 3 * repeats


def run_interaction(env, repeats, record):
    if env.fake:
        env.fake.stream_script = [ChatResult(content="READY")]
    turn(env, "Reply with the word READY to start the local benchmark.", set(), tool_profile="auto")

    for repeat in range(1, repeats + 1):
        tag = f"PORT-{repeat:03d}"
        with measured(env, record, "text", repeat) as times:
            if env.fake:
                env.fake.stream_script = [ChatResult(content=tag)]
            _, messages = turn(
                env, f"The delivery labels read NORTH, {tag}, and SOUTH. Reply with just the middle label.",
                set(), observer=times.update, tool_profile="auto",
            )
            assert messages[-1]["content"].strip().strip("`.*") == tag
            assert times["first_text_ms"] is not None and times["done_ms"] is not None

        marker = f"DISPATCH-{repeat:03d}"
        path = f"dispatch-{repeat:03d}.txt"
        with measured(env, record, "file", repeat) as times:
            write(env.workspace, path, f"Dispatch marker: {marker}\n")
            if env.fake:
                env.fake.stream_script = [call(1, "read_file", path=path), ChatResult(content=marker)]
            events, messages = turn(
                env, f"Read {path} with the file reading tool. Reply with only its dispatch marker.",
                {"read_file"}, observer=times.update, tool_profile="auto", allow_direct_read=True,
            )
            assert any(event == "tool_result" and data["name"] == "read_file" and data["ok"]
                       and marker in data["preview"] for event, data in events)
            assert messages[-1]["content"].strip().strip("`.*") == marker
            assert times["first_verified_tool_ms"] is not None and times["done_ms"] is not None
        if not times["verified"]:
            record({"step": "render", "repeat": repeat, "verified": False, "skip_reason": "file_unverified"})
            continue

        preview = f"dispatch-preview-{repeat:03d}.html"
        source = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                  '<meta name="viewport" content="width=device-width, initial-scale=1">'
                  '<title>Dispatch</title><main><h1>Dispatch preview</h1><p>WAITING</p></main></html>\n')
        with measured(env, record, "render", repeat) as times:
            write(env.workspace, preview, source)
            if env.fake:
                env.fake.stream_script = [
                    call(2, "replace_in_file", path=preview, old="WAITING", new=marker),
                    call(3, "look_at_page", path=preview),
                    ChatResult(content=f"Updated and checked {preview}."),
                ]
            rendered, messages = turn(
                env,
                f"In {preview}, replace WAITING with the dispatch marker we just read. Leave the remaining markup "
                "exactly as it is and verify the phone preview.",
                {"read_file", "replace_in_file", "apply_patch", "look_at_page"}, allowed_paths={preview},
                conversation_id=events[0][1]["conversation_id"], observer=times.update, tool_profile="auto",
            )
            page = env.workspace / preview
            assert page.is_file() and page.read_bytes() == source.replace("WAITING", marker).encode("utf-8")
            tools = [data["name"] for event, data in rendered if event == "tool_result" and data["ok"]]
            changes = [i for i, name in enumerate(tools) if name in ("replace_in_file", "apply_patch")]
            assert changes and "look_at_page" in tools[max(changes) + 1:]
            looks = [row["content"] for row in messages if row.get("tool_name") == "look_at_page"]
            assert looks and "fits a phone screen" in looks[-1]
            assert (env.workspace / "screenshots" / f"look-dispatch-preview-{repeat:03d}.png").is_file()
            assert all(times[key] is not None for key in ("first_verified_tool_ms", "first_render_ms", "done_ms"))
    wait_for_idle(env)


@pytest.mark.parametrize("altered_evidence", [None, "no_opt_in", "usage", "failed_read", "wrong_model", "other_reply"])
def test_interaction_exact_read_requires_direct_reply_evidence(env, native_read, monkeypatch, altered_evidence):  # noqa: F811
    path = "dispatch-test.txt"
    marker = "DISPATCH-TEST"
    write(env.workspace, path, f"Dispatch marker: {marker}\n")
    # Keep the app and scripted transport, and assert the native SSE contract.
    fake = env.fake
    monkeypatch.setattr(env, "fake", None)
    monkeypatch.setenv(NATIVE_DATA, str(env.workspace))
    agent = env.svc.agent
    original_frame = agent._frame

    def frame(event, data):
        if event == "done" and altered_evidence == "usage":
            data = {**data, "usage": {"completion_tokens": 1}}
        if event == "done" and altered_evidence == "wrong_model":
            data = {**data, "model": "other"}
        if event == "done" and altered_evidence == "other_reply":
            data = {**data, "inference": {"said_by_kiwibro": "look"}}
        if event == "tool_result" and altered_evidence == "failed_read":
            data = {**data, "ok": False}
        return original_frame(event, data)

    monkeypatch.setattr(agent, "_frame", frame)
    observed = []
    options = {"tool_profile": "auto", "observer": observed.append}
    if altered_evidence != "no_opt_in":
        options["allow_direct_read"] = True
    prompt = f"Read {path} with the file reading tool. Reply with only its dispatch marker."
    if altered_evidence is None:
        events, messages = turn(env, prompt, {"read_file"}, **options)
        assert messages[-1]["content"] == marker
        assert events[-1][1]["inference"]["said_by_kiwibro"] == "read_file"
        assert observed[0]["native_reply"] == "direct_read"
    else:
        with pytest.raises(AssertionError):
            turn(env, prompt, {"read_file"}, **options)
        assert observed[0]["terminal"] == "done" and observed[0]["native_reply"] is None
    assert len(observed) == 1 and not fake.stream_calls


def test_interaction_resumes_with_same_app_after_timeout_and_wrong_marker(env, offline_chrome, monkeypatch):  # noqa: F811
    private_text = "incorrect label from /private/model.gguf"
    original_turn = turn

    def inject_failure(app, prompt, allowed, **kwargs):
        assert app is env
        if "dispatch-preview-001.html" in prompt:
            def delayed():
                time.sleep(0.4)
                return ChatResult(content=private_text)

            env.fake.stream_script = [delayed]
            with monkeypatch.context() as config:
                config.setenv("KIWIBRO_ACCEPTANCE_TURN_TIMEOUT", "0.15")
                return original_turn(env, prompt, allowed, **kwargs)
        if "dispatch-002.txt" in prompt:
            env.fake.stream_script[-1] = ChatResult(content=private_text)
        return original_turn(env, prompt, allowed, **kwargs)

    monkeypatch.setattr(__name__ + ".turn", inject_failure)
    monkeypatch.setenv(INTERACTION_REPEATS, "3")
    rows = []
    with pytest.raises(AssertionError):
        test_warm_app_interaction(env, offline_chrome, lambda key, value: rows.append(value))
    assert [(row["step"], row["repeat"], row.get("failure_kind"), row.get("skip_reason"))
            for row in rows if not row["verified"]] == [
        ("render", 1, "timeout", None),
        ("file", 2, "assertion", None),
        ("render", 2, None, "file_unverified"),
    ]
    assert len(rows) == 9 and all(row["verified"] for row in rows[-3:])
    assert private_text not in str(rows)
    assert not (env.workspace / "dispatch-preview-002.html").exists()
    render = interaction_summary(rows, 3)["steps"]["render"]
    assert (render["attempted"], render["failed"], render["skipped"], render["unreported"]) == (2, 1, 1, 0)
    assert render["failure_counts"] == {"timeout": 1} and render["skip_counts"] == {"file_unverified": 1}
    assert render["metrics"]["done_ms"]["n"] == 1


def test_interaction_waits_for_turn_cleanup_and_preserves_resource_failure(env, monkeypatch):
    rows = []
    with monkeypatch.context() as stalled:
        stalled.setattr(env.svc.agent, "active_turns", lambda: 1)
        stalled.setattr(__name__ + ".time", SimpleNamespace(monotonic=iter([0, 11]).__next__))
        with pytest.raises(InteractionBusy):
            with measured(env, rows.append, "file", 1):
                pytest.fail("The previous turn still owns the app")
    assert rows == []

    with pytest.raises(MemoryError):
        with measured(env, rows.append, "file", 1):
            raise MemoryError("private resource diagnostics")
    assert rows == [{"step": "file", "repeat": 1, "verified": False, "failure_kind": "interrupted"}]


def test_suite_timeout_keeps_completed_interaction_turns(tmp_path, monkeypatch):
    journal = tmp_path / "events.jsonl"
    monkeypatch.setenv(JOURNAL, str(journal))
    row = {"step": "text", "repeat": 1, "verified": True, "first_text_ms": 10, "done_ms": 15}
    append_journal({"interaction_turn": row})
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"interaction_turn":')
    report = {"phases": [], "suite_timed_out": True, "passed": False}
    collect_journal(journal, report, 3)
    assert report["suite_timed_out"] and not report["passed"] and report["unfinished_evidence_write"]
    assert report["interaction"]["partial_turns"] == [row]
    steps = report["interaction"]["steps"]
    assert steps["text"]["metrics"]["done_ms"]["n"] == 1 and steps["text"]["unreported"] == 2
    assert steps["file"]["unreported"] == steps["render"]["unreported"] == 3


def test_interaction_percentiles_exclude_failed_and_reloaded_turns():
    verified = [{"step": "text", "verified": True, "first_text_ms": 10 * i, "done_ms": 100 * i}
                for i in range(1, 11)]
    excluded = [
        {"step": "text", "verified": False, "timed_out": True, "first_text_ms": 9999},
        {"step": "text", "verified": True, "first_text_ms": 9999, "native_execution": {
            "backend_before": "gpu", "backend_after": "cpu", "launches": [{"backend": "cpu"}], "gpu_fallbacks": 1,
        }},
    ]
    result = interaction_summary(verified + excluded, 12)["steps"]["text"]
    assert result["observed"] == 12 and result["verified"] == 11 and result["warm_verified"] == 10
    assert result["timeouts"] == 1
    assert result["metrics"]["first_text_ms"] == {"n": 10, "p50": 55, "p90": 90}
    assert result["metrics"]["done_ms"] == {"n": 10, "p50": 550, "p90": 900}
    assert result["metrics"]["first_verified_tool_ms"] == {"n": 0, "p50": None, "p90": None}
