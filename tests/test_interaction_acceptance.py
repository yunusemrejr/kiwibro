"""Opt-in warm interaction workload through a single KiwiBro application instance."""

from contextlib import contextmanager
import os

import pytest

from kiwibro import pagecheck
from kiwibro.llm import ChatResult
from scripts.mind.accept_finalist import INTERACTION_REPEATS, interaction_summary
from test_successor_acceptance import acceptance, call, offline_chrome, turn, verified_native_pack, write  # noqa: F401


@contextmanager
def measured(record_property, step, repeat):
    stats = {}
    verified = False
    try:
        yield stats
        verified = True
    finally:
        record_property("turn", {"step": step, "repeat": repeat, "verified": verified, **stats})


@pytest.mark.skipif(not os.environ.get(INTERACTION_REPEATS), reason="Select --case interaction in accept_finalist.")
def test_warm_app_interaction(acceptance, offline_chrome, record_property):
    repeats = int(os.environ[INTERACTION_REPEATS])
    assert 1 <= repeats <= 20
    env = acceptance
    if env.fake:
        env.fake.stream_script = [ChatResult(content="READY")]
    turn(env, "Reply with the word READY to start the local benchmark.", set(), tool_profile="auto")

    for repeat in range(1, repeats + 1):
        tag = f"PORT-{repeat:03d}"
        if env.fake:
            env.fake.stream_script = [ChatResult(content=tag)]
        with measured(record_property, "text", repeat) as times:
            _, messages = turn(
                env, f"The delivery labels read NORTH, {tag}, and SOUTH. Reply with just the middle label.",
                set(), observer=times.update, tool_profile="auto",
            )
            assert messages[-1]["content"].strip().strip("`.*") == tag
            assert times["first_text_ms"] is not None and times["done_ms"] is not None

        marker = f"DISPATCH-{repeat:03d}"
        path = f"dispatch-{repeat:03d}.txt"
        write(env.workspace, path, f"Dispatch marker: {marker}\n")
        if env.fake:
            env.fake.stream_script = [call(1, "read_file", path=path), ChatResult(content=marker)]
        with measured(record_property, "file", repeat) as times:
            events, messages = turn(
                env, f"Read {path} with the file reading tool. Reply with only its dispatch marker.",
                {"read_file"}, observer=times.update, tool_profile="auto",
            )
            assert any(event == "tool_result" and data["name"] == "read_file" and data["ok"] for event, data in events)
            assert messages[-1]["content"].strip().strip("`.*") == marker
            assert times["first_verified_tool_ms"] is not None and times["done_ms"] is not None

        preview = f"dispatch-preview-{repeat:03d}.html"
        source = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                  '<meta name="viewport" content="width=device-width, initial-scale=1">'
                  '<title>Dispatch</title><main><h1>Dispatch preview</h1><p>WAITING</p></main></html>\n')
        write(env.workspace, preview, source)
        if env.fake:
            env.fake.stream_script = [
                call(2, "replace_in_file", path=preview, old="WAITING", new=marker),
                call(3, "look_at_page", path=preview),
                ChatResult(content=f"Updated and checked {preview}."),
            ]
        with measured(record_property, "render", repeat) as times:
            rendered, messages = turn(
                env,
                f"In {preview}, replace WAITING with the dispatch marker we just read. Leave the remaining markup "
                "exactly as it is and verify the phone preview.",
                {"read_file", "replace_in_file", "apply_patch", "look_at_page"}, allowed_paths={preview},
                conversation_id=events[0][1]["conversation_id"], observer=times.update, tool_profile="auto",
            )
            assert (env.workspace / preview).read_text(encoding="utf-8") == source.replace("WAITING", marker)
            tools = [data["name"] for event, data in rendered if event == "tool_result" and data["ok"]]
            changes = [i for i, name in enumerate(tools) if name in ("replace_in_file", "apply_patch")]
            assert changes and "look_at_page" in tools[max(changes) + 1:]
            looks = [row["content"] for row in messages if row.get("tool_name") == "look_at_page"]
            assert looks and "fits a phone screen" in looks[-1]
            assert (env.workspace / "screenshots" / f"look-dispatch-preview-{repeat:03d}.png").is_file()
            assert all(times[key] is not None for key in ("first_verified_tool_ms", "first_render_ms", "done_ms"))


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
