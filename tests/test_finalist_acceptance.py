"""Finalist tasks that carry evidence between turns in one KiwiBro conversation."""

from __future__ import annotations

import logging
import os
import re
import sys

import pytest

from kiwibro.context import estimate_tokens
from kiwibro.llm import ChatResult, LLMError
from kiwibro.mind.native_generic import NATIVE_MODEL
from scripts.mind.accept_finalist import PLACEMENT
from test_backend import env  # noqa: F401 - scripted app fixture
from test_mind_native_generic import pack_at
from test_successor_acceptance import (  # noqa: F401 - imported pytest fixtures
    CHECK, FIXED_LIMIT, NATIVE_DATA, NOTE, ORIGINAL_LIMIT, PKG, acceptance, call, git, managed_backend,
    native_transitions, repo_fixture, run_fixture_tests, snapshot, turn, verified_native_pack,
)


def test_finalist_reviews_then_repairs_in_same_chat(acceptance, record_property):
    env = acceptance
    head = repo_fixture(env.workspace)
    contents = env.workspace / "stack"
    before = {path: digest for path, digest in snapshot(contents).items() if not path.startswith(".git/")}
    initial = run_fixture_tests(env.workspace)
    record_property("evidence", {"initial_fixture_failure_count": 2 if "failures=2" in initial.stderr else None})
    assert initial.returncode != 0 and "failures=2" in initial.stderr
    if env.fake:
        env.fake.stream_script = [
            call(1, "grep_files", path="stack", pattern="def queue_limit"),
            call(2, "read_file", path=PKG),
            ChatResult(content=f"Both consumers use {PKG}. The shared function replaces explicit zero with 16."),
            call(3, "apply_patch", path=PKG, edits=[{"old": ORIGINAL_LIMIT, "new": FIXED_LIMIT}]),
            call(4, "run_command", command=CHECK, cwd="stack"),
            ChatResult(content="Zero stays zero and omitted values still use 16. All four tests passed."),
        ]
    reads = {"git", "grep_files", "read_file", "read_files", "list_files"}
    first, messages = turn(
        env,
        "In stack/, trace why the dispatch and audit services both ignore an explicit zero. "
        "Read their shared policy implementation and name its path. This first step is diagnosis only.",
        reads, observer=lambda stats: record_property("turn", {"step": "diagnose", **stats}),
    )
    assert PKG in messages[-1]["content"]
    assert any(row.get("tool_name") in {"read_file", "read_files"} and ORIGINAL_LIMIT in row["content"]
               for row in messages)
    assert {path: value for path, value in snapshot(contents).items() if not path.startswith(".git/")} == before

    continued, _ = turn(
        env,
        "Fix the shared issue you just identified with one minimal source change. Keep omitted values at 16 "
        f"and explicit zero at zero. Leave the notes and examples untouched, and don't commit. Run {CHECK} in stack.",
        reads | {"apply_patch", "replace_in_file", "write_file", "run_command"},
        allowed_paths={PKG}, command=(CHECK, "stack"), conversation_id=first[0][1]["conversation_id"],
        observer=lambda stats: record_property("turn", {"step": "repair", **stats}),
    )
    final = run_fixture_tests(env.workspace)
    after = {path: digest for path, digest in snapshot(contents).items() if not path.startswith(".git/")}
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    clean_index = not git(contents, "diff", "--cached", "--name-only")
    same_head = git(contents, "rev-parse", "HEAD") == head
    policy_path = PKG.removeprefix("stack/")
    record_property("evidence", {
        "expected_file_changed": policy_path in changed,
        "unexpected_changed_files": len(set(changed) - {policy_path}), "fixture_tests_exit_code": final.returncode,
        "git_head_preserved": same_head, "git_index_clean": clean_index,
    })
    assert final.returncode == 0 and "Ran 4 tests" in final.stderr
    assert changed == [policy_path] and same_head and clean_index
    assert (env.workspace / NOTE).read_text() == "Hand edited before the task. Keep this.\n"
    assert any(event == "tool_result" and row["name"] == "run_command" and row["ok"] for event, row in continued)


def test_finalist_recovers_earlier_exact_constraint_after_long_history(acceptance, record_property):
    env = acceptance
    marker, path = "CRANE-58-W7P", "deliveries/orchid/final-proof"
    seed_goal = int(os.environ.get("KIWIBRO_ACCEPTANCE_HISTORY_TOKENS", 18000))
    folder = env.workspace / path
    folder.mkdir(parents=True)
    (folder / "keep.txt").write_text("Synthetic protected fixture.\n")
    before = snapshot(env.workspace)
    if env.fake:
        env.fake.models[0]["context"] = 8192
        env.fake.stream_script = [
            ChatResult(content="I recorded the final review constraint."),
            call(1, "recall", query="release checkpoint strictly read-only"),
            ChatResult(content=f"Keep {path} read-only. The release checkpoint is {marker}."),
        ]
    first, _ = turn(
        env,
        f"For the final review of this conversation, keep {path} strictly read-only. "
        f"The release checkpoint is {marker}. Acknowledge the constraint. Leave the workspace alone.",
        set(), observer=lambda stats: record_property("turn", {"step": "prime", **stats}),
    )
    conversation = first[0][1]["conversation_id"]
    # Load synthetic older notes in the same format as stored traffic. A separate native
    # reply for every filler page would measure note generation instead of history recovery.
    db = env.svc.db
    seeded = pages = 0
    while seeded < seed_goal:
        page = "\n".join(
            f"Audit row {pages:02d}.{index:03d}: inspection {index + 317} stayed in routine review. "
            f"The owner recorded package route {(pages + index) % 13} and tagged the item pending."
            for index in range(18)
        )
        db.add_message(conversation, "user", f"Working notes for routine review page {pages}.\n{page}")
        db.add_message(conversation, "assistant", "I have the routine review notes.")
        seeded += estimate_tokens(page)
        pages += 1
    record_property("evidence", {"seeded_history_tokens_estimate": seeded, "synthetic_history_pages": pages})
    _, messages = turn(
        env,
        "Recall the exact early final-review instruction from this chat. What directory must stay read-only, "
        "and what is the exact release checkpoint? Return both and leave the workspace alone.",
        {"recall"}, conversation_id=conversation,
        observer=lambda stats: record_property("turn", {"step": "recover", **stats}),
    )
    answer = messages[-1]["content"]
    summary_through = db.get_conversation(conversation)["summary_through_seq"]
    unchanged = snapshot(env.workspace) == before
    record_property("evidence", {"summary_through_seq": summary_through, "workspace_unchanged": unchanged,
                                 "found_directory": path in answer, "found_checkpoint": marker in answer})
    assert path in answer and marker in answer and re.search(r"read[- ]only", answer, re.I)
    assert unchanged
    if env.fake:
        assert any(row.get("tool_name") == "recall" and marker in row["content"] for row in messages)


def test_acceptance_turn_never_approves_writes_after_timeout(env, monkeypatch):
    monkeypatch.setenv("KIWIBRO_ACCEPTANCE_TURN_TIMEOUT", "0")
    env.fake.stream_script = [call(1, "write_file", path="after-timeout.txt", content="Too late.")]
    observed = []
    with pytest.raises(pytest.fail.Exception, match="Acceptance timeout"):
        turn(env, "Write after-timeout.txt.", {"write_file"},
             allowed_paths={"after-timeout.txt"}, observer=observed.append)
    assert observed[0]["timed_out"] and not (env.workspace / "after-timeout.txt").exists()
    assert observed[0]["terminal"] == "timeout"


def test_acceptance_reports_model_error_separately_from_done_without_usage(env):
    observed = []
    env.fake.stream_script = [ChatResult(content="The verification is complete.")]
    turn(env, "Acknowledge the verification result.", set(), observer=observed.append)
    assert observed[0]["terminal"] == "done" and observed[0]["usage"] is None
    assert observed[0]["sse_error_code"] is None

    private_message = "GPU lost while opening /private/model.gguf"
    env.fake.stream_script = [LLMError("device_lost", private_message)]
    with pytest.raises(AssertionError):
        turn(env, "Acknowledge the verification result.", set(), observer=observed.append)
    assert observed[1]["terminal"] == "error" and observed[1]["usage"] is None
    assert observed[1]["sse_error_code"] == "device_lost" and private_message not in str(observed[1])


def test_acceptance_native_placement_records_managed_launch_and_fallback(caplog):
    logger = logging.getLogger("kiwibro.engine")
    with caplog.at_level(logging.INFO, logger=logger.name):
        starting = "direct engine: starting %s (%s, ctx %d, %d slots, %d threads%s)"
        logger.info(starting, NATIVE_MODEL, "/private/runtime", 8192, 1, 6, ", GPU Radeon")
        logger.warning("direct engine: graphics device lost for %s, using CPU for this session", NATIVE_MODEL)
        logger.info(starting, "unrelated-model", "/private/runtime", 8192, 1, 6, "")
        logger.info(starting, NATIVE_MODEL, "/private/runtime", 8192, 1, 6, "")
    assert native_transitions(caplog.records) == {
        "launches": [
            {"backend": "gpu", "context": 8192, "slots": 1, "threads": 6},
            {"backend": "cpu", "context": 8192, "slots": 1, "threads": 6},
        ],
        "gpu_fallbacks": 1,
    }


def test_acceptance_native_cpu_setting_applies_before_model_launch(request, tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    pack_at(staged)
    monkeypatch.setenv(NATIVE_DATA, str(staged))
    monkeypatch.setenv(PLACEMENT, "cpu")
    monkeypatch.setenv("KIWIBRO_LLAMA_SERVER", sys.executable)
    monkeypatch.delenv("KIWIBRO_ACCEPTANCE_ENDPOINT", raising=False)

    def unexpected_launch(*_args, **_kwargs):
        pytest.fail("Selecting a backend launched a model before the first turn.")

    monkeypatch.setattr("kiwibro.engine.DirectEngine._launch", unexpected_launch)
    scenario = request.getfixturevalue("acceptance")
    assert scenario.svc.settings_provider()["direct_gpu"] == "off"
    assert managed_backend(scenario) is None
