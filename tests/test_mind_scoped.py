"""Kiwi Mind: scoped skills, the action unit, and what the conductor and the pack do with them.

The engine and the host are checked by tests/mind_scoped.mjs on the tiny random model (see mind_engine.mjs); this
file prepares what that script compares with (a NumPy forward pass that gives every position its own skill) and runs
it. The rest is plain Python: the host's command line, the conductor's rules and the pack's ``open_at``.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from mind_pack import ROOT, load_formats, make_pack, node_path
from test_mind_service import HELPER, FakeClient, Helper, ask

from kiwibro.llm import ChatResult, LLMClient
from kiwibro.mind import KIWIMIND
from kiwibro.mind.conductor import Conductor, Settings, from_kiwibro, unit_reading

SCRIPT = Path(__file__).resolve().parent / "mind_scoped.mjs"
needs_node = pytest.mark.skipif(node_path() is None, reason="node is not installed")

UNIT_NAMES = ["act", "tool:read", "tool:write"]
TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {}}}]
SYSTEM = "<|im_start|>system\nyou are a helpful tool user<|im_end|>\n"
TURN_ONE = "<|im_start|>user\nwhat is the weather today?<|im_end|>\n<|im_start|>assistant\n"
TURN_TWO = ("hello world!<|im_end|>\n<|im_start|>user\nthe quick brown fox jumps over the lazy dog.<|im_end|>\n"
            "<|im_start|>assistant\n")


# ------------------------------------------------------------ a forward pass with one skill per position


def load_reference() -> Any:
    """scripts/mind/reference.py (it imports formats from its own folder, which is put on the path only while it loads)."""
    folder = str(ROOT / "scripts" / "mind")
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location("kiwimind_reference_for_tests", Path(folder) / "reference.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(folder)
        sys.modules.pop("formats", None)
    return module


class Rows:
    """The low-rank change of one matrix for the positions that run under a skill: [(row indices, (a, b, scale))]."""

    def __init__(self, groups: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, float]]]) -> None:
        self.groups = groups


class PerPosition:
    """What Reference.forward reads as the deltas of a skill, but with a skill (or none) for every position."""

    def __init__(self, names: list[str | None], deltas: dict[str, dict]) -> None:
        self.rows = {name: np.array([i for i, n in enumerate(names) if n == name]) for name in set(names) if name}
        self.deltas = deltas

    def __bool__(self) -> bool:
        return True

    def get(self, key: tuple[int, str]) -> Rows | None:
        groups = [(rows, self.deltas[name][key]) for name, rows in self.rows.items() if key in self.deltas[name]]
        return Rows(groups) if groups else None


def per_position_logits(directory: Path, tokens: list[int], names: list[str | None]) -> list[float]:
    """The last-position logits when position i runs its upper layers under skill names[i] (None: no skill)."""
    reference = load_reference()
    formats = reference.formats

    class Rowwise(reference.Reference):
        def linear(self, x: np.ndarray, w: np.ndarray, typ: int, delta: Any = None) -> np.ndarray:
            if not isinstance(delta, Rows):
                return super().linear(x, w, typ, delta)
            xin = reference.quantize_act(x) if self.emulate and typ != formats.F32 else x
            y = xin.astype(np.float64) @ w.astype(np.float64).T
            for rows, (a, b, scale) in delta.groups:
                y[rows] += scale * ((x[rows].astype(np.float64) @ a.astype(np.float64).T) @ b.astype(np.float64).T)
            return y.astype(np.float32)

    model = Rowwise(directory / "tiny.fkm")
    deltas = {name: formats.skill_deltas(formats.read_skill(directory / f"{name}.kms"), model.config)
              for name in set(names) if name}
    with mock.patch.object(formats, "skill_deltas", lambda skill, config: skill):
        return model.forward(tokens, PerPosition(names, deltas))["logits"].tolist()


def prepare(directory: Path) -> dict[str, Any]:
    """The tiny set, an action unit, and plan.json for mind_scoped.mjs: the prompts as the host will see them and
    the logits the reference gives for the skill layout of every case."""
    formats = load_formats()
    formats.write_tiny_set(directory)
    formats.write_head(directory / "units.kmh", *formats.random_head(formats.TINY, UNIT_NAMES, hidden=16, seed=77),
                       UNIT_NAMES)
    formats.write_head(directory / "units_narrow.kmh", *formats.random_head({"dim": 32}, UNIT_NAMES, hidden=16, seed=8),
                       UNIT_NAMES)  # made for a state of 32 values, not the trunk's 64
    unit = formats.read_head(directory / "units.kmh")
    tokenizer = json.loads((directory / "tokenizer.json").read_text())
    one = formats.tiny_encode(tokenizer, SYSTEM + TURN_ONE)
    two = formats.tiny_encode(tokenizer, SYSTEM + TURN_ONE + TURN_TWO)
    system_end = one.index(tokenizer["added_tokens"][2]["id"]) + 2  # the closing token and the line break after it
    common = next(i for i, (a, b) in enumerate(zip(one, two)) if a != b) if one != two[:len(one)] else len(one)
    assert common == len(one), "the first turn must be a prefix of the second for the tiny tokenizer"
    cases = {
        "scoped, first turn": (one, [None] * system_end + ["alpha"] * (len(one) - system_end)),
        "scoped, second turn under another skill": (
            two, [None] * system_end + ["alpha"] * (len(one) - system_end) + ["beta"] * (len(two) - len(one))),
        "unscoped, first turn": (one, ["alpha"] * len(one)),
        "unscoped, second turn": (two, ["beta"] * len(two)),
    }
    plan = {
        "prompt_one": SYSTEM + TURN_ONE, "prompt_two": SYSTEM + TURN_ONE + TURN_TWO, "tokens_one": one,
        "tokens_two": two, "system_end": system_end,
        "cases": {name: {"tokens": tokens, "skills": names, "logits": per_position_logits(directory, tokens, names)}
                  for name, (tokens, names) in cases.items()},
        "unit": {"names": unit["names"], "eps": float(unit["eps"]), "w1": unit["w1"].tolist(), "b1": unit["b1"].tolist(),
                 "w2": unit["w2"].tolist(), "b2": unit["b2"].tolist()},
    }
    (directory / "plan.json").write_text(json.dumps(plan))
    return plan


def test_the_per_position_reference_tells_the_layouts_apart(tmp_path: Path) -> None:
    """The engine checks only mean something if a wrong skill layout would give clearly other logits."""
    logits = {name: np.array(case["logits"]) for name, case in prepare(tmp_path)["cases"].items()}
    scale = np.abs(logits["scoped, second turn under another skill"]).max()
    for other in ("unscoped, second turn",):
        gap = np.abs(logits["scoped, second turn under another skill"] - logits[other]).max()
        assert gap > 5 * 0.02 * scale, f"{other} is too close to the scoped layout ({gap / scale:.4f} of the peak)"
    gap = np.abs(logits["scoped, first turn"] - logits["unscoped, first turn"]).max()
    assert gap > 5 * 0.02 * np.abs(logits["scoped, first turn"]).max()


@needs_node
def test_scoped_skills_and_the_action_unit_in_the_engine_and_the_host(tmp_path: Path) -> None:
    prepare(tmp_path)
    result = subprocess.run([node_path() or "node", str(SCRIPT), str(tmp_path)], capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


# ------------------------------------------------------------------------ host.py: the command line


def add_units(pack: Path, *, scoped: bool | None) -> None:
    """What a pack with an action unit adds: units.kmh listed in the manifest, and the "scoped" key."""
    formats = load_formats()
    formats.write_head(pack / "units.kmh", *formats.random_head(formats.TINY, UNIT_NAMES, hidden=16, seed=5), UNIT_NAMES)
    manifest = json.loads((pack / "manifest.json").read_text())
    data = (pack / "units.kmh").read_bytes()
    manifest["files"]["units.kmh"] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if scoped is not None:
        manifest["scoped"] = scoped
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1))


@needs_node
@pytest.mark.parametrize(("units", "scoped"), [
    (False, None), (True, None), (True, False), (True, True), (False, True),
    (True, "yes"), (True, 1),  # only the value true switches it on
])
def test_the_host_command_line_follows_the_pack(tmp_path: Path, units: bool, scoped: Any) -> None:
    pack = make_pack(tmp_path)
    if units:
        add_units(pack, scoped=scoped)
    elif scoped is not None:
        manifest = json.loads((pack / "manifest.json").read_text())
        manifest["scoped"] = scoped
        (pack / "manifest.json").write_text(json.dumps(manifest))
    mind = LLMClient(lambda: {}, data_dir=tmp_path).mind()
    try:
        assert mind.usable(), mind.reason()
        answer = mind.client.control("hello there", "t")
        pid = mind.host.describe()["pid"]
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        if units:
            assert argv[argv.index("--units") + 1] == str(pack / "units.kmh")
            assert answer["units"]["names"] == UNIT_NAMES and len(answer["units"]["logits"]) == len(UNIT_NAMES)
        else:
            assert "--units" not in argv and answer["units"] is None
        if scoped is True:
            assert argv[argv.index("--scoped") + 1] == "1"
        else:
            assert "--scoped" not in argv
        assert answer["scoped"] is (scoped is True)
    finally:
        mind.stop()


@needs_node
def test_a_units_file_the_manifest_does_not_list_is_left_out(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    formats = load_formats()  # left over from another pack: it is not among the files that were checked
    formats.write_head(pack / "units.kmh", *formats.random_head(formats.TINY, UNIT_NAMES, hidden=16, seed=5), UNIT_NAMES)
    mind = LLMClient(lambda: {}, data_dir=tmp_path).mind()
    try:
        assert mind.usable(), mind.reason()
        mind.client.control("hello there", "t")
        argv = Path(f"/proc/{mind.host.describe()['pid']}/cmdline").read_bytes().decode().split("\0")
        assert "--units" not in argv
    finally:
        mind.stop()


# ----------------------------------------------------------------------------------- the pack: open_at


@pytest.mark.parametrize(("given", "expected"), [
    (0.9, 0.9), (0.5, 0.5), (1, 1.0), ("absent", None), (None, None), (True, None), (False, None), ("0.9", None),
    (0.2, None), (0.49, None), (3, None), (1.01, None), (float("nan"), None), ([0.9], None),
])
def test_open_at_is_a_number_from_half_to_one_or_nothing(tmp_path: Path, given: Any, expected: float | None) -> None:
    pack = make_pack(tmp_path)
    manifest = json.loads((pack / "manifest.json").read_text())
    if given != "absent":
        manifest["open_at"] = given
    (pack / "manifest.json").write_text(json.dumps(manifest))
    mind = LLMClient(lambda: {}, data_dir=tmp_path).mind()
    assert mind.open_at() == expected
    assert mind.conductor.cfg.act_mark() == expected  # the conductor asks the pack, not a copy made at start
    if expected is not None:
        assert isinstance(mind.open_at(), float)


# --------------------------------------------------------------------------------------- the conductor


class UnitClient(FakeClient):
    """A host with the action unit; whether it runs scoped skills is set by the test."""

    def __init__(self) -> None:
        super().__init__()
        self.scoped = False
        self.units: dict[str, Any] | None = None
        self.reading(0.5)

    def reading(self, act: float, tool: str = "write") -> None:
        logits = [math.log(act / (1 - act)), 0.0, 0.0]
        logits[UNIT_NAMES.index(f"tool:{tool}")] = 2.0
        self.units = {"names": UNIT_NAMES, "logits": logits}

    def control(self, prompt: str, slot: str) -> dict[str, Any]:
        return {**super().control(prompt, slot), "units": self.units, "scoped": self.scoped}


CALL = ['{"name": "write", ', '"arguments": {}}', "\n</tool_call>"]
def unit_conductor(client: FakeClient, mark: float | None = 0.9, helper: Any = None) -> Conductor:
    return Conductor(client, Settings(lambda: 1000, lambda: HELPER if helper else None, helper, lambda: mark))  # type: ignore[arg-type]


STEP = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                                                              "function": {"name": "write", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "done"}]
NEXT = [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "and now?"}]


def test_from_kiwibro_recognises_the_apps_own_notes() -> None:
    assert from_kiwibro({"role": "user", "content": "[Kiwibro: that is 4 rounds of searching.]"})
    assert from_kiwibro({"role": "user", "content": "[Kiwibro guard: a secret was hidden]"})
    assert not from_kiwibro({"role": "user", "content": "kiwibro, please write a poem"})
    assert not from_kiwibro({"role": "user", "content": " [Kiwibro: not at the start]"})
    assert not from_kiwibro({"role": "user", "content": [{"type": "text", "text": "[Kiwibro: x]"}]})
    assert not from_kiwibro({"role": "user"})


def test_a_note_of_the_app_in_the_middle_of_a_turn_does_not_ask_the_head_again() -> None:
    client = UnitClient()
    client.logits["terminal"] = 3.0
    cond = unit_conductor(client)
    ask(cond)
    ask(cond, extra_messages=[*STEP, {"role": "user", "content": "[Kiwibro: the last 3 tool calls failed.]"}])
    assert client.control_calls == 1
    client.logits.update({"terminal": 0.0, "web": 4.0})
    ask(cond, extra_messages=[*STEP, {"role": "user", "content": "[Kiwibro: x]"}, *NEXT])  # a real message follows
    assert client.control_calls == 2 and client.bodies[-1]["skill"] == "web"


@pytest.mark.parametrize(("scoped", "second"), [(False, "code"), (True, "web")])
def test_a_scoped_host_lets_the_head_change_skill_at_once(scoped: bool, second: str) -> None:
    """Unscoped, a change costs the upper cache, so a small lead is not enough; scoped, it costs nothing."""
    client = UnitClient()
    client.scoped = scoped
    cond = unit_conductor(client)
    client.logits["code"] = 3.0
    ask(cond, "write a function")
    client.logits.update({"code": 2.0, "web": 2.3})  # a lead of less than the margin
    ask(cond, "x", extra_messages=NEXT)
    assert [b["skill"] for b in client.bodies] == ["code", second]


@pytest.mark.parametrize(("units", "expected"), [
    ({"names": UNIT_NAMES, "logits": [0.0, 1.0, 3.0]}, (0.5, "write")),
    ({"names": UNIT_NAMES, "logits": [math.log(9), 5.0, 1.0]}, (0.9, "read")),
    ({"names": ["tool:b", "act", "tool:a"], "logits": [1.0, 0.0, 2.0]}, (0.5, "a")),  # order of the names is free
    ({"names": ["act"], "logits": [0.0]}, (0.5, "")),
    ({"names": ["tool:read"], "logits": [1.0]}, (None, "")),  # no "act" output: no reading
    ({"names": UNIT_NAMES, "logits": [0.0, 1.0]}, (None, "")),  # names and logits do not fit
    ({}, (None, "")), (None, (None, "")),
])
def test_unit_reading_is_the_chance_of_a_call_and_the_best_tool(units: Any, expected: tuple[float | None, str]) -> None:
    act, tool = unit_reading(units)
    assert tool == expected[1]
    assert (act is None) == (expected[0] is None) and (act is None or act == pytest.approx(expected[0]))


def test_unit_reading_survives_extreme_outputs() -> None:
    high, _ = unit_reading({"names": ["act"], "logits": [1e6]})
    low, _ = unit_reading({"names": ["act"], "logits": [-1e6]})
    assert high is not None and low is not None and high > 0.999999 and 0 < low < 1e-9


def test_the_reading_belongs_to_the_first_reply_of_a_turn() -> None:
    client = UnitClient()
    client.reading(0.8, "read")
    cond = unit_conductor(client, mark=None)
    assert cond.reading("c1") == (None, "")  # nothing asked yet
    events: list = []
    ask(cond, events=events)
    assert events[0][1]["act"] == 0.8 and events[0][1]["tool"] == "read"
    steps: list = []
    ask(cond, events=steps, extra_messages=STEP)  # the next reply of the same turn: no reading, no new head call
    assert steps == [("brain", {"brain": "kiwimind", "skill": "chat", "hard": False})]
    assert cond.reading("c1") == (None, "")
    client.reading(0.2, "write")
    later: list = []
    ask(cond, events=later, extra_messages=[*STEP, *NEXT])  # a new user message: read again
    assert later[0][1]["act"] == 0.2 and later[0][1]["tool"] == "write"


def test_a_host_without_the_unit_changes_nothing_in_the_events() -> None:
    client = UnitClient()
    client.units = None
    events: list = []
    ask(unit_conductor(client, mark=0.5), events=events, tools=TOOLS)
    assert events == [("brain", {"brain": "kiwimind", "skill": "chat", "hard": False})]
    assert not client.bodies[0]["prompt"].endswith("<tool_call>\n")


def test_a_sure_unit_opens_the_reply_as_a_tool_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIWIBRO_MIND_TRACE", str(tmp_path / "trace.jsonl"))
    client = UnitClient()
    client.reading(0.95, "write")
    client.pieces = CALL
    events: list = []
    shown: list[str] = []
    result = ask(unit_conductor(client), events=events, on_delta=shown.append, tools=TOOLS)
    assert client.bodies[0]["prompt"].endswith("<tool_call>\n")  # the model continues inside the call
    assert result.tool_calls and result.tool_calls[0]["function"]["name"] == "write" and shown == []
    assert [event for event in events if event[0] != "status"] == [("brain", {"brain": "kiwimind", "skill": "chat", "hard": False, "act": 0.95, "tool": "write"})]
    record = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
    assert record["opened"] is True and record["prompt"].endswith("<tool_call>\n") and record["reply"].startswith("<tool_call>\n")


@pytest.mark.parametrize(("act", "mark", "tools", "opened"), [
    (0.95, 0.9, TOOLS, True), (0.9, "same", TOOLS, True),  # a reading that is exactly the mark opens
    (0.89, 0.9, TOOLS, False),  # below the mark
    (0.99, None, TOOLS, False),  # the pack names no mark
    (0.99, 0.9, None, False), (0.99, 0.9, [], False),  # no tools offered: nothing to call
])
def test_the_reply_is_opened_only_at_the_mark_and_with_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, act: float,
                                                             mark: float | None, tools: Any, opened: bool) -> None:
    monkeypatch.setenv("KIWIBRO_MIND_TRACE", str(tmp_path / "trace.jsonl"))
    client = UnitClient()
    client.reading(act)
    if mark == "same":  # the sigmoid of a logit is not exactly the number it was made from
        mark = unit_reading(client.units)[0]
    ask(unit_conductor(client, mark=mark), tools=tools)
    assert client.bodies[0]["prompt"].endswith("<tool_call>\n") is opened
    assert json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])["opened"] is opened


def test_only_the_first_reply_of_a_turn_can_be_opened() -> None:
    client = UnitClient()
    client.reading(0.99)
    client.pieces = CALL
    cond = unit_conductor(client)
    ask(cond, tools=TOOLS)
    client.pieces = ["Done."]
    ask(cond, tools=TOOLS, extra_messages=STEP)
    assert client.bodies[0]["prompt"].endswith("<tool_call>\n") and not client.bodies[1]["prompt"].endswith("<tool_call>\n")


def test_an_opened_reply_does_not_go_to_the_helper() -> None:
    client = UnitClient()
    client.logits["hard"] = 2.0
    client.reading(0.99)
    client.pieces = CALL
    helper = Helper()
    result = ask(unit_conductor(client, helper=helper), tools=TOOLS)
    assert helper.calls == [] and result.tool_calls[0]["function"]["name"] == "write"
    # below the mark the same hard question still goes to the helper
    client.reading(0.6)
    client.pieces = [f"word{i} " for i in range(30)]
    result = ask(unit_conductor(client, helper=helper), tools=TOOLS, slot="c2")
    assert len(helper.calls) == 1 and result.content == "helper words"


@pytest.mark.parametrize(("tools", "open_call", "opened"), [
    (TOOLS, True, True), (TOOLS, False, False), (None, True, False), ([], True, False),
])
def test_open_call_opens_the_reply_without_a_reading_or_a_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tools: Any, open_call: bool, opened: bool,
) -> None:
    monkeypatch.setenv("KIWIBRO_MIND_TRACE", str(tmp_path / "trace.jsonl"))
    client = UnitClient()
    client.units = None  # the host has no action unit
    client.pieces = CALL
    events: list = []
    result = ask(unit_conductor(client, mark=None), events=events, tools=tools, open_call=open_call)
    assert client.bodies[0]["prompt"].endswith("<tool_call>\n") is opened
    assert json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])["opened"] is opened
    assert [event for event in events if event[0] != "status"] == [("brain", {"brain": "kiwimind", "skill": "chat", "hard": False})]
    assert bool(result.tool_calls) is opened


def test_open_call_opens_a_reply_the_unit_is_not_sure_of_and_the_helper_is_not_asked() -> None:
    client = UnitClient()
    client.logits["hard"] = 2.0
    client.reading(0.1)  # the unit expects words
    client.pieces = CALL
    helper = Helper()
    result = ask(unit_conductor(client, mark=0.9, helper=helper), tools=TOOLS, open_call=True)
    assert client.bodies[0]["prompt"].endswith("<tool_call>\n")
    assert helper.calls == [] and result.tool_calls[0]["function"]["name"] == "write"


def test_the_client_hands_open_call_to_the_conductor(tmp_path: Path) -> None:
    llm = LLMClient(lambda: {}, data_dir=tmp_path)
    seen: list[dict[str, Any]] = []
    conductor = llm.mind().conductor
    conductor.stream = lambda **kwargs: seen.append(kwargs) or ChatResult(content="ok")  # type: ignore[method-assign]
    messages = [{"role": "user", "content": "hi"}]
    llm.stream_chat(model=KIWIMIND, messages=messages, tools=TOOLS, open_call=True)
    llm.stream_chat(model=KIWIMIND, messages=messages, tools=TOOLS)
    assert [call["open_call"] for call in seen] == [True, False]
