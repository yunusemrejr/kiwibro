import json

import pytest

from kiwibro import components, tools, visual_action
from kiwibro.sandbox import Workspace


KIT = components.HINGED_PANEL
BINDINGS = {
    "filename": "guard.html",
    "edge": "bottom",
    "direction": "away",
    "angle": 57,
    "part": "Vent <guard>",
    "toggle_id": "guard-control",
    "closed_label": "Guard latched",
}


def _call(args, *, name="component", kind="function"):
    return {
        "id": "choice1",
        "type": kind,
        "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
    }


def test_selector_has_no_model_authored_slot():
    defs = visual_action.hinged_tools(BINDINGS)
    assert len(defs) == 1
    definition = defs[0]
    assert definition["type"] == "function"
    assert definition["function"]["name"] == "component"
    assert definition["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["use", "show"]},
            "name": {"type": "string", "enum": [KIT]},
        },
        "required": ["action", "name"],
        "additionalProperties": False,
    }
    assert defs == visual_action.hinged_tools({"filename": "flap.html", "edge": "left"})


def test_use_reinserts_bound_values_for_the_existing_writer(tmp_path):
    result = visual_action.hinged_action(BINDINGS, [_call({"action": "use", "name": KIT})])
    assert result == {"action": "use", "name": KIT, "values": {**BINDINGS, "angle": "57"}}
    assert BINDINGS["angle"] == 57
    args = tools.validate_args("component", result)
    ctx = tools.ToolContext(Workspace(tmp_path), None, None, "")
    assert tools.execute_tool("component", args, ctx).ok
    html = (tmp_path / "guard.html").read_text(encoding="utf-8")
    assert 'data-edge="bottom" data-direction="away"' in html
    assert 'style="--swing: 57deg"' in html
    assert 'id="guard-control"' in html
    assert "Guard latched" in html
    assert "Vent &lt;guard&gt;" in html
    assert len(list(tmp_path.iterdir())) == 1


def test_show_is_the_existing_read_only_component_action(tmp_path):
    result = visual_action.hinged_action(BINDINGS, [_call({"action": "show", "name": KIT})])
    assert result == {"action": "show", "name": KIT}
    ctx = tools.ToolContext(Workspace(tmp_path), None, None, "")
    assert tools.execute_tool("component", tools.validate_args("component", result), ctx).ok
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bindings", [
    {},
    {"filename": "guard.html"},
    {"edge": "bottom"},
    {**BINDINGS, "filename": "../guard.html"},
    {**BINDINGS, "edge": "middle"},
    {**BINDINGS, "angle": True},
    {**BINDINGS, "angle": 89},
    {**BINDINGS, "scene_id": "frame"},
    {**BINDINGS, "title": "two\nlines"},
    {**BINDINGS, "title": "x" * 161},
    {**BINDINGS, "script": "unrecognized slot"},
])
def test_invalid_user_bindings_abstain_before_or_after_generation(bindings):
    assert visual_action.hinged_tools(bindings) is None
    assert visual_action.hinged_action(bindings, [_call({"action": "use", "name": KIT})]) is None


@pytest.mark.parametrize("calls", [
    [],
    [_call({"action": "use", "name": KIT}), _call({"action": "show", "name": KIT})],
    [_call({"action": "use", "name": KIT}, name="write_file")],
    [_call({"action": "use", "name": KIT}, kind="other")],
    [_call({"action": "use", "name": "web/scene-3d"})],
    [_call({"action": "delete", "name": KIT})],
    [_call({"action": "use", "name": KIT, "path": "elsewhere"})],
    [_call({"action": "use", "name": KIT, "values": {"filename": "other.html"}})],
    [_call({"action": "show", "name": KIT, "query": "other"})],
    [_call({"action": "use"})],
    [_call("[]")],
    [_call('{"action": "use"')],
    [_call('{"action":"use","action":"show","name":"web/hinged-panel"}')],
])
def test_unrecognized_or_ambiguous_model_choice_abstains(calls):
    assert visual_action.hinged_action(BINDINGS, calls) is None
