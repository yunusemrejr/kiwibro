"""Restrict a native tool choice to a hinged panel with explicit user bindings."""

from __future__ import annotations

import json
from typing import Any

from . import components

_REQUIRED_BINDINGS = frozenset({"filename", "edge"})
_MAX_ARGUMENT_CHARS = 1024


def _checked_bindings(bindings: dict[str, str | int]) -> dict[str, str] | None:
    if not isinstance(bindings, dict) or not _REQUIRED_BINDINGS <= bindings.keys():
        return None
    try:
        kit = components.default_library().get(components.HINGED_PANEL)
        if kit.kind != components.KIT or not kit.finished or bindings.keys() - kit.blanks.keys():
            return None
        values = {}
        for key in kit.blanks:
            if key not in bindings:
                continue
            value = bindings[key]
            if not isinstance(value, str) and not (key == "angle" and type(value) is int):
                return None
            values[key] = str(value)
        # Check the same effective defaults and constraints as the writer. Keep the raw bindings
        # since the writer escapes user-facing labels when it fills the HTML.
        components._hinge_values(kit, values)
        return values
    except (KeyError, OSError, TypeError, ValueError):
        return None


def hinged_tools(bindings: dict[str, str | int]) -> list[dict[str, Any]] | None:
    """Return one narrow component tool when the provided user values fit the shipped kit.

    It offers `use` and `show` with no value slots. Its caller must run
    :func:`hinged_action` before dispatch to attach the checked values to `use`.
    """
    if _checked_bindings(bindings) is None:
        return None
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["use", "show"]},
            "name": {"type": "string", "enum": [components.HINGED_PANEL]},
        },
        "required": ["action", "name"],
        "additionalProperties": False,
    }
    return [{"type": "function", "function": {
        "name": "component",
        "description": ("Use the hinged panel for a single rectangular leaf swinging on a fixed frame. "
                        "Choose show if the kit does not fit. Show does not write a file."),
        "parameters": parameters,
    }}]


def _unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate tool argument")
        result[key] = value
    return result


def hinged_action(
    bindings: dict[str, str | int], tool_calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Accept a single exact native choice and return ordinary `component` arguments.

    This helper has no effects. The caller still applies the regular tool gates.
    """
    values = _checked_bindings(bindings)
    if values is None or not isinstance(tool_calls, list) or len(tool_calls) != 1:
        return None
    try:
        call = tool_calls[0]
        function = call["function"]
        payload = function["arguments"]
        if (call["type"] != "function" or function["name"] != "component"
                or not isinstance(payload, str) or len(payload) > _MAX_ARGUMENT_CHARS):
            return None
        args = json.loads(payload, object_pairs_hook=_unique_keys)
    except (KeyError, RecursionError, TypeError, ValueError):
        return None
    if not isinstance(args, dict) or args.keys() != {"action", "name"} or args["name"] != components.HINGED_PANEL:
        return None
    if args["action"] == "show":
        return {"action": "show", "name": components.HINGED_PANEL}
    if args["action"] != "use":
        return None
    return {"action": "use", "name": components.HINGED_PANEL, "values": values}
