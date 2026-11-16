"""How Kiwi Mind 3 reads a conversation.

Kiwi Mind 3 was trained on exactly this rendering of a conversation (the training script loads
this file), so a change here needs retrained layers. It imports nothing from Kiwibro but its
sibling ``render.py``, whose system text, tool manual and tool-call form it keeps.

The turns are marked with the role tokens of the base model, Spark X2.6. Every reply opens with
a route in the thinking channel: ``<think>code light act</think>`` names the skill that answers,
says whether the request is light or heavy, and whether the reply says something or acts (opens
with a tool call). The route layer is switched on at ``<think>`` and the skill's layer at
``</think>``; everything else is read by the trunk alone, so a change of skill re-reads nothing.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Sequence

try:
    from . import render as _v2
except ImportError:  # loaded by path, as the training script does
    _spec = importlib.util.spec_from_file_location("kiwimind_render", Path(__file__).with_name("render.py"))
    _v2 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_v2)

BOS = "<｜start▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
SYSTEM_ROLE, USER, TOOL, BOT = "<|System|>", "<|User|>", "<|Tool|>", "<|Bot|>"
THINK, THINK_END = "<think>", "</think>"
MEDIA = "<__media__>"  # where llama.cpp puts the embeddings of one image
STOP = (EOS, BOS)

SKILLS = ("chat", "web", "code", "terminal", "computer", "tools", "write", "see")
PLAIN_ROUTE = "chat light say"

SYSTEM = _v2.SYSTEM
signature, system_text, call_text = _v2.signature, _v2.system_text, _v2.call_text
split_notes, parse_reply = _v2.split_notes, _v2.parse_reply

_ROLE = {"system": SYSTEM_ROLE + "\n", "user": USER, "tool": TOOL}
_ROUTE_RE = re.compile(rf"\s*({'|'.join(SKILLS)}) (light|heavy) (say|act)\s*\Z")
# Text that would read as prompt structure when it arrives inside a message or a tool result.
_STRUCTURE_RE = re.compile(
    r"<\|[^|<>\s]{1,40}\|>"  # <|User|> and the base model's other markers
    r"|<｜[^｜<>\n]{1,40}｜>"  # its sentence markers
    r"|</?think>|</?tool_(?:call|response)>|</?arg_(?:key|value)>|<__media__>"
)


def plain(text: Any) -> str:
    """Message text with anything that looks like prompt structure defused."""
    return _STRUCTURE_RE.sub(lambda found: found.group(0).replace("<", "<​"), _v2.plain(text))


def images(message: dict[str, Any]) -> int:
    """How many pictures a message carries."""
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if isinstance(part, dict) and part.get("type") in ("image_url", "image"))


def assistant_text(message: dict[str, Any]) -> str:
    """An assistant turn as the model writes it: its words, then its tool calls."""
    parts = []
    content = plain(message.get("content")).strip()
    if content:
        parts.append(content)
    for call in message.get("tool_calls") or []:
        function = call.get("function") or call
        parts.append(call_text(str(function.get("name") or ""), function.get("arguments")))
    return "\n".join(parts)


def acts(reply: str) -> bool:
    """Whether a reply opens with a tool call."""
    return reply.startswith("<tool_call>")


def route_text(skill: str, heavy: bool = False, act: bool = False) -> str:
    return f"{skill} {'heavy' if heavy else 'light'} {'act' if act else 'say'}"


def parse_route(text: str) -> tuple[str, bool, bool] | None:
    """A route as (skill, heavy, act), or None when the text is not one."""
    found = _ROUTE_RE.match(text)
    return (found.group(1), found.group(2) == "heavy", found.group(3) == "act") if found else None


def route_grammar(skill: str | None = None, *, weigh: bool = True) -> str:
    """The route as a llama.cpp grammar. A later step of the same request keeps its skill and is
    not weighed again."""
    skills = " | ".join(f'"{name}"' for name in ((skill,) if skill else SKILLS))
    weight = '"light" | "heavy"' if weigh else '"light"'
    return f'root ::= ({skills}) " " ({weight}) " " ("say" | "act")'


def turns(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    """The conversation as (role, text) turns. The results of one step share a tool turn; what
    Kiwibro wrote under them follows as a user turn."""
    workspace, summary = _v2._system_facts(messages)
    out: list[tuple[str, str]] = [("system", system_text(tools, plain(workspace), plain(summary)))]
    results: list[str] = []
    asides: list[str] = []

    def flush() -> None:
        if results:
            out.append(("tool", "\n".join(results)))
            results.clear()
        if asides:
            out.append(("user", "\n".join(asides)))
            asides.clear()

    for message in messages:
        role = message.get("role")
        if role == "tool":
            body, notes = split_notes(plain(message.get("content")))
            results.append(f"<tool_response>\n{body}\n</tool_response>")
            asides.extend(notes)
            continue
        flush()
        if role == "user":
            out.append(("user", f"{MEDIA}\n" * images(message) + plain(message.get("content"))))
        elif role == "assistant":
            out.append(("assistant", assistant_text(message)))
    flush()
    return out


def segments(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    routes: Sequence[str | None] | None = None,
    add_generation_prompt: bool = True,
) -> list[tuple[str, str]]:
    """The prompt in pieces, each named by the layer that reads it: ``read`` (the trunk alone),
    ``route`` (the route layer, from ``<think>``) and ``reply`` (the turn's skill, from
    ``</think>``). ``routes`` holds the route of each assistant turn in order; a turn whose
    route is not known reads as chat."""
    pieces: list[tuple[str, str]] = []
    known = iter(routes or ())
    reading = ""
    for role, body in turns(messages, tools):
        if role != "assistant":
            reading += f"{BOS}{_ROLE[role]}{body}{EOS}"
            continue
        route = next(known, None) or route_text("chat", act=acts(body))
        pieces += [("read", reading + BOS + BOT), ("route", THINK + route), ("reply", THINK_END + body)]
        reading = EOS
    if add_generation_prompt:
        reading += BOS + BOT
    if reading:
        pieces.append(("read", reading))
    if add_generation_prompt:
        pieces.append(("route", THINK))
    return pieces


def render(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    routes: Sequence[str | None] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """The prompt as one text. With a generation prompt it ends in ``<think>``: the route comes next."""
    pieces = segments(messages, tools, routes=routes, add_generation_prompt=add_generation_prompt)
    return "".join(text for _, text in pieces)


def render_plain(messages: list[dict[str, Any]]) -> str:
    """A request that carries its own instructions (a summary, an extraction): no tool manual,
    the caller's system text kept as it is, and the route already written, so the reply starts
    at once under the chat skill."""
    text = ""
    for message in messages:
        role = message.get("role")
        if role in _ROLE and role != "tool":
            text += f"{BOS}{_ROLE[role]}{plain(message.get('content'))}{EOS}"
        elif role == "assistant":
            text += f"{BOS}{BOT}{THINK}{PLAIN_ROUTE}{THINK_END}{plain(message.get('content'))}{EOS}"
    return text + f"{BOS}{BOT}{THINK}{PLAIN_ROUTE}{THINK_END}"
