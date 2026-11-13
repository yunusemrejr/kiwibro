"""How Kiwi Mind reads a conversation.

Kiwibro's own models were trained on exactly this rendering of a conversation (the training
script loads this file), so a change here needs retrained skills. It has no imports from the
rest of Kiwibro on purpose.

The prompt is ChatML with Qwen's tool tags. The system text is short and the same for every
skill, so the part of the network all skills share reads it once; the tool manual is one line
per tool (name and argument names), because the skills learned the tools themselves.
"""

from __future__ import annotations

import json
import re
from typing import Any

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM = (
    "You are Kiwibro, an AI agent that lives on the user's own computer. "
    "Use a tool when the task needs a file, a command, the web or saved memory. "
    "Answer from what the tools return and from the context note, and say plainly "
    "when something failed or is not known. Never delete anything: give the user "
    "the command to run instead."
)
STOP = (IM_END, "<|endoftext|>")

_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OPEN_CALL_RE = re.compile(r"<tool_call>\s*(\{.*)\Z", re.DOTALL)
_WORKSPACE_RE = re.compile(r"work inside the folder `([^`]+)`")
_SUMMARY_RE = re.compile(r"## Earlier conversation summary\n(.*?)(?=\n## |\Z)", re.DOTALL)
# Text that would read as prompt structure when it arrives inside a message or a tool result.
_STRUCTURE_RE = re.compile(r"<\|(im_start|im_end|endoftext)\|>|</?tool_(call|response)>")


def plain(text: Any) -> str:
    """Message text with anything that looks like prompt structure defused."""
    if isinstance(text, list):  # multimodal parts: only the words are read
        text = "\n".join(
            str(part.get("text") or "") for part in text if isinstance(part, dict) and part.get("type") == "text"
        )
    return _STRUCTURE_RE.sub(lambda found: found.group(0).replace("<", "<\u200b"), str(text or ""))


def signature(tool: dict[str, Any]) -> str:
    """One line for one tool: ``browser(action=open|list, url?, max_chars?)``."""
    function = tool.get("function") or tool
    schema = function.get("parameters") or {}
    properties = schema.get("properties") or {}
    required = [name for name in schema.get("required") or [] if name in properties]
    ordered = required + [name for name in properties if name not in required]
    parts = []
    for name in ordered:
        spec = properties.get(name) or {}
        text = name
        if spec.get("enum"):
            text += "=" + "|".join(str(choice) for choice in spec["enum"])
        elif spec.get("type") == "array":
            text += "[]"
        parts.append(text if name in required else text + "?")
    return f"{function.get('name', '')}({', '.join(parts)})"


def system_text(tools: list[dict[str, Any]] | None, workspace: str = "", summary: str = "") -> str:
    lines = [SYSTEM]
    if workspace:
        lines.append(f"Workspace: {workspace} (paths are relative to it).")
    if tools:
        lines.append("")
        lines.append("Tools:")
        lines.extend(signature(tool) for tool in tools)
    if summary:
        lines.append("")
        lines.append("Earlier in this conversation: " + summary.strip())
    return "\n".join(lines)


def call_text(name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"_raw": arguments}
    body = json.dumps({"name": name, "arguments": arguments or {}}, ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call>"


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


def _system_facts(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """The two things Kiwibro's own system message says that the model has to know."""
    if not messages or messages[0].get("role") != "system":
        return "", ""
    text = str(messages[0].get("content") or "")
    workspace = _WORKSPACE_RE.search(text)
    summary = _SUMMARY_RE.search(text)
    return (workspace.group(1) if workspace else ""), (summary.group(1).strip() if summary else "")


def turns(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    """The conversation as (role, text) turns: tool results become user turns, as Qwen reads them."""
    workspace, summary = _system_facts(messages)
    out: list[tuple[str, str]] = [("system", system_text(tools, workspace, summary))]
    results: list[str] = []

    def flush() -> None:
        if results:
            out.append(("user", "\n".join(results)))
            results.clear()

    for message in messages:
        role = message.get("role")
        if role == "tool":
            results.append(f"<tool_response>\n{plain(message.get('content'))}\n</tool_response>")
            continue
        flush()
        if role == "user":
            out.append(("user", plain(message.get("content"))))
        elif role == "assistant":
            out.append(("assistant", assistant_text(message)))
    flush()
    return out


def render(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    add_generation_prompt: bool = True,
) -> str:
    text = "".join(f"{IM_START}{role}\n{body}{IM_END}\n" for role, body in turns(messages, tools))
    return text + f"{IM_START}assistant\n" if add_generation_prompt else text


def render_plain(messages: list[dict[str, Any]]) -> str:
    """A request that carries its own instructions (a summary, an extraction): no tool manual,
    the caller's system text kept as it is."""
    text = ""
    for message in messages:
        role = message.get("role")
        if role in ("system", "user", "assistant"):
            text += f"{IM_START}{role}\n{plain(message.get('content'))}{IM_END}\n"
    return text + f"{IM_START}assistant\n"


def parse_reply(text: str) -> tuple[str, list[dict[str, Any]]]:
    """What the model wrote, split into its words and its tool calls.

    A call that does not parse stays in the text: Kiwibro's own repair reads it there."""
    calls: list[dict[str, Any]] = []

    def take(found: re.Match[str]) -> str:
        try:
            body = json.loads(found.group(1))
        except ValueError:
            return found.group(0)
        if not isinstance(body, dict) or not isinstance(body.get("name"), str):
            return found.group(0)
        arguments = body.get("arguments")
        calls.append({"name": body["name"], "arguments": arguments if isinstance(arguments, dict) else {}})
        return ""

    rest = _CALL_RE.sub(take, text)
    rest = _OPEN_CALL_RE.sub(take, rest)  # a last call the model did not close
    return rest.strip(), calls
