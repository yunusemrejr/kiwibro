"""Situational guidance: the right instructions for this turn, chosen by code.

Message 0 has to stay byte-identical between turns (LM Studio reuses its KV cache), and
on a CPU every prompt token costs prefill time, so it carries only what every turn
needs. How to research, code, use the desktop, render a chart or handle memory is
picked here from the user's message (plus attachments and the turn plan) with cheap
regexes, and rides in the per-turn context note on the newest user message — present
when it applies, absent otherwise. A greeting gets nothing.
"""

from __future__ import annotations

import re
from typing import Iterable

from . import plan

# Each snippet is one short line; order is the order they appear in the note.
SNIPPETS: dict[str, str] = {
    "capabilities": (
        "What Kiwibro can do: talk and remember facts across chats; search the web and read pages; "
        "work on code and documents in the workspace (search, read, patch, run tests and commands); "
        "use this computer (open files or URLs, clipboard, notifications, screenshots, system info); "
        "analyse data with code; describe images with a helper model."
    ),
    "steps": (
        "Several steps: keep task_board current, and batch independent calls in one reply "
        "(multi_search, fetch_pages, read_files, multi_command)."
    ),
    "web": (
        "Research: use web_research (or multi_search + fetch_pages), cite evidence as [1], [2] with the "
        "URLs the tools returned, and answer once you have enough."
    ),
    "code": (
        "Code: find with glob_files / grep_files, read with read_files or file_outline, change with "
        "apply_patch, then verify by running the tests (multi_command); terminal_session for servers."
    ),
    "compute": (
        "Numbers and data: compute them with run_python or run_javascript (safe scratch folder) instead of "
        "estimating; study_file for long files."
    ),
    "desktop": (
        "Desktop: open_path shows a file or URL, clipboard reads or sets text, notify signals completion, "
        "screenshot captures the screen, system_info describes this machine."
    ),
    "memory": (
        "Memory: facts the user states are saved automatically. Call remember only to correct one "
        "(action \"update\", old = the outdated fact) or remove one (action \"forget\"); use recall to "
        "look up earlier conversations instead of asking the user to repeat themselves."
    ),
    "visual": (
        "Rich output: the chat renders markdown tables, ```csv blocks as tables, and one self-contained "
        "```html or ```svg block (inline CSS/JS, no external links) as a live preview — use it for a "
        "chart, diagram or small UI."
    ),
    "attachments": (
        "Attached files: the message starts with a note per file (its workspace path and a description by "
        "a vision model or by code). Treat it as what you see; use the path with your tools for more."
    ),
}
TOOL_SNIPPETS = frozenset({"steps", "web", "code", "compute", "desktop", "memory"})
HEADER = "## Guidance for this turn"

_CAPABILITIES = re.compile(
    r"\bwhat\s+(?:can|could)\s+you\s+do\b|\bwhat\s+are\s+you\s+(?:able|capable)\b|"
    r"\byour\s+(?:capabilities|features|tools|abilities)\b|\bwhat\s+tools\b|\bhow\s+can\s+you\s+help\b",
    re.I,
)
_CODE = re.compile(
    r"```|\b(?:code|coding|function|class|method|bug|error|exception|traceback|stack\s*trace|tests?|testing|"
    r"repo(?:sitory)?|git|commit|branch|refactor|compile|build|lint|script|endpoint|python|javascript|"
    r"typescript|rust|golang|java|c\+\+|bash|sql|regex|npm|pip|pytest|dockerfile)\b|"
    r"\b[\w-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|c|cpp|h|sh|rb|php|css|html|sql|toml|ya?ml)\b",
    re.I,
)
_WEB_EXTRA = re.compile(
    r"\b(?:current(?:ly)?|price|prices|weather|forecast|stocks?|scores?|released?|release\s+date|"
    r"who\s+won|look\s+up|20[2-3]\d)\b",
    re.I,
)
_COMPUTE = re.compile(
    r"\b(?:calculate|compute|how\s+much\s+is|average|mean|median|percent(?:age)?|statistics|"
    r"convert\s+\d|analy[sz]e\s+(?:the\s+|this\s+|my\s+)?(?:data|csv|table|spreadsheet|numbers))\b|"
    r"\d\s*[-+*/^×÷]\s*\d",
    re.I,
)
_DESKTOP = re.compile(
    r"\bopen\s+(?:the\s+|this\s+|that\s+|my\s+)?(?:file|folder|url|link|page|it|https?://)|\bclipboard\b|"
    r"\bscreenshot\b|\bnotif(?:y|ication)\b|\bsystem\s+info\b|\b(?:cpu|ram|battery|disk\s+space)\b",
    re.I,
)
_MEMORY = re.compile(
    r"\b(?:remember|forget|memory|memories|recall|last\s+time|earlier|previously|"
    r"we\s+(?:talked|discussed|spoke|said)|did\s+i\s+(?:tell|say|mention)|i\s+(?:told|said|mentioned)|"
    r"you\s+know\s+(?:that\s+)?my|what\s+do\s+you\s+know\s+about\s+me|my\s+name)\b",
    re.I,
)
_VISUAL = re.compile(
    r"\b(?:chart|graph|plot|diagram|visuali[sz]e|visuali[sz]ation|table|svg|html|mock-?up|dashboard|"
    r"flowchart|ui|widget|infographic)\b",
    re.I,
)


def select(
    message: str,
    *,
    tools: bool = True,
    attachments: bool = False,
    plan_kinds: Iterable[str] = (),
    plan_steps: int = 0,
) -> list[str]:
    """The snippet keys this turn needs, in SNIPPETS order."""
    text = message or ""
    keys: set[str] = set()
    if attachments:
        keys.add("attachments")
    if not text.strip() or plan.is_small_talk(text):
        return [k for k in SNIPPETS if k in keys]
    kinds = set(plan_kinds)
    if _CAPABILITIES.search(text):
        keys.add("capabilities")
    if plan_steps > 1:
        keys.add("steps")
    if "web" in kinds or plan._SEARCH_VERB.search(text) or plan._WEB_WORD.search(text) or _WEB_EXTRA.search(text):
        keys.add("web")
    if _CODE.search(text) or "run" in kinds:
        keys.add("code")
    if _COMPUTE.search(text):
        keys.add("compute")
    if _DESKTOP.search(text):
        keys.add("desktop")
    if _MEMORY.search(text):
        keys.add("memory")
    if _VISUAL.search(text):
        keys.add("visual")
    if not tools:
        keys -= TOOL_SNIPPETS
    return [k for k in SNIPPETS if k in keys]


def note(keys: Iterable[str]) -> str:
    lines = [f"- {SNIPPETS[k]}" for k in keys if k in SNIPPETS]
    return f"{HEADER}\n" + "\n".join(lines) if lines else ""
