"""Turn plan: split a multi-step request into steps, keep the model on them, verify, stop.

"Tell me what X is, then search the web for Y, and finally write a doc on my Desktop"
is three jobs. Small models given it often do one or two and stop, write the file into
the workspace instead of the folder the user named, keep searching after the work is
done, or say they saved a file they never wrote. The plan handles that in code, with
no model calls:

* :func:`build` splits the newest user message into ordered steps (sequencing words,
  sentences, list items; a fragment that is not an instruction stays with its step)
  and classifies each one: answer, web, write, run or read. A write or read step gets
  its target from the message: a path (``~/Desktop/notes/pi.md``), a place ("on my
  desktop", XDG-aware) plus a file name, or just a place.
* :meth:`TurnPlan.note` joins the per-turn context note (KV-safe) when the request has
  several steps or names a place outside the workspace.
* :meth:`TurnPlan.route` lets file tools reach a file or folder the user named in this
  message although it is outside the workspace: that one call runs in a workspace
  rooted at the named folder, so the usual confinement and safety grading apply there.
  Places a conversation may never work in (home itself, credentials, start-up files,
  system folders, Kiwibro's data) stay out of reach.
* :meth:`TurnPlan.observe` ticks steps off from successful tool results; the progress
  line it returns is appended to the tool result (stored, so the KV prefix stays
  valid) and tells the model what is next, or that only the reply is left.
* :meth:`TurnPlan.unmet` verifies at the end: files exist where the user asked and are
  not empty, web steps searched, an explain-step's topic is in the reply.
  :meth:`TurnPlan.followup` turns what is left into one corrective nudge.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, directives, safety, workspaces

MAX_STEPS = 8
MAX_FOLLOWUPS = 2  # corrective rounds per turn when the model stops early
STEP_CHARS = 150
MIN_DOC_BYTES = 40  # a document, report or summary shorter than this is not the deliverable

KINDS = ("answer", "web", "write", "run", "read", "other")  # other: not checkable in code
WEB_TOOLS = frozenset({"web_search", "web_research", "multi_search", "fetch_page", "fetch_pages", "browse_page", "browser"})
WRITE_TOOLS = frozenset({"write_file", "replace_in_file", "apply_patch", "convert_document"})
RUN_TOOLS = frozenset({"run_command", "multi_command", "terminal_session", "run_python", "run_javascript"})
READ_TOOLS = frozenset({"read_file", "read_files", "list_files", "file_outline", "study_file", "doc_read", "doc_search",
                        "analyze_table", "describe_image", "grep_files", "glob_files"})
# File tools whose `path` may be routed to a place the user named outside the workspace.
ROUTED_TOOLS = frozenset({"read_file", "list_files", "file_outline", "study_file", "analyze_table", "describe_image",
                          "write_file", "replace_in_file", "apply_patch"})
_WRITES = frozenset({"write_file", "replace_in_file", "apply_patch"})
# Research pacing: small models keep searching near-identical queries ("let me get the
# details, then write it") until the round limit while every round re-prefills more results.
SEARCH_TOOLS = frozenset({"web_search", "web_research", "multi_search"})
SEARCH_ROUNDS_SOFT = 3  # rounds with searches before the "you have enough" note
SEARCH_ROUNDS_MAX = 5  # rounds with searches before searching is refused for the turn
SEARCH_ROUNDS_AFTER_PLAN = 1  # extra search rounds once every planned tool step is done
REPEAT_JACCARD = 0.6
_TOOL_HINT = {"web": "web_search", "write": "write_file", "run": "run_command", "read": "read_file"}

_CODE = re.compile(r"```.*?```", re.S)
# Greetings and thanks only: "ok" or "great" often means "go ahead" and needs the tools.
_SMALL_TALK = re.compile(
    r"^(?:hi|hello|hey|yo|hiya|thanks?(?:\s+(?:you|a lot|so much))?|thx|ty|cheers|good\s+(?:morning|afternoon|evening|night)|"
    r"bye|goodbye|see\s+you|how\s+are\s+you(?:\s+doing)?|what'?s\s+up)[\s.!?,]*(?:kiwi(?:bro)?)?[\s.!?]*$",
    re.I,
)
# Over-splitting is safe: a fragment that is not an instruction is glued back on.
_SPLIT = re.compile(
    r"(\n+|(?<=[.!?])\s+|\s*;\s*"
    r"|,?\s+(?:and\s+)?(?:then|after\s+that|afterwards|next|finally|lastly)\b,?\s*"
    r"|,\s+and\s+|\s+and\s+(?:also\s+)?(?=\w+\s))",
    re.I,
)
_MARKER = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•]|step\s*\d+\s*[:.)-]?)\s*", re.I)
_LEAD = re.compile(
    r"^(?:(?:hey|hi|hello|ok(?:ay)?|so|and|then|also|now|next|finally|lastly|after\s+that|afterwards|"
    r"first(?:ly)?|second(?:ly)?|third(?:ly)?|please|pls|kiwi(?:bro)?)\b[\s,.:!-]*)+",
    re.I,
)
# "Can you…", "I want you to…": how a request is asked, not what it asks for.
_ASK = re.compile(
    r"^(?:(?:can|could|would|will)\s+you(?:\s+please)?|i\s+(?:want|need|would\s+like|'d\s+like)(?:\s+you)?\s+to|"
    r"help\s+me(?:\s+to)?|let'?s|go\s+and|try\s+to)\s+",
    re.I,
)
_WANT = re.compile(r"\b(?:want|need|like|give\s+me|get\s+me)\b", re.I)
_QUESTION = re.compile(r"^(?:what|how|why|when|where|who|which|is|are|was|were|can|could|does|do|did|should)\b", re.I)
_WRITE_VERB = re.compile(r"\b(write|save|create|make|generate|draft|put|store|export|produce|prepare|compose|dump|jot)\b", re.I)
_FILE_NOUN = re.compile(
    r"\b(files?|docs?|documents?|markdown|md|readme|txt|pdf|docx|spreadsheet|csv|json|html|note\s+file|text\s+file)\b", re.I
)
_DOC_NOUN = re.compile(r"\b(docs?|documents?|reports?|summary|summari[sz]ing|article|essay|write-?up|overview)\b", re.I)
_SEARCH_VERB = re.compile(r"\b(search|google|look\s+(?:it\s+|them\s+|this\s+)?up|research|browse|find\s+out)\b", re.I)
_WEB_WORD = re.compile(
    r"\b(web|internet|online|google|news|latest|newest|today'?s|this\s+(?:week|month)|recent(?:ly)?|websites?)\b|https?://",
    re.I,
)
_LOCAL_WORD = re.compile(
    r"\b(files?|folders?|director(?:y|ies)|code(?:base)?|repo(?:sitory)?|workspace|project|disk|memory|memories|"
    r"chat|conversation|logs?)\b",
    re.I,
)
_ANSWER_VERB = re.compile(
    r"^(?:tell|explain|describe|compare|summari[sz]e|list|give|show|recommend|suggest|answer|define|teach|outline|"
    r"name|translate)\b",
    re.I,
)
_RUN_VERB = re.compile(r"^(?:run|execute|test|build|compile|install|launch)\b", re.I)
_READ_VERB = re.compile(r"^(?:read|open|look\s+at|check|review|analy[sz]e|summari[sz]e|study|inspect|go\s+through)\b", re.I)
_EXT = (r"md|markdown|txt|text|pdf|docx?|odt|rtf|html?|csv|tsv|json|jsonl|ya?ml|xml|ini|toml|log|tex|"
        r"py|js|ts|sh|pptx|xlsx|ods|odp|svg|png|jpe?g")
_PATH = re.compile(
    r"(?<![\w:/~.-])("
    r"~/[^\s\"'`<>|,;]*|/[\w.-]+(?:/[^\s\"'`<>|,;]*)*|\.{1,2}/[^\s\"'`<>|,;]+"
    rf"|[\w][\w-]*(?:\.[\w-]+)*\.(?:{_EXT}))(?![\w/])",
    re.I,
)
_PLACE = re.compile(
    r"\b(?:on|to|in|into|onto|at|inside|under)\s+(?:my\s+|the\s+)?"
    r"(desktop|documents|downloads|pictures|music|videos)(?:\s+(?:folder|directory))?\b",
    re.I,
)
_XDG = {"desktop": "DESKTOP", "documents": "DOCUMENTS", "downloads": "DOWNLOAD",
        "pictures": "PICTURES", "music": "MUSIC", "videos": "VIDEOS"}
# Words of a step that say how, not what: they never count as the topic to check.
_FILLER = frozenset("""tell give show explain describe briefly brief short quick quickly little bit summary summarize
    summarise about write doc docs document file markdown save search web internet online latest news find
    what is are me us please then finally first also""".split())


def _xdg_dir(place: str) -> Path:
    """The user's real folder for a place word (localised names come from user-dirs.dirs)."""
    home = Path.home()
    key = _XDG[place.lower()]
    try:
        config_home = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
        for line in (config_home / "user-dirs.dirs").read_text(encoding="utf-8").splitlines():
            match = re.match(rf'\s*XDG_{key}_DIR="(.+)"\s*$', line)
            if match:
                return Path(match.group(1).replace("$HOME", str(home))).expanduser()
    except OSError:
        pass
    return home / place.capitalize()


@dataclass(frozen=True)
class Target:
    path: Path  # absolute (it may not exist yet)
    is_dir: bool  # a folder: any file in it will do
    outside: bool  # outside the workspace (reached through route())
    shown: str  # as the model should write it: ~/… outside the workspace, relative inside

    @property
    def root(self) -> Path:
        return self.path if self.is_dir else self.path.parent

    def covers(self, path: Path) -> bool:
        return path == self.path or (self.is_dir and path.is_relative_to(self.path))


@dataclass
class Step:
    index: int
    text: str
    kind: str
    target: Target | None = None
    done: bool = False
    written: Path | None = None  # the file a write step produced

    @property
    def short(self) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= STEP_CHARS else text[: STEP_CHARS - 1].rstrip() + "…"

    def hint(self) -> str:
        if self.kind == "answer":
            return "answer it in your reply"
        if self.kind == "other":
            return "do it as part of the work"
        tool = _TOOL_HINT[self.kind]
        if self.target is None:
            return f"use {tool}"
        if not self.target.is_dir:
            return f'{tool} with path "{self.target.shown}"'
        return f'{tool} in "{self.target.shown}"'


@dataclass
class Route:
    workspace: Any  # sandbox.Workspace rooted at the named folder
    args: dict[str, Any]
    shown: str  # the path as the user sees it


def _strip(fragment: str) -> str:
    text = _LEAD.sub("", _MARKER.sub("", fragment.strip())).strip(" ,:;-")
    return text


def _is_instruction(core: str) -> bool:
    # "php zeroshell?" is the user thinking aloud; a question to answer has a few words.
    asks = bool(_QUESTION.match(core)) or core.endswith("?")
    return bool(directives.REQUEST.match(core)) or (asks and len(core.split()) >= 3)


def split_steps(message: str) -> list[str]:
    """The ordered instructions in one user message ([] for a plain chat line)."""
    text = _CODE.sub(" ", message or "")
    if len(text) > 4000:  # a pasted document: only its framing carries instructions
        text = text[:1500] + "\n" + text[-1000:]
    parts = _SPLIT.split(text)
    steps: list[str] = []
    context = ""  # what the user says before the first instruction ("there was an attack yesterday…")
    for index in range(0, len(parts), 2):
        piece, separator = parts[index], parts[index - 1] if index else ""
        core = _strip(piece)
        if not core:
            continue
        joins = steps and (not _is_instruction(core) or len(steps[-1].split()) < 2)
        if joins:  # a continuation ("cats and dogs"), or the step before was a bare verb ("find and fix")
            steps[-1] = f"{steps[-1]}{separator if separator.strip() in (',', 'and', '') else ' '}{piece.strip()}"
        elif _is_instruction(core):
            steps.append(core)
        elif not context:
            context = core
    if not steps and context:  # a statement only: it is the whole request
        steps = [context]
    return [" ".join(step.split()).rstrip(" ,;") for step in steps][:MAX_STEPS]


def _target(text: str, kind: str, workspace_root: Path) -> Target | None:
    place = _PLACE.search(text)
    folder = _xdg_dir(place.group(1)) if place else None
    for raw in _PATH.findall(text):
        raw = raw.rstrip(".,;:!?)]}'\"")
        if not raw or raw in ("~", "/") or raw.startswith("//"):
            continue
        if raw.startswith(("~", "/")):
            path = Path(os.path.expanduser(raw))
        elif "/" in raw:
            path = workspace_root / raw
        else:
            path = (folder or workspace_root) / raw
        path = Path(os.path.normpath(path))
        is_dir = raw.endswith("/") or path.is_dir() or (not path.suffix and kind == "write")
        return _checked(path, is_dir, workspace_root)
    if folder is not None:
        return _checked(Path(os.path.normpath(folder)), True, workspace_root)
    return None


def _checked(path: Path, is_dir: bool, workspace_root: Path) -> Target | None:
    """A target, or None for a place no conversation may work in."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    inside = resolved == workspace_root or resolved.is_relative_to(workspace_root)
    if not inside:
        root = resolved if is_dir else resolved.parent
        home = Path.home().resolve()
        if not root.is_relative_to(home) or root == home:
            return None
        if safety.place_of(str(root)) in workspaces.FORBIDDEN_PLACES:
            return None
        if safety.place_of(str(resolved)) in workspaces.FORBIDDEN_PLACES:
            return None
    if inside:
        shown = os.path.relpath(resolved, workspace_root) if resolved != workspace_root else "."
    else:
        shown = config.display_path(str(resolved))
    return Target(resolved, is_dir, not inside, shown + ("/" if is_dir and shown != "." else ""))


def _kind(text: str, has_path: bool) -> str:
    core = _ASK.sub("", text)
    if _WRITE_VERB.search(core) and (has_path or _FILE_NOUN.search(core) or _PLACE.search(core)):
        return "write"
    if _WANT.search(core) and _PLACE.search(core) and not _READ_VERB.match(core):  # "I want a doc on my desktop"
        return "write"
    if _RUN_VERB.match(core):
        return "run"
    if _SEARCH_VERB.search(core) and (_WEB_WORD.search(core) or not _LOCAL_WORD.search(core)):
        return "web"
    if _WEB_WORD.search(core) and not _LOCAL_WORD.search(core):
        return "web"
    if has_path and _READ_VERB.match(core):
        return "read"
    if _QUESTION.match(core) or core.endswith("?") or _ANSWER_VERB.match(core):
        return "answer"
    return "other"  # a constraint or a step code cannot check ("make sure it ends with a newline")


def build(message: str, workspace_root: Path | str) -> TurnPlan | None:
    """The plan for one user message, or None when there is nothing to track."""
    root = Path(workspace_root).resolve()
    steps: list[Step] = []
    for text in split_steps(message):
        has_path = bool(_PATH.search(text)) or bool(_PLACE.search(text))
        kind = _kind(text, has_path)
        target = _target(text, kind, root) if kind in ("write", "read") else None
        if steps and kind == "write" and steps[-1].kind == "write" and (
            target is None or steps[-1].target is None
        ) and re.match(r"^(?:save|put|store|place|export)\b", text, re.I):
            # "write a doc about X and save it on my desktop" is one deliverable
            previous = steps[-1]
            previous.text = f"{previous.text}, {text}"
            previous.target = previous.target or target
            continue
        steps.append(Step(len(steps) + 1, text, kind, target))
    if not steps:
        return None
    plan = TurnPlan(steps, root)
    return plan if plan.worth_tracking else None


@dataclass
class TurnPlan:
    steps: list[Step]
    workspace_root: Path
    followups: int = 0
    _written: dict[Path, int] = field(default_factory=dict)

    @property
    def worth_tracking(self) -> bool:
        tracked = [s for s in self.steps if s.kind not in ("answer", "other")]
        return bool(tracked) and (len(self.steps) > 1 or any(s.target is not None for s in tracked))

    @property
    def outside_targets(self) -> list[Target]:
        return [s.target for s in self.steps if s.target is not None and s.target.outside]

    @property
    def checkable(self) -> bool:
        """Every step can be verified in code (no self-check round is needed)."""
        return all(s.kind != "other" for s in self.steps)

    def remaining(self) -> list[Step]:
        return [s for s in self.steps if s.kind not in ("answer", "other") and not s.done]

    # --- prompt ----------------------------------------------------------------

    def note(self) -> str:
        """Steps for the per-turn context note ('' when the message says it plainly)."""
        if len(self.steps) < 2 and not self.outside_targets:
            return ""
        lines = [f"This request has {len(self.steps)} steps. Do them in order, then answer the user:"
                 if len(self.steps) > 1 else "To do, then answer the user:"]
        lines += [f"{s.index}. {s.short} → {s.hint()}" for s in self.steps]
        if self.outside_targets:
            places = sorted({t.shown for t in self.outside_targets})
            lines.append(f"The user named {', '.join(places)} in this message, so file tools may use "
                         f"{'that path' if len(places) == 1 else 'those paths'} this turn although "
                         f"{'it is' if len(places) == 1 else 'they are'} outside the workspace.")
        if any(s.kind == "write" for s in self.steps):
            lines.append("When you answer, say briefly what you saved where; do not paste a saved file's "
                         "content again.")
        return "\n".join(lines)

    # --- routing ---------------------------------------------------------------

    def route(self, name: str, args: dict[str, Any], workspace: Any) -> Route | None:
        """Run a file call in the folder the user named when its path points there."""
        if name not in ROUTED_TOOLS or not isinstance(args.get("path"), str) or not self.outside_targets:
            return None
        raw = args["path"].strip()
        if not raw:
            return None
        from .sandbox import Workspace  # local: sandbox imports safety, as this module does

        for target in self.outside_targets:
            path = self._resolve_toward(raw, target, workspace)
            if path is None:
                continue
            # A named file allows that file (and listing its folder). A named folder ("on my
            # desktop") allows reading and *new* files in it: its existing files stay untouched.
            if not (target.covers(path) or (name == "list_files" and path == target.root)):
                continue
            if name in _WRITES and target.is_dir and (name != "write_file" or os.path.lexists(path)):
                continue
            if name not in _WRITES and not target.root.exists():
                continue
            relative = os.path.relpath(path, target.root)
            return Route(Workspace(target.root), {**args, "path": relative}, config.display_path(str(path)))
        return None

    def _resolve_toward(self, raw: str, target: Target, workspace: Any) -> Path | None:
        """Where a call's path points when it is meant for this target (None if not)."""
        expanded = Path(os.path.expanduser(raw))
        if expanded.is_absolute():  # resolved like the targets (a symlinked ~/Desktop still matches)
            return expanded.resolve()
        home_relative = (Path.home() / raw).resolve()  # "Desktop/notes/pi.md"
        if target.covers(home_relative) or home_relative == target.root:
            return home_relative
        # The model dropped the folder ("pi.md"): only for a file the workspace does not have.
        files = [t for t in self.outside_targets if not t.is_dir]
        if not target.is_dir and len(files) == 1 and Path(raw).name == target.path.name and "/" not in raw:
            try:
                if not workspace.resolve(raw).exists():
                    return target.path
            except Exception:  # a path the workspace refuses is not one it has
                return target.path
        return None

    # --- progress --------------------------------------------------------------

    def observe(self, name: str, args: dict[str, Any], ok: bool, written: Path | None = None) -> bool:
        """Tick off the first open step this successful call completes. True if one was."""
        if not ok:
            return False
        kind = ("web" if name in WEB_TOOLS else "write" if name in WRITE_TOOLS
                else "run" if name in RUN_TOOLS else "read" if name in READ_TOOLS else None)
        if kind is None:
            return False
        if written is not None:
            self._written[written] = self._written.get(written, 0) + 1
        for step in self.steps:
            if step.done or step.kind != kind:
                continue
            if kind == "write" and step.target is not None and not _delivers(step.target, written):
                continue
            step.done = True
            step.written = written if kind == "write" else None
            return True
        return False

    def progress_line(self, reply_text: str = "") -> str:
        """What comes next, for the tool result of the call that finished a step."""
        left = self.remaining()
        if left:
            following = left[0]
            done = sum(step.done for step in self.steps)
            return (f"[Kiwibro plan: {done} of {len(self.steps)} steps done. "
                    f"Next: {following.index}. {following.short} → {following.hint()}]")
        # Only what the reply has not covered yet: asking again for an answer already given
        # makes small models repeat it word for word.
        replies = [step for step, _why in self.unmet(reply_text) if step.kind == "answer"]
        saved = [s.written for s in self.steps if s.written is not None]
        checks = "; ".join(f"{config.display_path(str(p))} exists ({_size(p)})" for p in saved if p.is_file())
        parts = ["[Kiwibro plan: every step that needs a tool is done."]
        if checks:
            parts.append(f"Checked: {checks}.")
        if replies:
            parts.append("Now answer the user, covering: " + "; ".join(s.short for s in replies) + ".")
        else:
            parts.append("Now finish in one or two sentences (what you saved where); do not repeat what you "
                         "already wrote.")
        parts.append("No more tool calls are needed.]")
        return " ".join(parts)

    def statuses(self) -> list[dict[str, Any]]:
        return [{"n": s.index, "text": s.short, "kind": s.kind, "done": s.done,
                 **({"target": s.target.shown} if s.target is not None else {})} for s in self.steps]

    # --- verification ----------------------------------------------------------

    def unmet(self, reply_text: str) -> list[tuple[Step, str]]:
        """Steps not delivered, with why (checked against the disk and the reply)."""
        problems: list[tuple[Step, str]] = []
        reply = reply_text.lower()
        for step in self.steps:
            if step.kind == "answer":
                topic = _topic(step.text)  # catches an answer left out, not one worded differently
                if topic and not any(w in reply for w in topic):
                    problems.append((step, "your reply does not cover it yet"))
            elif step.kind == "other":
                continue
            elif not step.done:
                problems.append((step, "not done yet"))
            elif step.kind == "write" and step.written is not None:
                if not step.written.is_file():
                    problems.append((step, f"{config.display_path(str(step.written))} does not exist"))
                elif step.written.stat().st_size < (MIN_DOC_BYTES if _DOC_NOUN.search(step.text) else 1):
                    problems.append((step, f"{config.display_path(str(step.written))} is (almost) empty"))
        return problems

    def followup(self, reply_text: str) -> str:
        """One corrective nudge for what is left ('' when everything checks out or the budget is spent)."""
        problems = self.unmet(reply_text)
        if not problems or self.followups >= MAX_FOLLOWUPS:
            return ""
        self.followups += 1
        tools_needed = any(step.kind != "answer" for step, _why in problems)
        lines = ["The user's request is not finished. Still open:"]
        lines += [f"{step.index}. {step.short} → {why}; {step.hint()}" for step, why in problems]
        lines.append("Do these now with your tools, then reply briefly." if tools_needed
                     else "Add this to your reply now; no tool calls are needed.")
        return "\n".join(lines)

    def settle(self, reply_text: str) -> None:
        """Mark the answer steps the final reply covers (for the closing status)."""
        unmet = {step.index for step, _why in self.unmet(reply_text)}
        for step in self.steps:
            if step.kind in ("answer", "other") and step.index not in unmet:
                step.done = True

    def summary_line(self, reply_text: str) -> str:
        """An honest closing line when the turn ends with steps still open ('' otherwise)."""
        left = [step for step, _why in self.unmet(reply_text) if step.kind != "answer"]
        if not left:
            return ""
        listed = "; ".join(f"{s.index}. {s.short}" for s in left)
        return f"Not finished: {listed}."


def is_small_talk(message: str) -> bool:
    """A greeting or thanks: the reply needs no tools."""
    return bool(_SMALL_TALK.match((message or "").strip()))


def search_terms(name: str, args: dict[str, Any]) -> frozenset[str]:
    """The content words of a search call (empty for calls that are not searches)."""
    if name not in SEARCH_TOOLS:
        return frozenset()
    query = " ".join(str(q) for q in args.get("queries") or []) if name == "multi_search" else str(args.get("query") or "")
    return directives.keywords(re.sub(r"[-_./]+", " ", query))  # "zero-day" and "zero day" are the same search


def repeated(terms: frozenset[str], earlier: list[tuple[frozenset[str], str]]) -> str | None:
    """The earlier query of this turn that this search nearly repeats, if any."""
    for seen, query in earlier:
        if terms and seen and len(terms & seen) / len(terms | seen) >= REPEAT_JACCARD:
            return query
    return None


def _delivers(target: Target, written: Path | None) -> bool:
    if written is None:
        return False
    if target.covers(written):
        return True
    return not target.outside and not target.is_dir and written.name == target.path.name


def _topic(text: str, limit: int = 4) -> list[str]:
    """The first few content words of a step, in message order."""
    words = directives.keywords(text) - _FILLER
    ordered = dict.fromkeys(w for w in re.findall(r"[a-z0-9][a-z0-9+#._-]{1,}", text.lower()) if w in words)
    return list(ordered)[:limit]


def _size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    return f"{size} bytes" if size < 2048 else f"{size / 1024:.1f} KB"


def written_path(name: str, args: dict[str, Any], workspace: Any) -> Path | None:
    """The file a successful write call produced (absolute), or None."""
    raw = args.get("output") if name == "convert_document" and args.get("output") else args.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return workspace.resolve(raw)
    except Exception:  # validated already; never fail a turn over bookkeeping
        return None
