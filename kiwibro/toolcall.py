"""Tool-call rescue for small local models.

Small, heavily quantised models (1-4B) get tool calls *almost* right: JSON with
single quotes or trailing commas, calls printed as text instead of structured
``tool_calls``, near-miss tool names, ``file`` instead of ``path``, numbers as
strings, values out of range. This module turns those near-misses into valid
calls and real failures into short corrective errors that show valid usage.
It also implements the text tool protocol for models without native tool
support. Everything here is pure and cheap enough to run on every round.
"""

from __future__ import annotations

import ast
import difflib
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Collection, Iterable, Iterator, Mapping

POSITIONAL = "__positional__"  # carries positional values from name(a, b) style calls
MAX_PARSE_CHARS = 200_000
MAX_DEPTH = 64

# --- lenient JSON ---------------------------------------------------------------------

_QUOTES = {'"': '"', "'": "'", "`": "`", "“": "”", "”": "”", "‘": "’", "’": "’"}
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "/": "/", "\\": "\\", '"': '"', "'": "'", "`": "`"}
_LITERALS: dict[str, Any] = {
    "true": True, "false": False, "null": None, "none": None, "nil": None, "undefined": None, "nan": None,
}
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_WORD_RE = re.compile(r"[A-Za-z_]+")
_HEX4_RE = re.compile(r"[0-9a-fA-F]{4}")
_KEY_AHEAD_RE = re.compile(r"\s*(?:[\"'“‘][^\"'”’\n]{1,64}[\"'”’]|[A-Za-z_$][\w$-]{0,63})\s*[:=]")
_FENCE_RE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\n?(.*?)```", re.DOTALL)


class _Lenient:
    """Recursive-descent parser for almost-JSON.

    Accepts single/smart quotes, bare keys and values, ``=`` for ``:``,
    missing or trailing commas, comments, Python literals, unescaped quotes
    inside strings and missing closers. Stops after the first value.
    """

    def __init__(self, text: str, pos: int) -> None:
        self.text = text
        self.pos = pos
        self.containers: list[str] = []

    def parse(self) -> tuple[Any, int]:
        value = self.value()
        return value, self.pos

    def skip(self) -> None:
        text, size = self.text, len(self.text)
        while self.pos < size:
            char = text[self.pos]
            if char in " \t\r\n﻿":
                self.pos += 1
            elif char == "#" or text.startswith("//", self.pos):
                end = text.find("\n", self.pos)
                self.pos = size if end < 0 else end + 1
            elif text.startswith("/*", self.pos):
                end = text.find("*/", self.pos + 2)
                self.pos = size if end < 0 else end + 2
            else:
                return

    def peek(self) -> str:
        self.skip()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def value(self) -> Any:
        char = self.peek()
        if char in ("{", "["):
            return self.container(char)
        if char in _QUOTES:
            return self.string()
        if not char:
            raise ValueError("unexpected end of input")
        return self.bare()

    def container(self, opener: str) -> Any:
        if len(self.containers) >= MAX_DEPTH:
            raise ValueError("nesting is too deep")
        self.containers.append(opener)
        self.pos += 1
        is_object = opener == "{"
        result: Any = {} if is_object else []
        while True:
            char = self.peek()
            if char in ("}", "]", ""):
                self.pos += 1 if char else 0  # mismatched or missing closers end the container
                break
            if char == ",":
                self.pos += 1
                continue
            if not is_object:
                result.append(self.value())
                continue
            key = self.string() if char in _QUOTES else self.bare_key()
            if self.peek() in (":", "="):
                self.pos += 1
                result[str(key)] = None if self.peek() in (",", "}", "]", "") else self.value()
            elif key != "":
                result[str(key)] = None
        self.containers.pop()
        return result

    def string(self) -> str:
        text, size = self.text, len(self.text)
        opener = text[self.pos]
        closer = _QUOTES[opener]
        self.pos += 1
        chars: list[str] = []
        while self.pos < size:
            char = text[self.pos]
            if char == "\\" and self.pos + 1 < size:
                escaped = text[self.pos + 1]
                if escaped == "u" and _HEX4_RE.match(text, self.pos + 2):
                    chars.append(chr(int(text[self.pos + 2 : self.pos + 6], 16)))
                    self.pos += 6
                    continue
                chars.append(_ESCAPES.get(escaped, "\\" + escaped))
                self.pos += 2
                continue
            if char in (closer, opener) and self.closes_at(self.pos + 1):
                self.pos += 1
                return "".join(chars)
            chars.append(char)
            self.pos += 1
        return "".join(chars)

    def closes_at(self, index: int) -> bool:
        """A quote ends a string only when what follows looks like JSON structure."""
        rest = self.text[index : index + 96].lstrip(" \t\r")
        if not rest or rest[0] in ":}]\n=":
            return True
        if rest[0] != ",":
            return False
        tail = rest[1:]
        in_array = bool(self.containers) and self.containers[-1] == "["
        return in_array or not tail.strip() or tail.lstrip()[:1] in ("}", "]") or bool(_KEY_AHEAD_RE.match(tail))

    def bare_key(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ":=,}]\n":
            self.pos += 1
        return self.text[start : self.pos].strip()

    def bare(self) -> Any:
        text = self.text
        number = _NUMBER_RE.match(text, self.pos)
        if number and self.boundary(number.end()):
            self.pos = number.end()
            return _number(number.group())
        word = _WORD_RE.match(text, self.pos)
        if word and word.group().lower() in _LITERALS and self.boundary(word.end()):
            self.pos = word.end()
            return _LITERALS[word.group().lower()]
        start = self.pos
        while self.pos < len(text) and text[self.pos] not in ",}]\n":
            self.pos += 1
        return text[start : self.pos].strip()

    def boundary(self, index: int) -> bool:
        rest = self.text[index : index + 64].lstrip(" \t\r")
        return not rest or rest[0] in ",}]\n\"'"


def _number(token: str) -> int | float:
    try:
        return int(token)
    except ValueError:
        return float(token)


def _parse_at(text: str, start: int) -> tuple[Any, int] | None:
    try:
        return json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        pass
    try:
        return _Lenient(text, start).parse()
    except (ValueError, RecursionError):
        return None


def repair_json(text: Any) -> Any:
    """Parse model-written JSON leniently. Returns None when nothing usable is found."""
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    source = text.strip()[:MAX_PARSE_CHARS]
    if not source:
        return None
    try:
        return json.loads(source)
    except ValueError:
        pass
    fence = _FENCE_RE.search(source)
    if fence and fence.group(1).lstrip()[:1] in ("{", "["):
        source = fence.group(1).strip()
    for start in sorted(index for index in (source.find("{"), source.find("[")) if index >= 0):
        parsed = _parse_at(source, start)
        if parsed is not None and isinstance(parsed[0], (dict, list)):
            return parsed[0]
    return None


def parse_arguments(raw: Any) -> tuple[dict[str, Any], list[str]]:
    """Turn a tool call's ``arguments`` (string, dict, anything) into a dict plus repair notes."""
    if raw is None or raw == "" or raw == {}:
        return {}, []
    notes: list[str] = []
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            value = repair_json(raw)
            if value is None:
                return {POSITIONAL: [raw.strip()]}, ["arguments were not JSON; used them as plain text"]
            notes.append("repaired malformed JSON arguments")
    if isinstance(value, str):  # double-encoded JSON
        inner = repair_json(value)
        if isinstance(inner, dict):
            value = inner
            notes.append("decoded double-encoded arguments")
        else:
            return {POSITIONAL: [value]}, notes
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict):
            value = value[0]
        else:
            return {POSITIONAL: value}, notes
    if not isinstance(value, dict):
        return {POSITIONAL: [value]}, notes
    return dict(value), notes


# --- tool names -----------------------------------------------------------------------

_NAME_PREFIX_RE = re.compile(r"^(?:functions|function|tools|tool|default_api|api)[.:]", re.IGNORECASE)
_NAME_SUFFIX_RE = re.compile(r"(?:\.call|\(\s*\))$")
_IDENT_RE = re.compile(r"[A-Za-z_][\w.\-]{0,63}")

# Squashed alias -> candidate tool names in priority order (only existing names are used).
_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(
        ("search", "google", "googlesearch", "bing", "internetsearch", "searchweb", "searchinternet",
         "searchtheweb", "bravesearch", "duckduckgo", "ddg", "websearchtool", "searchonline"),
        ("web_search",),
    ),
    **dict.fromkeys(
        ("browse", "openurl", "fetch", "fetchurl", "geturl", "readurl", "visit", "visitpage", "openpage",
         "readpage", "readwebpage", "webpage", "scrape", "openwebsite", "getpage"),
        ("fetch_page", "browse_page"),
    ),
    **dict.fromkeys(
        ("read", "cat", "openfile", "viewfile", "fileread", "getfile", "view", "readtextfile", "showfile"),
        ("read_file",),
    ),
    **dict.fromkeys(("ls", "listdir", "listdirectory", "dir", "listfolder", "tree", "listfile"), ("list_files",)),
    **dict.fromkeys(
        ("write", "createfile", "savefile", "filewrite", "newfile", "writetofile", "makefile"), ("write_file",)
    ),
    **dict.fromkeys(
        ("edit", "editfile", "strreplace", "replace", "strreplaceeditor", "modifyfile", "updatefile", "replacetext"),
        ("replace_in_file", "apply_patch"),
    ),
    "patch": ("apply_patch", "replace_in_file"),
    **dict.fromkeys(
        ("bash", "shell", "exec", "execute", "terminal", "run", "cmd", "command", "executecommand", "runshell",
         "sh", "runbash", "shellcommand", "runterminal", "executebash"),
        ("run_command",),
    ),
    **dict.fromkeys(
        ("python", "pythonexec", "executepython", "runcode", "codeinterpreter", "ipython", "py", "runpythoncode"),
        ("run_python", "run_code"),
    ),
    **dict.fromkeys(("javascript", "node", "js", "runjs", "nodejs", "executejavascript"), ("run_javascript", "run_code")),
    **dict.fromkeys(("memorize", "savememory", "storememory", "addmemory", "memory", "rememberfact"), ("remember",)),
    **dict.fromkeys(("searchmemory", "recallmemory", "memorysearch"), ("recall",)),
    **dict.fromkeys(("calc", "calculator", "math", "compute", "evaluate", "eval", "arithmetic"), ("calculate",)),
    **dict.fromkeys(("grep", "searchfiles", "findinfiles", "codesearch", "ripgrep", "rg"), ("search_code", "grep_files")),
    **dict.fromkeys(("glob", "findfiles", "find"), ("glob_files", "search_code")),
    **dict.fromkeys(("searchdocs", "searchdocuments", "documentsearch", "docsearch"), ("doc_search",)),
    **dict.fromkeys(("readdoc", "readdocument", "opendocument"), ("doc_read",)),
    **dict.fromkeys(
        ("image", "describe", "vision", "lookatimage", "analyzeimage", "viewimage", "seeimage"), ("describe_image",)
    ),
    **dict.fromkeys(("undo", "revert", "rollback", "restore", "undochanges"), ("undo_changes",)),
}


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@lru_cache(maxsize=64)
def _squashed_names(names: frozenset[str]) -> dict[str, str]:
    return {_squash(name): name for name in names}


def resolve_name(raw: Any, names: Collection[str]) -> str | None:
    """Map a model-written tool name onto a real tool name, or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    if raw in names:
        return raw
    cleaned = _NAME_SUFFIX_RE.sub("", _NAME_PREFIX_RE.sub("", raw.strip().strip("`'\" ")))
    squashed_names = _squashed_names(frozenset(names))
    key = _squash(cleaned)
    for variant in (key, key.removesuffix("tool"), key.removeprefix("tool")):
        if variant in squashed_names:
            return squashed_names[variant]
    for target in _NAME_ALIASES.get(key, ()):
        if target in names:
            return target
    close = difflib.get_close_matches(key, list(squashed_names), n=1, cutoff=0.82)
    return squashed_names[close[0]] if close else None


def unknown_tool_error(raw: str, names: Collection[str]) -> str:
    squashed_names = _squashed_names(frozenset(names))
    close = difflib.get_close_matches(_squash(raw or ""), list(squashed_names), n=3, cutoff=0.5)
    hint = f" Did you mean {', '.join(squashed_names[item] for item in close)}?" if close else ""
    return f"Unknown tool '{raw}'.{hint} Available tools: {', '.join(sorted(names))}."


# --- arguments ------------------------------------------------------------------------

_WRAPPER_KEYS = frozenset(
    {"arguments", "args", "parameters", "params", "input", "kwargs", "toolinput", "actioninput", "properties"}
)

# Squashed argument synonym -> candidate property names in priority order.
_KEY_SYNONYMS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(
        ("file", "filename", "filepath", "fname", "targetfile", "relativepath", "pathname", "filelocation",
         "dirpath", "filetoread", "sourcefile"),
        ("path", "document"),
    ),
    **dict.fromkeys(("directory", "folder", "dir"), ("path", "cwd")),
    **dict.fromkeys(
        ("q", "searchquery", "searchterm", "searchterms", "keywords", "keyword", "term", "terms", "topic",
         "searchstring", "querystring"),
        ("query",),
    ),
    "search": ("query", "old"),
    "question": ("query", "question", "task"),
    "prompt": ("question", "task", "query"),
    **dict.fromkeys(("cmd", "bash", "shell", "commandline", "shellcommand", "commandtorun"), ("command",)),
    "script": ("command", "code"),
    **dict.fromkeys(("contents", "body", "data", "filecontent", "filecontents", "filetext", "fulltext"), ("content",)),
    "text": ("content", "query", "expression", "question", "code"),
    "source": ("content", "code"),
    "newcontent": ("content", "new"),
    **dict.fromkeys(
        ("link", "href", "uri", "website", "address", "site", "webpage", "pageurl", "weburl", "linkurl"), ("url",)
    ),
    "page": ("url", "chunk"),
    **dict.fromkeys(("links", "pages", "urllist"), ("urls",)),
    **dict.fromkeys(("searches", "questions", "querylist"), ("queries",)),
    **dict.fromkeys(("oldstr", "oldstring", "oldtext", "before", "original", "oldcode", "searchtext"), ("old",)),
    "find": ("old", "query"),
    "target": ("old", "path"),
    "pattern": ("pattern", "old", "query"),
    **dict.fromkeys(
        ("newstr", "newstring", "newtext", "replacement", "replacewith", "after", "with", "replace", "newcode"),
        ("new",),
    ),
    **dict.fromkeys(("limit", "count", "n", "k", "topk"), ("max_results", "limit", "count")),
    **dict.fromkeys(("num", "numresults", "results", "maxresult", "numberofresults", "resultcount"), ("max_results",)),
    **dict.fromkeys(
        ("startline", "from", "fromline", "linestart", "begin", "offset", "firstline", "line", "startingline"),
        ("start",),
    ),
    **dict.fromkeys(("endline", "to", "toline", "lineend", "stop", "lastline", "endingline"), ("end",)),
    **dict.fromkeys(("workdir", "workingdir", "workingdirectory"), ("cwd",)),
    **dict.fromkeys(("expr", "calculation", "formula", "equation", "math", "calc"), ("expression",)),
    "input": ("expression", "query", "code"),
    **dict.fromkeys(("program", "snippet", "pythoncode", "jscode"), ("code",)),
    **dict.fromkeys(("doc", "docid", "handle", "documentid", "docname"), ("document",)),
    **dict.fromkeys(("type", "category"), ("kind", "type")),
    **dict.fromkeys(("memory", "fact", "note"), ("content",)),
}
_TEXT_KEYS = frozenset({"content", "code", "command", "new", "old", "text"})
# Sandbox roots small models remember from training (agent harnesses, code interpreters).
# The real workspace lives under $HOME, so these can only ever mean "the workspace".
_PHANTOM_ROOT_RE = re.compile(r"^(?:/workspace|/mnt/data|/mnt/user-data|/home/user/workspace|/tmp/workspace)(?=/|$)/*")
_PATH_KEYS = frozenset({"path", "paths", "cwd", "destination", "image_path", "filename"})
_TRUE = frozenset({"true", "yes", "y", "1", "on", "enable", "enabled"})
_FALSE = frozenset({"false", "no", "n", "0", "off", "disable", "disabled", "none"})
_NUMERIC_TEXT_RE = re.compile(r"\s*([-+]?\d+(?:\.\d+)?)(?:\s*[A-Za-z%]+)?\s*")


def _match_key(key: str, properties: Mapping[str, Any], taken: Collection[str]) -> str | None:
    if key in properties:
        return key
    squashed = _squash(key)
    by_squash = {_squash(name): name for name in properties if name not in taken}
    if squashed in by_squash:
        return by_squash[squashed]
    for candidate in _KEY_SYNONYMS.get(squashed, ()):
        if candidate in properties and candidate not in taken:
            return candidate
    tokens = {token for token in re.split(r"[^a-z0-9]+", key.lower()) if len(token) > 1}
    overlap = [name for name in by_squash.values() if tokens & set(name.lower().split("_"))]
    if len(overlap) == 1:
        return overlap[0]
    close = difflib.get_close_matches(squashed, list(by_squash), n=1, cutoff=0.8)
    return by_squash[close[0]] if close else None


def _unwrap(args: dict[str, Any], properties: Mapping[str, Any], notes: list[str]) -> dict[str, Any]:
    """Peel ``{"arguments": {...}}`` / ``{"name": tool, "parameters": {...}}`` wrappers."""
    for _ in range(3):
        wrappers = [key for key in args if _squash(key) in _WRAPPER_KEYS and key not in properties]
        extras = [key for key in args if key not in wrappers and _squash(key) not in ("name", "tool", "type", "id")]
        if len(wrappers) != 1 or extras:
            return args
        inner = args[wrappers[0]]
        if isinstance(inner, str):
            inner = repair_json(inner)
        if not isinstance(inner, dict):
            return args
        notes.append(f"unwrapped '{wrappers[0]}'")
        args = dict(inner)
    return args


def normalize_args(args: Mapping[str, Any], parameters: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map near-miss argument names/types onto a tool's JSON schema. Returns (args, notes)."""
    notes: list[str] = []
    properties: Mapping[str, Any] = parameters.get("properties") or {}
    required = [name for name in parameters.get("required") or [] if name in properties]
    raw = dict(args or {})
    positional = raw.pop(POSITIONAL, None)
    raw = _unwrap(raw, properties, notes)
    if not properties:
        if raw or positional:
            notes.append("this tool takes no arguments; ignored them")
        return {}, notes
    result: dict[str, Any] = {}
    leftovers: dict[str, Any] = {}
    for key, value in raw.items():
        target = _match_key(str(key), properties, result)
        if target is None or target in result:
            leftovers[str(key)] = value
            continue
        result[target] = value
        if target != key:
            notes.append(f"argument '{key}' → '{target}'")
    missing = [name for name in required if name not in result]
    open_slots = missing + [name for name in properties if name not in result and name not in missing]
    values = list(positional or [])
    if not values and len(leftovers) == 1 and missing:
        (key, value), = leftovers.items()
        values, leftovers = [value], {}
        notes.append(f"argument '{key}' → '{missing[0]}'")
    for name, value in zip(open_slots, values):
        result[name] = value
    if leftovers:
        notes.append("ignored unknown argument(s): " + ", ".join(sorted(leftovers)))
    for key in list(result):
        result[key] = _coerce(result[key], properties[key], key, notes)
    return result, notes


def _schema_type(prop: Mapping[str, Any]) -> str | None:
    kind = prop.get("type")
    if isinstance(kind, list):
        kind = next((item for item in kind if item != "null"), None)
    return kind if isinstance(kind, str) else None


def _coerce(value: Any, prop: Mapping[str, Any], key: str, notes: list[str]) -> Any:
    kind = _schema_type(prop)
    if kind == "string":
        value = _to_text(value, key)
        if key in _PATH_KEYS and isinstance(value, str) and _PHANTOM_ROOT_RE.match(value):
            relative = _PHANTOM_ROOT_RE.sub("", value) or "."
            notes.append(f"{key} '{value}' → '{relative}' (paths are relative to the workspace)")
            value = relative
    elif kind in ("integer", "number"):
        value = _to_number(value, prop, key, notes, integer=kind == "integer")
    elif kind == "boolean":
        value = _to_bool(value)
    elif kind == "array":
        value = _to_list(value, prop, key, notes)
    elif kind == "object" and isinstance(value, str):
        parsed = repair_json(value)
        value = parsed if isinstance(parsed, dict) else value
    if prop.get("enum") and value not in prop["enum"]:
        value = _to_enum(value, prop["enum"], key, notes)
    return value


def _to_text(value: Any, key: str) -> Any:
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value):
        return ("\n" if key in _TEXT_KEYS else " ").join(str(item) for item in value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _to_number(value: Any, prop: Mapping[str, Any], key: str, notes: list[str], *, integer: bool) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        match = _NUMERIC_TEXT_RE.fullmatch(value)
        if not match:
            return value
        value = float(match.group(1))
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return value
    number: int | float = int(round(value)) if integer else value
    low, high = prop.get("minimum"), prop.get("maximum")
    if isinstance(low, (int, float)) and number < low:
        notes.append(f"{key} {number} raised to minimum {low}")
        number = low
    if isinstance(high, (int, float)) and number > high:
        notes.append(f"{key} {number} lowered to maximum {high}")
        number = high
    return number


def _to_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    return value


def _to_list(value: Any, prop: Mapping[str, Any], key: str, notes: list[str]) -> Any:
    if isinstance(value, str):
        parsed = repair_json(value) if value.lstrip().startswith("[") else None
        if isinstance(parsed, list):
            value = parsed
        else:
            value = [line.strip() for line in value.splitlines() if line.strip()] or [value]
    elif isinstance(value, (tuple, set)):
        value = list(value)
    elif value is not None and not isinstance(value, list):
        value = [value]
    if not isinstance(value, list):
        return value
    item_schema = prop.get("items") if isinstance(prop.get("items"), Mapping) else {}
    items = [_coerce(item, item_schema, key, notes) for item in value]
    limit = prop.get("maxItems")
    if isinstance(limit, int) and len(items) > limit:
        notes.append(f"{key}: kept the first {limit} of {len(items)} items")
        items = items[:limit]
    return items


def _to_enum(value: Any, allowed: list[Any], key: str, notes: list[str]) -> Any:
    if not isinstance(value, str):
        return value
    by_squash = {_squash(str(option)): option for option in allowed}
    match = by_squash.get(_squash(value))
    if match is None:
        close = difflib.get_close_matches(_squash(value), list(by_squash), n=1, cutoff=0.6)
        match = by_squash[close[0]] if close else None
    if match is None:
        return value
    notes.append(f"{key} '{value}' → '{match}'")
    return match


@dataclass
class PreparedCall:
    """A model tool call after name resolution and argument normalisation."""

    id: str
    raw_name: str
    name: str
    args: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def prepare_call(call: Mapping[str, Any], specs: Mapping[str, Any]) -> PreparedCall:
    """Resolve and normalise one OpenAI-shaped tool call against the tool specs."""
    function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
    raw_name = str(function.get("name") or "").strip()
    call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}")
    args, notes = parse_arguments(function.get("arguments"))
    name = resolve_name(raw_name, specs)
    if name is None:
        args.pop(POSITIONAL, None)
        return PreparedCall(call_id, raw_name, raw_name, args, notes, unknown_tool_error(raw_name, specs))
    if name != raw_name:
        notes.insert(0, f"tool '{raw_name}' → '{name}'")
    args, more = normalize_args(args, specs[name].parameters)
    return PreparedCall(call_id, raw_name, name, args, notes + more)


# --- usage text -----------------------------------------------------------------------


def _type_label(prop: Mapping[str, Any]) -> str:
    if prop.get("enum"):
        return " | ".join(json.dumps(option) for option in prop["enum"])
    kind = _schema_type(prop) or "any"
    if kind == "array":
        items = prop.get("items") if isinstance(prop.get("items"), Mapping) else {}
        return f"list of {_type_label(items)}"
    low, high = prop.get("minimum"), prop.get("maximum")
    if low is not None and high is not None:
        return f"{kind} {low}-{high}"
    return kind


def signature(spec: Any) -> str:
    """Compact call signature, e.g. ``web_search(query: string, max_results?: integer 1-8)``."""
    properties = spec.parameters.get("properties") or {}
    required = set(spec.parameters.get("required") or [])
    parts = [f"{key}{'' if key in required else '?'}: {_type_label(prop)}" for key, prop in properties.items()]
    return f"{spec.name}({', '.join(parts)})"


def _placeholder(key: str, prop: Mapping[str, Any]) -> Any:
    if prop.get("enum"):
        return prop["enum"][0]
    kind = _schema_type(prop)
    if kind in ("integer", "number"):
        return prop.get("minimum", 1)
    if kind == "boolean":
        return True
    if kind == "array":
        items = prop.get("items") if isinstance(prop.get("items"), Mapping) else {"type": "string"}
        return [_placeholder(key.removesuffix("s"), items)]
    if kind == "object":
        return {}
    return f"<{key}>"


def example_args(parameters: Mapping[str, Any]) -> dict[str, Any]:
    properties = parameters.get("properties") or {}
    keys = [key for key in parameters.get("required") or [] if key in properties] or list(properties)[:1]
    return {key: _placeholder(key, properties[key]) for key in keys}


def correction(spec: Any, error: str) -> str:
    """Corrective tool error: what went wrong plus the exact valid usage."""
    example = json.dumps(example_args(spec.parameters), ensure_ascii=False)
    return (
        f"Error: {error}\nUsage: {signature(spec)}\nExample arguments: {example}\n"
        "Fix the arguments and call the tool again, or continue without it."
    )


# --- tool calls written as text -------------------------------------------------------


@dataclass
class _Found:
    name: str
    args: Any
    start: int
    end: int


_NAME_KEYS = ("name", "tool", "tool_name", "function", "function_name", "action", "recipient_name")
_ARG_KEYS = ("arguments", "parameters", "args", "input", "action_input", "tool_input", "params", "kwargs")
_OPEN_TAG_RE = re.compile(r"<\|?(tool_calls?|toolcall|function_calls?|functioncall)\|?>", re.IGNORECASE)
_FUNCTION_TAG_RE = re.compile(r"<function=([\w.\-]+)>(.*?)(?:</function>|(?=<function=)|\Z)", re.DOTALL)
_PARAM_TAG_RE = re.compile(r"<parameter=([\w.\-]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|(?=</function>)|\Z)", re.DOTALL)
_INVOKE_RE = re.compile(r"<invoke\s+name=[\"']([\w.\-]+)[\"']\s*>(.*?)(?:</invoke>|\Z)", re.DOTALL)
_INVOKE_PARAM_RE = re.compile(r"<parameter\s+name=[\"']([\w.\-]+)[\"'][^>]*>(.*?)</parameter>", re.DOTALL)
_PYTHON_TAG_RE = re.compile(r"<\|python_tag\|>(.*?)(?:<\|eom_id\|>|<\|eot_id\|>|\Z)", re.DOTALL)
_MISTRAL_RE = re.compile(r"\[TOOL_CALLS\](.*?)(?=\[TOOL_CALLS\]|\Z)", re.DOTALL)
_MISTRAL_ARGS_RE = re.compile(r"([A-Za-z_][\w.\-]*)(?:\[CALL_ID\][\w\-]+)?\[ARGS\]")
_HARMONY_RE = re.compile(
    r"(?:<\|start\|>\w*\s*)?(?:<\|channel\|>\w+\s*)?to=functions\.([\w.\-]+)[^\n<]*?"
    r"(?:<\|constrain\|>\s*\w+\s*)?<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|\Z)",
    re.DOTALL,
)
_DEEPSEEK_RE = re.compile("<｜tool▁call▁begin｜>(.*?)(?:<｜tool▁call▁end｜>|\\Z)", re.DOTALL)
_DEEPSEEK_SEP = "<｜tool▁sep｜>"
_TOOL_FENCE_RE = re.compile(r"```(?:tool_code|tool_calls?|tool)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_REACT_RE = re.compile(
    r"Action\s*:\s*`?([\w.\-]+)`?\s*\n+\s*Action\s*Input\s*:\s*(.+?)(?=\n\s*(?:Observation|Thought|Action|Final Answer)\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_CALL_OBJECT_RE = re.compile(r"\{\s*[\"']?(?:name|tool|tool_name|function|function_name|action|recipient_name)[\"']?\s*:")
_EXAMPLE_RE = re.compile(r"\b(?:example|e\.g\.|for instance|such as|syntax|format)\b", re.IGNORECASE)
_STRAY_TOKENS_RE = re.compile(
    r"<\|python_tag\|>|<\|eom_id\|>|<\|eot_id\|>|<\|call\|>|<\|end\|>|\[TOOL_CALLS\]"
    r"|<\|?/?(?:tool_calls?|toolcall|function_calls?|functioncall)\|?>"
    "|<｜tool▁calls▁(?:begin|end)｜>",
    re.IGNORECASE,
)
WEAK_PROSE_LIMIT = 400


def _json_values(text: str, starts: Iterable[int] | None = None) -> Iterator[tuple[Any, int, int]]:
    """Yield (value, start, end) for JSON values found in text (lenient)."""
    if starts is None:
        starts = (match.start() for match in re.finditer(r"[{\[]", text))
    cursor = 0
    for start in starts:
        if start < cursor:
            continue
        parsed = _parse_at(text, start)
        if parsed is None:
            continue
        value, end = parsed
        yield value, start, end
        cursor = max(end, start + 1)


def _calls_from_value(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, list):
        return [call for item in value[:16] for call in _calls_from_value(item)]
    if not isinstance(value, dict):
        return []
    function = value.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return [(function["name"], function.get("arguments", function.get("parameters", {})))]
    for key in _NAME_KEYS:
        name = value.get(key)
        if isinstance(name, str) and _IDENT_RE.fullmatch(name.strip()):
            for arg_key in _ARG_KEYS:
                if arg_key in value:
                    return [(name.strip(), value[arg_key])]
            return [(name.strip(), {k: v for k, v in value.items() if k not in (key, "type", "id")})]
    if "tool_calls" in value:
        return _calls_from_value(value["tool_calls"])
    return []


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return ast.unparse(node)


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _pythonic_calls(text: str) -> list[tuple[str, Any]]:
    """Parse ``name(a=1, b="x")`` or ``[f(...), g(...)]`` call syntax."""
    source = text.strip().strip("`").strip()
    if not source or len(source) > 20_000:
        return []
    try:
        tree: ast.AST = ast.parse(source, mode="eval")
        roots = [tree.body]  # type: ignore[attr-defined]
    except SyntaxError:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        roots = [stmt.value for stmt in tree.body if isinstance(stmt, ast.Expr)]  # type: ignore[attr-defined]
    calls: list[tuple[str, Any]] = []
    for root in roots:
        for node in root.elts if isinstance(root, (ast.List, ast.Tuple)) else [root]:
            name = _dotted(node.func) if isinstance(node, ast.Call) else None
            if not name:
                continue
            args: dict[str, Any] = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
            if node.args:
                args[POSITIONAL] = [_literal(arg) for arg in node.args]
            calls.append((name, args))
    return calls


def _param_value(text: str) -> Any:
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        parsed = repair_json(stripped)
        if parsed is not None:
            return parsed
    return stripped


def _parse_body(body: str) -> list[tuple[str, Any]]:
    """Calls inside one wrapped block (<tool_call>…, [TOOL_CALLS]…, tool_code fences)."""
    if "<function=" in body:
        return [
            (match.group(1), {key: _param_value(value) for key, value in _PARAM_TAG_RE.findall(match.group(2))}
             or match.group(2).strip())
            for match in _FUNCTION_TAG_RE.finditer(body)
        ]
    if "<invoke" in body:
        return [
            (match.group(1), {key: _param_value(value) for key, value in _INVOKE_PARAM_RE.findall(match.group(2))})
            for match in _INVOKE_RE.finditer(body)
        ]
    calls = [call for value, _start, _end in _json_values(body) for call in _calls_from_value(value)]
    if calls:
        return calls
    headed = re.match(r"\s*([A-Za-z_][\w.\-]*)\s*:?\s*\n?\s*(\{.*\})\s*$", body, re.DOTALL)
    if headed:
        return [(headed.group(1), headed.group(2))]
    return _pythonic_calls(body)


def _wrapped_blocks(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield (body, start, end) for every <tool_call>…</tool_call>-style block."""
    position = 0
    while True:
        opener = _OPEN_TAG_RE.search(text, position)
        if opener is None:
            return
        tag = opener.group(1)
        closer = re.compile(rf"<\|?/{re.escape(tag)}\|?>", re.IGNORECASE).search(text, opener.end())
        following = _OPEN_TAG_RE.search(text, opener.end())
        body_end = closer.start() if closer else len(text)
        block_end = closer.end() if closer else len(text)
        if following and following.start() < body_end:
            body_end = block_end = following.start()
        yield text[opener.end() : body_end], opener.start(), block_end
        position = max(block_end, opener.end())


def _mistral_calls(body: str) -> list[tuple[str, Any]]:
    named = list(_MISTRAL_ARGS_RE.finditer(body))
    if not named:
        return [call for value, _s, _e in _json_values(body) for call in _calls_from_value(value)]
    calls = []
    for match in named:
        parsed = _parse_at(body, match.end()) if body[match.end() : match.end() + 1] in ("{", "[") else None
        calls.append((match.group(1), parsed[0] if parsed else {}))
    return calls


def _deepseek_call(body: str) -> list[tuple[str, Any]]:
    head, _sep, tail = body.partition(_DEEPSEEK_SEP)
    if not _sep:
        return []
    if head.strip() in ("", "function"):
        match = re.match(r"\s*([\w.\-]+)\s*(.*)", tail, re.DOTALL)
        if not match:
            return []
        name, raw = match.group(1), match.group(2)
    else:
        name, raw = head.strip(), tail
    return [(name, repair_json(raw) or {})]


def _strong_calls(text: str) -> list[_Found]:
    """Explicitly marked call formats: always treated as calls."""
    found: list[_Found] = []
    for body, start, end in _wrapped_blocks(text):
        found += [_Found(name, args, start, end) for name, args in _parse_body(body)]
    if not found:
        for match in _FUNCTION_TAG_RE.finditer(text):
            found += [_Found(name, args, match.start(), match.end()) for name, args in _parse_body(match.group(0))]
    for match in _INVOKE_RE.finditer(text) if not found else ():
        found += [_Found(name, args, match.start(), match.end()) for name, args in _parse_body(match.group(0))]
    for match in _PYTHON_TAG_RE.finditer(text):
        body = match.group(1)
        calls = [call for value, _s, _e in _json_values(body) for call in _calls_from_value(value)]
        found += [_Found(n, a, match.start(), match.end()) for n, a in calls or _pythonic_calls(body)]
    for match in _MISTRAL_RE.finditer(text):
        found += [_Found(n, a, match.start(), match.end()) for n, a in _mistral_calls(match.group(1))]
    for match in _HARMONY_RE.finditer(text):
        found.append(_Found(match.group(1), repair_json(match.group(2)) or match.group(2).strip(), match.start(), match.end()))
    for match in _DEEPSEEK_RE.finditer(text):
        found += [_Found(n, a, match.start(), match.end()) for n, a in _deepseek_call(match.group(1))]
    for match in _TOOL_FENCE_RE.finditer(text):
        found += [_Found(n, a, match.start(), match.end()) for n, a in _parse_body(match.group(1))]
    return found


@lru_cache(maxsize=32)
def _pythonic_line_re(names: frozenset[str]) -> re.Pattern[str]:
    alternatives = "|".join(sorted((re.escape(name) for name in names), key=len, reverse=True))
    return re.compile(rf"^[ \t]*\[?[ \t]*(?:{alternatives})[ \t]*\(.*\)[ \t]*\]?[ \t]*$", re.MULTILINE)


def _weak_calls(text: str, names: Collection[str]) -> list[_Found]:
    """Unmarked formats (bare/fenced JSON, ReAct, name(...) lines): need a known tool name."""
    found: list[_Found] = []
    starts = (match.start() for match in _CALL_OBJECT_RE.finditer(text))
    for value, start, end in _json_values(text, starts):
        found += [_Found(name, args, start, end) for name, args in _calls_from_value(value)]
    for match in _REACT_RE.finditer(text):
        raw = match.group(2).strip()
        args = repair_json(raw) if raw[:1] in ("{", "[") else None
        found.append(_Found(match.group(1), args if args is not None else {POSITIONAL: [raw.strip("\"'")]}, match.start(), match.end()))
    if names:
        for match in _pythonic_line_re(frozenset(names)).finditer(text):
            found += [_Found(n, a, match.start(), match.end()) for n, a in _pythonic_calls(match.group(0))]
    return [item for item in found if resolve_name(item.name, names)]


def _merge_spans(found: list[_Found]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start, end in sorted((item.start, item.end) for item in found):
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def _without_spans(text: str, found: list[_Found]) -> str:
    pieces, cursor = [], 0
    for start, end in _merge_spans(found):
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    cleaned = _STRAY_TOKENS_RE.sub("", "".join(pieces))
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def extract_calls(content: str, names: Collection[str]) -> tuple[list[dict[str, Any]], str]:
    """Find tool calls written as text. Returns (OpenAI-shaped calls, content without them).

    Explicitly marked formats (``<tool_call>``, ``<function=…>``, ``<|python_tag|>``,
    ``[TOOL_CALLS]``, harmony/DeepSeek markers, ``tool_code`` fences) always count.
    Unmarked formats (bare JSON, ReAct, ``name(...)``) count only for known tool
    names, outside "for example" prose, and when the call dominates the message.
    """
    if not content or not content.strip():
        return [], content
    text = content[:MAX_PARSE_CHARS]
    found = [item for item in _strong_calls(text) if _IDENT_RE.fullmatch(item.name or "")]
    if not found:
        found = [
            item for item in _weak_calls(text, names)
            if not _EXAMPLE_RE.search(text[max(0, item.start - 80) : item.start])
        ]
        if not found:
            return [], content
        prose = _without_spans(text, found)
        ends_message = max(item.end for item in found) >= len(text.rstrip()) - 3
        if len(prose) > WEAK_PROSE_LIMIT and not ends_message:
            return [], content
    calls = []
    seen: set[str] = set()
    for item in sorted(found, key=lambda entry: entry.start):
        arguments = item.args if isinstance(item.args, str) else json.dumps(item.args, ensure_ascii=False, default=str)
        key = f"{item.name}\x00{arguments}"
        if key in seen:
            continue
        seen.add(key)
        calls.append(
            {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function", "function": {"name": item.name, "arguments": arguments}}
        )
    return calls, _without_spans(text, found)


# --- text tool protocol (models without native tool support) --------------------------

PROTOCOL_HEADER = """
## Tools
Tools are available. To use one, reply with only a tool call block and nothing else:
<tool_call>
{"name": "TOOL_NAME", "arguments": {"ARGUMENT": "VALUE"}}
</tool_call>
Several blocks in a row run several tools. Each result comes back in a <tool_result> \
message; then call another tool or answer the user normally, without a tool_call block. \
Use the exact tool and argument names listed here:
"""
TOOL_RESULT_TEMPLATE = '<tool_result name="{name}">\n{content}\n</tool_result>'


def _first_sentence(text: str, limit: int = 160) -> str:
    sentence = re.split(r"(?<=[.!?])\s", " ".join((text or "").split()), maxsplit=1)[0]
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


def protocol_block(specs: Iterable[Any]) -> str:
    """Byte-stable tools section for the text protocol (deterministic for a given tool list)."""
    lines = [f"- {signature(spec)} — {_first_sentence(spec.description)}" for spec in specs]
    return PROTOCOL_HEADER + "\n".join(lines) + "\n"


def render_call(name: str, args: Mapping[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": dict(args)}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def to_text_protocol(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite tool-role history as plain user/assistant text for template-less models."""
    names_by_id: dict[str, str] = {}
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            blocks = []
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                name = str(function.get("name") or "tool")
                names_by_id[str(call.get("id") or "")] = name
                args, _notes = parse_arguments(function.get("arguments"))
                args.pop(POSITIONAL, None)
                blocks.append(render_call(name, args))
            text = "\n".join(part for part in [str(message.get("content") or "").strip(), *blocks] if part)
            converted.append({"role": "assistant", "content": text})
        elif role == "tool":
            name = message.get("name") or names_by_id.get(str(message.get("tool_call_id") or "")) or "tool"
            content = TOOL_RESULT_TEMPLATE.format(name=name, content=message.get("content") or "")
            converted.append({"role": "user", "content": content})
        else:
            converted.append({key: value for key, value in message.items() if key != "tool_calls"})
    merged: list[dict[str, Any]] = []
    for message in converted:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous["role"] == message["role"] in ("user", "assistant")
            and isinstance(previous.get("content"), str)
            and isinstance(message.get("content"), str)
        ):
            merged[-1] = {**previous, "content": f"{previous['content']}\n\n{message['content']}"}
        else:
            merged.append(message)
    return merged


# --- streaming --------------------------------------------------------------------------

_STREAM_OPENERS = (
    "<tool_call>", "<|tool_call|>", "<tool_calls>", "<function=", "<function_calls>", "<|python_tag|>",
    "[TOOL_CALLS]", "<invoke name=", "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>",
)


class StreamFilter:
    """Hold back tool-call markup while streaming so users never see raw call syntax.

    Text before the first call marker streams through; from the marker on it is
    kept in ``held``. If no call is extracted at the end of the round, the
    caller should emit ``held`` (it was ordinary text after all).
    """

    def __init__(self) -> None:
        self._pending = ""
        self.held = ""

    def feed(self, text: str) -> str:
        if self.held:
            self.held += text
            return ""
        buffer = self._pending + text
        hits = [index for index in (buffer.find(opener) for opener in _STREAM_OPENERS) if index >= 0]
        if hits:
            first = min(hits)
            self._pending, self.held = "", buffer[first:]
            return buffer[:first]
        keep = max(
            (size for opener in _STREAM_OPENERS for size in range(min(len(opener) - 1, len(buffer)), 0, -1)
             if opener.startswith(buffer[-size:])),
            default=0,
        )
        self._pending = buffer[len(buffer) - keep :]
        return buffer[: len(buffer) - keep]

    def flush(self) -> str:
        """Release text held only because it might have started a marker."""
        pending, self._pending = self._pending, ""
        return pending


EMPTY_REPLY_NUDGE = (
    "Your last reply was empty. Continue the task: call a tool if you still need one, "
    "otherwise answer the user now using the results above."
)


def is_empty_reply(content: str | None, tool_calls: list[Any] | None) -> bool:
    text = re.sub(r"<think>.*?(?:</think>|\Z)", "", content or "", flags=re.DOTALL)
    return not tool_calls and not text.strip()
