"""Explicit field bindings for a standalone hinged HTML component."""

from __future__ import annotations

import re
from collections.abc import Iterator

from . import components


# Match path-like references too, so one external/nested filename cannot quietly
# become a different workspace-local filename.
_HTML = re.compile(
    r"""[“"'‘`]([^\n“”"'‘’`]{1,104}\.html?)[”"'’`]|"""
    r"(?<![\w./\\~:-])([\w./\\~:-]+\.html?)(?![\w/\\#-]|\.[\w/\\-]|\?\S)",
    re.I,
)
_CREATE = re.compile(
    r"(?:^|[.!?;\n]\s*)(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:create|make|build|save|write|generate|produce|export)\b"
    r"[^.!?;\n]{0,140}$", re.I,
)
_EXISTING = re.compile(
    r"(?:^|[.!?;\n]\s*)(?:please\s+)?(?:edit|update|patch|modify|repair|fix|replace|overwrite|revise)\b"
    r"|\b(?:edit|update|modify|rewrite|use|read|open)\s+(?:the\s+)?existing\s+"
    r"(?:(?:\w+\s+)?(?:file|page|html)\b|[`'\"\w-]+\.html?\b)"
    r"|\b(?:save|write)\s+over\b",
    re.I,
)
_SEPARATE_JOB = re.compile(
    r"(?:^|[.!?;\n]\s*|\b(?:then|and|also|afterward|before)\b\s+)"
    r"(?:(?:please|first|also)\s+)*(?:run|execute|commit|push|publish|deploy|upload|install|read|"
    r"search|browse|compare|explain|report|summari[sz]e|tell|send|copy|move|rename|delete|download)\b",
    re.I,
)
_EDGES = r"top|bottom|left|right"
_EDGES_IN_BRIEF = re.compile(
    rf"\b(?P<leading>{_EDGES})[- ]+(?:edge[- ]+)?hinge(?:d)?\b"
    rf"|\bhinge[ds]?\s+(?:at|along|on|from)\s+(?:(?:its|the|a)\s+)?(?P<trailing>{_EDGES})[- ]+edge\b",
    re.I,
)
_UNSUPPORTED = re.compile(
    rf"\b(?:{_EDGES})(?:[- ]edge)?\s*(?:/|,|\bor\b|\band\b)\s*(?:{_EDGES})(?:[- ]edge)?\s+hinge\b"
    r"|\b(?:two|three|four|2|3|4|several|multiple|independent|both|pair of|another|second|additional)"
    r"\s+(?:(?:independently|moving|extra|distinct|different|separate|service|inner|outer)\s+){0,2}"
    r"(?:hinges?|panels?|leaves|flaps?|lids?|doors?|hatches|buttons|controls)\b"
    r"|\b(?:hinge\s+chain|exploded\s+view|cutaway|WebGL|three\.js|gltf|glb|CAD)\b"
    r"|\b(?:custom|specified|provided|following)\s+(?:(?:inline|external)\s+)?(?:stylesheet|CSS|JavaScript)\b",
    re.I,
)
_NEGATED = re.compile(r"\b(?:not|no|without|never)\s+(?:\w+\s+){0,3}$", re.I)

# The viewport, fixed mount, moving part and each control have different roles.
# Match the role words, not ID names that coincidentally look like those roles.
_ROLES = {
    "scene_id": r"scene|stage|viewport",
    "frame_id": r"fixed frame|stationary frame|frame|fixed base|base|housing",
    "leaf_id": r"moving part|moving leaf|hinged leaf|leaf|service lid|lid|cover|flap|door|panel",
    "toggle_id": r"native button|toggle button|toggle|button|control",
    "status_id": r"visible status|status node|status region|live region|status",
    "detail_id": r"hidden detail|revealed detail|detail",
}
_ROLE = rf"(?:{'|'.join(_ROLES.values())})"
_ID = r"[`'\"“‘]?([A-Za-z][A-Za-z0-9_-]*)[`'\"”’]?"
_DIRECT_ID = re.compile(
    rf"\b({_ROLE})(?:['’]s)?\s+(?:with\s+|using\s+)?(?:HTML\s+)?ID\s*(?:[:=]\s*|(?:of|is|to)\s+)?"
    rf"{_ID}(?=\s|[,.!?;]|$)",
    re.I,
)
_INLINE_ID = re.compile(
    rf"\b(hidden detail|visible status)\s+{_ID}(?=\s*(?:[.,;]|$))",
    re.I,
)
_ID_LIST = re.compile(
    rf"\b(?:give|assign)\s+(?:the\s+)?({_ROLE}(?:\s*(?:,\s*(?:and\s+)?|\band\s+)\s*(?:the\s+)?{_ROLE})+)"
    r"\s+(?:the\s+)?IDs\s+([^.!?;\n]{1,400})",
    re.I,
)
_ID_FIRST = re.compile(r"\bIDs?\b\s*([^.!?;\n]{1,450})", re.I)
_ID_FOR_ROLE = re.compile(rf"(?:^|,\s*(?:and\s+)?|\band\s+){_ID}\s+for\s+(?:the\s+)?({_ROLE})\b", re.I)
_LIST_DELIMITER = re.compile(r"\s*(?:,\s*(?:and\s+)?|\band\s+)\s*", re.I)
_QUOTE = r"""(?:“([^”\n]*)”|"([^"\n]*)"|‘([^’\n]*)’|'([^'\n]*)'|`([^`\n]*)`)"""
_QUOTED_PROSE = re.compile(r"(?<!\w)" + _QUOTE + r"(?!\w)")
# Only text with an explicit name/label becomes a literal bound into the schema.
_LABELS = {
    "title": r"\b(?:page\s+)?title\s*(?:to\s+|of\s+|named\s+|[:=]\s*)",
    "part": r"\b(?:(?:moving part|moving leaf|leaf|lid)\s+(?:named|called)\s+"
            r"|name\s+(?:the\s+)?(?:moving part|moving leaf|leaf|lid)\s+)",
    "button_label": r"\b(?:button|toggle)(?:\s+with\s+ID\s+[`'\"\w-]+)?\s+(?:named|labeled|label(?:ed)?|called)\s+",
    "detail": r"\b(?:revealed\s+)?detail(?:\s+text)?\s*(?:should\s+read\s*|reads?\s*|to\s*|[:=]\s*)",
    "closed_label": r"\bclosed\s+(?:status|label)(?:\s+text)?\s*(?:(?:should\s+read|reads?|to)\s*|[:=]\s*)?",
    "open_label": r"\bopen\s+(?:status|label)(?:\s+text)?\s*(?:(?:should\s+read|reads?|to)\s*|[:=]\s*)?",
}
_STATUS_PAIR = re.compile(rf"\b(?:show|display)\s+{_QUOTE}\s+then\s+{_QUOTE}\s+in\s+(?:the\s+)?(?:visible\s+)?status\b", re.I)
_ANGLE = re.compile(r"\b(?:open|swing|tilt|rotate)s?\b[^.!?;\n]{0,55}?\b(?:to|by)\s+(\d+(?:\.\d+)?)\s*(?:°|degrees?\b)", re.I)
_DIRECTIONS = {
    "toward": re.compile(r"\btowards?\s+(?:the\s+)?(?:viewer|camera|front)\b", re.I),
    "away": re.compile(r"\b(?:away\s+from\s+(?:the\s+)?(?:viewer|camera)|towards?\s+(?:the\s+)?(?:rear|back))\b", re.I),
}
_NOUN_WORD = r"(?!(?:and|or|with|without|for|of|under|beneath|behind)\b)[A-Za-z][\w-]*"
_COMPOSITION = re.compile(
    rf"\b(?:show|reveal)\s+((?:an?|the)\s+{_NOUN_WORD}(?:\s+{_NOUN_WORD}){{0,3}})\s+(?:under|beneath|behind)\s+"
    rf"(?:an?|the)\s+((?:{_NOUN_WORD}\s+){{0,3}}(?:lid|panel|leaf|door|cover|flap|hatch))(?=\s*[.!?;]|$)",
    re.I,
)


def _role_id(role: str) -> str:
    return next(key for key, forms in _ROLES.items() if re.fullmatch(forms, role, re.I))


def _ids(text: str) -> Iterator[tuple[str, str]]:
    for grammar in (_DIRECT_ID, _INLINE_ID):
        for match in grammar.finditer(text):
            role, value = match.groups()
            yield _role_id(role), value
    for match in _ID_FIRST.finditer(text):
        for pair in _ID_FOR_ROLE.finditer(match[1]):
            value, role = pair.groups()
            yield _role_id(role), value
    for match in _ID_LIST.finditer(text):
        roles = _LIST_DELIMITER.split(match[1])
        names = _LIST_DELIMITER.split(match[2].strip())
        if len(roles) != len(names) or any(not re.fullmatch(_ID, name) for name in names):
            raise ValueError("Ambiguous element ID list")
        for role, name in zip(roles, names, strict=True):
            yield _role_id(re.sub(r"^the\s+", "", role, flags=re.I)), name.strip("`'\"“”‘’")


def _bound_fields(text: str, unquoted: str) -> Iterator[tuple[str, str]]:
    yield from _ids(unquoted)
    for key, prefix in _LABELS.items():
        for match in re.finditer(prefix + _QUOTE, text, re.I):
            if not unquoted[match.start()].isspace():
                yield key, next(value for value in match.groups() if value is not None)
    for match in _STATUS_PAIR.finditer(text):
        closed, opened = (value for value in match.groups() if value is not None)
        yield "closed_label", closed
        yield "open_label", opened
    yield from (("angle", match[1]) for match in _ANGLE.finditer(unquoted))
    yield from (("direction", name) for name, pattern in _DIRECTIONS.items() if pattern.search(unquoted))


def _mask_label(match: re.Match[str]) -> str:
    # A quoted token can name an ID or a file. Label prose cannot give directions.
    token = next(value for value in match.groups() if value is not None)
    if re.fullmatch(_ID, token) or components._HTML_FILENAME.fullmatch(token):
        return match[0]
    return " " * len(match[0])


def hinge_bindings(message: str) -> dict[str, str] | None:
    """Bind stated values for one new hinged panel. Leave the user's brief untouched."""
    text = message.strip()
    unquoted = _QUOTED_PROSE.sub(_mask_label, text)
    if "```" in text or _EXISTING.search(unquoted) or _SEPARATE_JOB.search(unquoted) or _UNSUPPORTED.search(unquoted):
        return None
    files = list(_HTML.finditer(unquoted))
    names = {match[1] or match[2] for match in files}
    if len(names) != 1 or not any(_CREATE.search(unquoted[:match.start()]) for match in files):
        return None
    edges = list(_EDGES_IN_BRIEF.finditer(unquoted))
    if len(edges) != 1 or _NEGATED.search(unquoted[:edges[0].start()]):
        return None
    filename, = names
    edge = (edges[0]["leading"] or edges[0]["trailing"]).lower()
    values = {"filename": filename, "edge": edge}
    try:
        for name, value in _bound_fields(text, unquoted):
            if name in values and values[name] != value:
                return None
            values[name] = value
        assemblies = {(inside, part) for inside, part in _COMPOSITION.findall(unquoted)}
        if len(assemblies) > 1:
            return None
        if assemblies:
            (inside, part), = assemblies
            values.setdefault("part", part)
            values.setdefault("detail", f"{inside[0].upper()}{inside[1:]} is behind the {part}.")
        kit = components.default_library().get(components.HINGED_PANEL)
        components._hinge_values(kit, values)
    except ValueError:
        return None
    return values
