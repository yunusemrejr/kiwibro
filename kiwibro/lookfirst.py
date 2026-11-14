"""What Kiwibro looks at on the computer before the model writes a word.

"How many files are on my desktop", "how much free disk space do I have", "what is eating my
disk": a small model answers these from nothing, with made-up counts or a template. The answer
is one read-only call away, so Kiwibro makes that call first and the model writes from the result.
Only read-only tools are used, and every call still passes the safety policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import plan

FOLLOW_WORDS = 12  # a follow-up about the same folder is short

_PART = re.compile(
    r"\b(?:cpu|processor|cores?|ram|memory|gpu|graphics\s+card|vram|disk(?:\s+space)?|storage|free\s+space|"
    r"battery|uptime|operating\s+system|os\s+version|kernel|distro|hostname|specs?|hardware)\b", re.I)
_THIS_MACHINE = re.compile(
    r"\b(?:this|my|our|the)\s+(?:computer|pc|laptop|machine|system|device|notebook|workstation)\b|"
    r"\b(?:do|does)\s+(?:i|we|it|this)\s+have\b|\bam\s+i\s+(?:running|using|on)\b|"
    r"\b(?:my|our)\s+(?:cpu|processor|ram|gpu|disk|storage|battery|os|kernel|specs?|hardware)\b|"
    r"\bhow\s+(?:much|many)\b.*\b(?:free|left|available|installed|used)\b", re.I)
_PLACES = "desktop|documents|downloads|pictures|music|videos"
_PLACE = re.compile(
    rf"\b(?:on|in|inside|from|at|under|within|of)\s+my\s+(?:(?:computer|pc|laptop|machine)(?:'s|s)?\s+)?({_PLACES})\b|"
    rf"\b(?:the|this|my)\s+(?:computer|pc|laptop|machine)(?:'s|s)?\s+({_PLACES})\b|"
    rf"\b(?:on|in|inside|from)\s+the\s+(desktop|downloads)\b|"
    rf"\b(?:the|my)\s+({_PLACES})\s+(?:folder|directory)\b|\bmy\s+(desktop|documents|downloads)\b", re.I)
_ASKS = re.compile(
    r"\b(?:how\s+many|what|which|list|show|find|is\s+there|are\s+there|do\s+i\s+have|biggest|largest|smallest|"
    r"newest|oldest|latest|count)\b", re.I)
_POINTER = re.compile(r"\b(?:which\s+(?:one|ones|file|folder)|of\s+(?:them|those|these)|them|those|the\s+(?:biggest|largest|smallest|newest|oldest))\b", re.I)

# "what's eating my disk", "why is my disk full", "which folders are the biggest", "what used up the
# space this week": the question is what fills the disk, which only a measure of the folders answers.
# Free space alone ("how much space is left") is system_info's.
_EATS = (r"(?:ea(?:t|ts|ting|ten)|ate|tak(?:e|es|ing|en)|took|us(?:e|es|ing|ed)|fill(?:s|ed|ing)?|hogg?(?:s|ed|ing)?|"
         r"clogg?(?:s|ed|ing)?|consum(?:e|es|ed|ing)|gobbl(?:e|es|ed|ing)|occup(?:y|ies|ied|ying)|swallow(?:s|ed|ing)?)")
# "space" alone is also the sky's: it counts as the disk's when it is "the space", "my space" or the computer's
_STORE = (r"(?:disk|drive|ssd|hdd|storage|(?:the|my|our)\s+space|"
          rf"(?:most\s+)?space\s+(?:on|in|of)\s+(?:my|the|this|our)\s+(?:computer|pc|laptop|machine|notebook|system|device|home|{_PLACES}))")
_SPACE_EATEN = re.compile(
    rf"\b(?:what|which|who|whats|what's)\b[^?.!]{{0,40}}?\b{_EATS}\b(?:\s+up)?[^?.!]{{0,40}}?\b{_STORE}\b", re.I)
_DISK_FULL = re.compile(
    rf"\bwhy\b[^?.!]{{0,40}}?\b{_STORE}\b[^?.!]{{0,30}}?\b(?:full|filling|filled|low|tight|gone)\b|"
    r"\bwhy\b[^?.!]{0,30}?\b(?:out\s+of|low\s+on|short\s+on|running\s+out\s+of)\s+(?:disk|storage|space)\b", re.I)
_SPACE_WENT = re.compile(
    rf"\bwhere\b[^?.!]{{0,30}}?\b{_STORE}\b[^?.!]{{0,20}}?\b(?:go|gone|went|disappear\w*|vanish\w*)\b", re.I)
_BIGGEST_FOLDERS = re.compile(
    r"\b(?:biggest|largest|heaviest|fattest)\s+(?:folders?|directories|dirs?|space\s+hogs?)\b|"
    r"\b(?:folders?|directories|dirs?)\b[^?.!]{0,30}?\b(?:biggest|largest|heaviest|most\s+space)\b|"
    r"\bspace\s+hogs?\b", re.I)
_FULL_AND_ASKING = re.compile(
    r"\b(?:disk|drive|storage|ssd)\b\W+(?:\w+\W+){0,3}?(?:full|filling\s+up|running\s+out)\b", re.I)
_ELSEWHERE = re.compile(  # a measure of the home folder says nothing about these
    r"\b(?:this|the|my|our)\s+(?:repo|repository|project|workspace|codebase|container|database|db|vm|image|"
    r"notes?|chat|memory|memories|models?|library)\b|\bnode_modules\b|\bdocker\b", re.I)
_TODAY = re.compile(r"\btoday\b|\bthis\s+morning\b|\bsince\s+(?:midnight|this\s+morning)\b", re.I)
_THIS_WEEK = re.compile(
    r"\b(?:this|the\s+(?:past|last))\s+week\b|\blast\s+(?:few\s+|\d+\s+)?days\b|\bpast\s+(?:few\s+|\d+\s+)?days\b|"
    r"\bthese\s+days\b|\brecently\b|\blately\b|\byesterday\b", re.I)
_THIS_MONTH = re.compile(r"\b(?:this|the\s+(?:past|last))\s+month\b|\blast\s+30\s+days\b", re.I)
_ASKS_ABOUT_DISK = re.compile(r"\?|\b(?:what|which|why|where|who|how|find|show|tell|list|check|help)\b", re.I)
_SPACE_QUESTIONS = (_SPACE_EATEN, _DISK_FULL, _SPACE_WENT, _BIGGEST_FOLDERS)
_FILES = re.compile(r"\bfiles?\b", re.I)
_FOLDERS = re.compile(r"\b(?:folders?|directories|dirs?)\b", re.I)


def asks_what_fills_the_disk(message: str) -> bool:
    """The message wants to know what takes up the space on the user's own disk."""
    text = (message or "").replace("’", "'")
    if _ELSEWHERE.search(text):
        return False
    return any(pattern.search(text) for pattern in _SPACE_QUESTIONS) or bool(
        _FULL_AND_ASKING.search(text) and _ASKS_ABOUT_DISK.search(text))


def _changed(message: str) -> str | None:
    """How far back a question about the disk looks: what was written then, not everything."""
    for window, pattern in (("today", _TODAY), ("week", _THIS_WEEK), ("month", _THIS_MONTH)):
        if pattern.search(message):
            return window
    return None


@dataclass(frozen=True)
class Look:
    name: str  # a read-only tool
    args: dict[str, Any]
    folder: Path | None = None  # a folder the user named outside the workspace: the call runs there
    shown: str = ""


def machine(message: str) -> bool:
    """The message asks about this computer's own hardware or system."""
    text = (message or "").replace("’", "'")
    return bool(_PART.search(text) and _THIS_MACHINE.search(text))


def place(message: str, workspace_root: Path) -> plan.Target | None:
    """The user's folder a message names (Desktop, Documents, ...), if it exists and may be read."""
    found = _PLACE.search((message or "").replace("’", "'"))
    if not found:
        return None
    folder = plan._xdg_dir(next(group for group in found.groups() if group))
    if not folder.is_dir():
        return None
    return plan._checked(folder, True, Path(workspace_root))


# "is my local copy at the same commit as github.com/me/app?" A small model answers that with a
# guess or with instructions; git answers it outright, so Kiwibro asks git first.
_REPO = re.compile(r"\b(?:repo(?:s|sitory|sitories)?|github|gitlab|git|commits?|branch(?:es)?|checkout|clone|origin|"
                   r"upstream|remote|push(?:ed)?)\b", re.I)
_SAME = re.compile(
    r"\bsame\s+(?:commit|version|revision|state|code|thing|as)\b|\bthe\s+same\s*[?.!]*\s*$|\bidentical\b|\bin\s+sync\b|"
    r"\bsync(?:ed|hroni[sz]ed)\b|\bup\s*-?\s*to\s*-?\s*date\b|\bout\s+of\s+date\b|\bbehind\b|\bahead\b|\bmatch(?:es)?\b|"
    r"\b(?:did|have|has)\b.*\bpush(?:ed)?\b|\bpushed\s+everything\b", re.I)
_CHECKS = re.compile(r"\?|^\s*(?:is|are|did|do|does|have|has|am|check|compare|verify|see|tell\s+me)\b|\b(?:whether|if)\b", re.I)
_OWN_COPY = re.compile(r"\b(?:my|our|local|this|here)\b", re.I)
_REMOTE = re.compile(r"https://[A-Za-z0-9][A-Za-z0-9.-]*/[\w./~-]+|\bgithub\.com/[A-Za-z0-9_][\w.-]*/[A-Za-z0-9_][\w.-]*", re.ASCII)


def repo(message: str, workspace_root: Path) -> Look | None:
    """The comparison a message asks for between the user's checkout and its remote. The
    checkout is the folder the message names, else the workspace; None when that is not a
    git checkout, so a question that only sounds like one costs nothing."""
    text = (message or "").replace("’", "'")
    if not (_REPO.search(text) and _SAME.search(text) and _OWN_COPY.search(text) and _CHECKS.search(text)):
        return None
    address = _REMOTE.search(text)
    remote = "origin"
    if address:
        remote = address.group(0).rstrip(".,;:!?)")
        remote = remote if remote.startswith("https://") else f"https://{remote}"
        text = text.replace(address.group(0), " ")
    target = plan._target(text, "read", Path(workspace_root))
    folder = target.path if target is not None and target.is_dir else Path(workspace_root)
    if not (folder / ".git").exists():
        return None
    args = {"action": "compare", "remote": remote}
    if target is not None and target.is_dir and target.outside:
        return Look("git", {**args, "path": "."}, folder, target.shown)
    return Look("git", {**args, "path": target.shown.rstrip("/") or "." if target is not None and target.is_dir else "."})


def calls(message: str, workspace_root: Path, tools: frozenset[str], before: str = "") -> list[Look]:
    """The looks a message needs. ``before`` is the user's previous message: a short question
    pointing back at a folder it named ("which one is the biggest") looks there again."""
    looks: list[Look] = []
    compare = repo(message, workspace_root) if "git" in tools else None
    if compare is not None:
        return [compare]
    if "disk_usage" in tools and asks_what_fills_the_disk(message):
        target = place(message, workspace_root)  # "what takes the space in my downloads" measures that folder
        if not (target is not None and _FILES.search(message) and not _FOLDERS.search(message)):  # "which file takes the most space on my desktop" is the folder listing's
            window = _changed(message)
            args: dict[str, Any] = {**({"path": str(target.path)} if target is not None else {}), **({"changed": window} if window else {})}
            return [Look("disk_usage", args, shown=target.shown if target is not None else "")]
    if "system_info" in tools and machine(message):
        looks.append(Look("system_info", {}))
    if "list_files" in tools:
        target = place(message, workspace_root) if _ASKS.search(message or "") else None
        if target is None and before and len(message.split()) <= FOLLOW_WORDS and _POINTER.search(message):
            target = place(before, workspace_root)
        if target is not None:
            looks.append(Look("list_files", {"path": "."}, target.path if target.outside else None, target.shown)
                         if target.outside else Look("list_files", {"path": target.shown}))
    return looks


# --- the answer, when the look itself is the answer ---------------------------
# A small model asked to read a 200-line listing reports the wrong one of its numbers. When
# the question is one the result answers outright (how many, the biggest, which CPU), Kiwibro
# writes the answer from the result's own figures and the model is not asked. That holds only
# when every word of the question is accounted for: one word more ("how many PDFs", "is my CPU
# good for games") and the question wants something the figures alone do not say.
_HOW_MANY = re.compile(r"\bhow\s+many\b|\bcount\b|\bnumber\s+of\b", re.I)
_BIGGEST = re.compile(r"\b(?:biggest|largest|heaviest)\b|\btakes?\s+(?:up\s+)?the\s+most\s+space\b", re.I)
_NEWEST = re.compile(r"\b(?:newest|most\s+recent(?:ly)?|last\s+(?:changed|modified|edited|added))\b", re.I)
_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cpu", re.compile(r"\b(?:cpu|processor|cores?)\b", re.I)),
    ("ram", re.compile(r"\b(?:ram|memory)\b", re.I)),
    ("gpu", re.compile(r"\b(?:gpu|graphics\s+card|vram)\b", re.I)),
    ("disk", re.compile(r"\b(?:disk(?:\s+space)?|storage|free\s+space)\b", re.I)),
    ("battery", re.compile(r"\bbattery\b", re.I)),
    ("uptime", re.compile(r"\buptime\b", re.I)),
    ("os", re.compile(r"\b(?:operating\s+system|os\s+version|kernel|distro)\b", re.I)),
    ("hostname", re.compile(r"\bhostname\b", re.I)),
)
_ALL_HARDWARE = re.compile(r"\b(?:specs?|hardware)\b", re.I)
_GLUE = frozenset(
    "a an the is are was there do does did i we you it have has got on in of my our this that what whats which one "
    "ones tell me us please pls can could would hey hi kiwibro and or total exactly right now currently how much "
    "many so just check show see look at to s kind type model name called here computer pc laptop machine system "
    "device with for".split())
_ABOUT_FILES = frozenset("file files folder folders item items thing things directory directories stuff".split())
_ABOUT_PARTS = frozenset("free left available installed space running using am size amount inside".split())


def _only(text: str, patterns: tuple[re.Pattern[str], ...], known: frozenset[str]) -> bool:
    """Nothing is left of the question once what it asks for and its small words are taken out."""
    for pattern in patterns:
        text = pattern.sub(" ", text)
    return all(word in _GLUE or word in known for word in re.findall(r"[a-z0-9]+", text.lower().replace("'", "")))


def _machine_answer(message: str, info: dict[str, Any]) -> str:
    asked = [name for name, pattern in _FAMILIES if pattern.search(message)]
    if _ALL_HARDWARE.search(message):
        asked = list(dict.fromkeys([*asked, "cpu", "ram", "gpu", "disk"]))
    home = bool(info.get("home_disk_free"))
    disk = (info.get("home_disk_free"), info.get("home_disk_total")) if home else (info.get("disk_free"), info.get("disk_total"))
    said = {  # part -> (in a list, as a sentence); only the parts that could be read
        "cpu": info.get("cpu_model") and (
            f"CPU: {info['cpu_model']}" + (f" ({info['cpu_cores']})" if info.get("cpu_cores") else ""),
            f"The CPU is {info['cpu_model']}" + (f" ({info['cpu_cores']})." if info.get("cpu_cores") else ".")),
        "ram": info.get("memory_total") and (
            f"RAM: {info['memory_total']}" + (f", {info['memory_available']} available right now" if info.get("memory_available") else ""),
            f"This computer has {info['memory_total']} of RAM"
            + (f"; {info['memory_available']} of it is available right now." if info.get("memory_available") else ".")),
        "gpu": info.get("gpu") and (f"GPU: {info['gpu']}", f"The GPU is {info['gpu']}."),
        "disk": disk[0] and disk[1] and (
            f"Disk: {disk[0]} free of {disk[1]}",
            f"{disk[0]} is free of {disk[1]} on the disk " + ("your home folder is on." if home else "Kiwibro works on.")),
        "battery": info.get("battery") and (f"Battery: {info['battery']}", f"Battery: {info['battery']}."),
        "uptime": info.get("uptime") and (f"Uptime: {info['uptime']}", f"This computer has been up for {info['uptime']}."),
        "os": info.get("os") and (f"System: {info['os']}", f"This computer runs {info['os']}."),
        "hostname": info.get("hostname") and (f"Hostname: {info['hostname']}", f"The hostname is {info['hostname']}."),
    }
    if not asked or not all(said.get(name) for name in asked):
        return ""  # a part that was asked for could not be read: the model answers from what there is
    return said[asked[0]][1] if len(asked) == 1 else "\n".join(f"- {said[name][0]}" for name in asked)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _folder_answer(message: str, meta: dict[str, Any]) -> str:
    place = str(meta.get("shown") or "")
    place = "that folder" if place in (".", "") else place
    here = (int(meta.get("files_here", 0)), int(meta.get("folders_here", 0)))
    deep = (int(meta.get("files", 0)), int(meta.get("folders", 0)))
    said: list[str] = []
    if _HOW_MANY.search(message):
        said.append(f"{place[0].upper()}{place[1:]} has {_plural(here[0], 'file')} and {_plural(here[1], 'folder')} "
                    "directly in it (hidden ones not counted).")
        if deep != here:
            said.append(f"Counting what is inside those folders too, {meta.get('depth', 2)} levels deep: "
                        f"{_plural(deep[0], 'file')} and {_plural(deep[1], 'folder')}.")
    if _BIGGEST.search(message):
        if not (meta.get("largest") and meta.get("largest_size")):
            return ""
        said.append(f"The largest file in {place} is {meta['largest']} ({meta['largest_size']}).")
    if _NEWEST.search(message):
        if not (meta.get("newest") and meta.get("newest_time")):
            return ""
        said.append(f"The most recently changed in {place} is {meta['newest']} ({meta['newest_time']}).")
    return " ".join(said)


def _repo_answer(message: str, found: dict[str, Any]) -> str:
    """git's own verdict, when the message asks one thing; "" when the compare could not be made
    or the message asks for more."""
    lines = str(found.get("output") or "").split("\n")
    if found.get("same") is None or not lines[0].startswith("Same commit: ") or len(plan.questions(message)) > 1:
        return ""
    verdict = lines[0].replace("Same commit: yes.", "Yes, the same commit.", 1).replace("Same commit: no.", "No, not the same commit.", 1)
    said = [verdict] + [line for line in lines[1:] if line.startswith("The ")]
    more = [f"- Last commit in the local copy: {line[len('Local: '):]}" for line in lines if line.startswith("Local: ")]
    more += [f"- {line}" for line in lines if line.startswith("Uncommitted changes")]
    return " ".join(said) + ("\n\n" + "\n".join(more) if more else "")


# disk_usage's result is text: "~: 120 GiB in 25 entries[ changed since 14 Nov 00:00]", "Disk: 3 GiB free of 7 GiB",
# then one "   45 GiB   38%  Downloads/" row per entry and, for a time window, the largest files written in it.
_DISK_HEAD = re.compile(r"^(.+?): (\S+ \S+) in (\d+) entr(?:y|ies)(?: changed since (.+))?$")
_DISK_ROW = re.compile(r"^\s*(\S+ \S+)\s+(<1%|\d+%)\s+(\S.*)$")
_DISK_FILE = re.compile(r"^\s*(\S+ \S+)\s+((?:\d{2} \w{3} )?\d{2}:\d{2})\s+(\S.*)$")
_DISK_FREE = re.compile(r"^Disk: (\S+ \S+) free of (\S+ \S+)$")
_DISK_UNLISTED = re.compile(r"^\((\d+) more not listed")
_DISK_WORDS = frozenset("folder folders directory directories biggest largest heaviest most space disk drive storage hogs "
                        "full filling filled eating taking using used up go gone went where why did".split())


def _entries(count: int) -> str:
    return "1 entry" if count == 1 else f"{count} entries"


def _disk_answer(found: dict[str, Any]) -> str:
    """What fills a folder, in disk_usage's own figures and its own order (largest first); "" when its text
    is not in the form it always has."""
    lines = str(found.get("output") or "").splitlines()
    head = _DISK_HEAD.match(lines[0]) if lines else None
    if head is None:
        return ""
    shown, total, count, since = head.groups()
    where = "your home folder" if shown == "~" else shown
    rows = [f"- {m.group(3)}: {m.group(1)} ({m.group(2)})" for m in map(_DISK_ROW.match, lines[1:]) if m]
    files = [f"- {m.group(3)}: {m.group(1)}, {m.group(2)}" for m in map(_DISK_FILE.match, lines[1:]) if m]
    if not rows:
        return f"Nothing in {where} was written since {since}." if since else f"{where[0].upper()}{where[1:]} is empty."
    said = [f"Since {since}, {total} was written in {where}. Largest first:" if since
            else f"{where[0].upper()}{where[1:]} holds {total} in {_entries(int(count))}, largest first:"]
    said += rows
    unlisted = next((int(m.group(1)) for m in map(_DISK_UNLISTED.match, lines) if m), 0)
    if unlisted:
        said.append(f"{unlisted} more {'entry is' if unlisted == 1 else 'entries are'} smaller and not listed.")
    if files:
        said += ["", f"Largest files written since {since}:", *files]
    free = next((m for m in map(_DISK_FREE.match, lines) if m), None)
    if free:
        said += ["", f"{free.group(1)} is free of {free.group(2)} on this disk."]
    said += [line for line in lines if line.endswith("could not be read") or line.startswith("The walk was cut short")]
    return "\n".join(said)


def answer(message: str, looked: dict[str, dict[str, Any]]) -> str:
    """The answer to a question the looks answer outright, in the results' own figures; "" when
    the question wants more than that (a list, an opinion, a kind of file) and the model writes it."""
    text = (message or "").replace("’", "'")
    if "git" in looked:
        return _repo_answer(text, looked["git"])
    if "disk_usage" in looked:  # the question is only what the look measured: nothing to do, delete or compare
        return _disk_answer(looked["disk_usage"]) if _only(
            text, (*_SPACE_QUESTIONS, _FULL_AND_ASKING, _TODAY, _THIS_WEEK, _THIS_MONTH, _PLACE), _DISK_WORDS) else ""
    asks_machine, asks_folder = "system_info" in looked, "list_files" in looked
    patterns = (*((_PART, _THIS_MACHINE) if asks_machine else ()), *((_PLACE, _HOW_MANY, _BIGGEST, _NEWEST) if asks_folder else ()))
    known = (_ABOUT_PARTS if asks_machine else frozenset()) | (_ABOUT_FILES if asks_folder else frozenset())
    if not (asks_machine or asks_folder) or not _only(text, patterns, known):
        return ""
    parts = [_machine_answer(text, looked["system_info"])] if asks_machine else []
    if asks_folder:
        parts.append(_folder_answer(text, looked["list_files"]))
    return "\n\n".join(parts) if all(parts) else ""
