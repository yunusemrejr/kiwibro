"""Is a reply that was written from sources backed by them?

When Kiwibro has put search results, a news bulletin or a tool's output in front of the
model, the answer has to come from there. A small model fills the gaps from memory: a
product that does not exist, a figure from another story, a person nobody mentioned, a
link it made up. Here a reply is read piece by piece (sentence, list item, table row);
the names, figures and links in each piece are looked up in the sources and in what the
user wrote. Code only: nothing here writes prose or calls a model.

    check(reply, sources, known)   the pieces that carry something the sources do not have
    strip(reply, findings)         the reply without those pieces ("" when too little is left)
    digest(items)                  the findings as the sources give them, for when nothing is left
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlsplit

from . import terms

# Words that are capitalised without naming anything a source has to back.
_CALENDAR = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday january february march april may june july "
    "august september october november december jan feb mar apr jun jul aug sep sept oct nov dec "
    "today yesterday tomorrow tonight".split()
)
# What a person is called by their office: "President Xi" is backed by a source that says "Xi".
_TITLES = frozenset(
    "mr mrs ms dr prof sir st jr sr president vice prime minister premier chancellor senator sen governor gov "
    "mayor king queen prince princess pope judge justice professor doctor general captain chairman "
    "chairwoman chair director secretary commissioner ambassador coach rep representative".split()
)
_COMMON = frozenset(
    "ai api app apps ceo cfo cto cpu gpu ram url pdf faq tv ok id q a i ii iii iv "
    "note notes summary source sources key top news update updates story stories overview highlights "
    "takeaways conclusion background context details result results answer question web internet "
    "gmt utc am pm "
    "kb mb gb tb kib mib gib tib hz khz mhz ghz km kg mph kph kwh mah usb ssd hdd os".split()  # units and parts, not names
)
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6, "billion": 1e9, "b": 1e9, "bn": 1e9,
          "trillion": 1e12, "t": 1e12, "tn": 1e12}
SMALL_NUMBER = 12  # counts and list positions up to here are the model's own to state
NEAR = 0.015  # a figure this close to one in the sources is that figure, rounded
KEEP_SHARE = 0.75  # less of the reply than this left after stripping: it is not an answer any more
ITEM_SHARED = 2  # content words a statement shares with a result for it to be about that result
MAX_LISTED = 8

_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)\s]*)\)")
_EMPHASIS = re.compile(r"(\*\*|\*|`|~~|(?<![^\W_])_+|_+(?![^\W_]))")  # not the "_" inside "my_file.pdf"
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(\[]?[A-Z0-9])")
_TOKEN = re.compile(r"[^\W_][\w'’.+&/-]*", re.UNICODE)
_NUMBER = re.compile(
    r"(?<![\w.,/:-])([$€£¥₺]\s?)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(\s?%|\s?(?:thousand|million|billion|"
    r"trillion|bn|mn|tn|k|m|b|t)\b)?(?![\w/:-]|\.\d|,\d)",
    re.I,
)
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")
_NAME_WORD = re.compile(r"[^\W\d_][\w'’.+&-]*|\d[\w.+-]*", re.UNICODE)
_JOINERS = frozenset({"of", "the", "de", "van", "von", "da", "del", "la", "le", "bin", "and", "for", "&"})
_SHAPED = re.compile(r"^(?:[A-Z]{2,}[\w.+&-]*|[^\W\d_]+[\w.+&-]*(?:[A-Z]|\d)[\w.+&-]*)$")  # "NASA", "OpenAI", "GPT-5"


@dataclass(frozen=True)
class Finding:
    line: int  # the reply line the piece is on
    piece: str  # the sentence, list item or table row as written
    missing: tuple[str, ...]  # what it says that the sources do not


@dataclass(frozen=True)
class _Piece:
    line: int
    text: str  # as written
    kind: str  # "heading", "item", "row", "sentence", "rule", "blank", "fence"


def _plain(text: str) -> str:
    """Markdown taken off: link text without its target, no emphasis marks."""
    return _EMPHASIS.sub("", _MD_LINK.sub(r"\1", text))


def _forms(token: str) -> set[str]:
    """The ways one written word may appear in a source: lower case, without a possessive,
    letters and digits only, its hyphen parts, its stem."""
    low = token.lower().replace("’", "'").strip(".'-+&/")
    if low.endswith("'s"):
        low = low[:-2]
    forms = {low}
    bare = re.sub(r"[^\w]", "", low)
    if bare:
        forms.add(bare)
        forms.add(terms.stem(bare))
    return {form for form in forms if form}


class _Sources:
    """The words, figures and links of everything the model was given."""

    def __init__(self, texts: Iterable[str]) -> None:
        self.words: set[str] = set()
        self.acronyms: set[str] = set()  # written in capitals in a source ("RSNA")
        self.initials: set[str] = set()  # the first letters of the names a source spells out
        self.values: list[float] = []
        self.figures: set[str] = set()
        self.urls: set[str] = set()
        self.hosts: set[str] = set()
        for text in texts:
            self._read(text or "")

    def _read(self, text: str) -> None:
        for url in _URL.findall(text):
            url = url.rstrip(".,;:!?")
            self.urls.add(url.lower().rstrip("/"))
            host = (urlsplit(url).hostname or "").lower()
            if host:
                self.hosts.add(host.removeprefix("www."))
        plain = _plain(text)
        previous = ""
        for token in _TOKEN.findall(plain):
            forms = _forms(token)
            self.words |= forms
            if len(token) >= 2 and token.isalpha() and token.isupper():
                self.acronyms.add(token.lower())
            elif re.fullmatch(r"(?:[A-Z]\.){2,5}", token) or re.fullmatch(r"(?:[A-Z]\.){1,4}[A-Z]", token):
                self.acronyms.add(token.replace(".", "").lower())  # "U.S." is "US"
                self.words.add(token.replace(".", "").lower())
            for part in re.split(r"[-./&+'’]", token):
                if part:
                    self.words |= _forms(part)
                    if len(part) >= 2 and part.isalpha() and part.isupper():
                        self.acronyms.add(part.lower())  # the "US" of "Sino-US" and of "the US."
            bare = re.sub(r"[^\w]", "", token.lower())
            if previous and bare:
                self.words.add(previous + bare)  # "GPT 5" also backs "GPT-5"
            previous = bare
        for value, written in _figures(plain):
            self.values.append(value)
            self.figures.add(written)
        for clock in _CLOCK.findall(plain):
            self.figures.add(clock)
        for run in _names(plain):
            short = _initials(run)
            if 2 <= len(short) <= 6:
                self.initials.add(short)  # "Northern Territory" also backs "NT"
            if len(short) >= 3:  # the names inside a longer one ("Italian Grand Prix" backs "GP", a headline's "United Nations" backs "UN")
                together: list[str] = []  # capitalised words with nothing between them
                for token in [*run, "the"]:
                    if token.lower() not in _JOINERS and token[0].isupper():
                        together.append(token)
                        continue
                    for size in (2, 3):
                        for at in range(len(together) - size + 1):
                            self.initials.add("".join(word[0] for word in together[at:at + size]).lower())
                    together = []

    def has_word(self, token: str) -> bool:
        if _forms(token) & self.words:
            return True
        parts = [part for part in re.split(r"[-/&+]", token) if part]
        if len(parts) > 1 and not any(ch.isdigit() for ch in token):
            # "AI-related", "Sino-US": the parts that name something each need backing
            named = [part for part in parts if part[0].isupper() and part.lower() not in _COMMON]
            return all(_forms(part) & self.words for part in named)
        return False

    def has_value(self, value: float, written: str) -> bool:
        if written in self.figures:
            return True
        for known in self.values:
            if known == value:
                return True
            if known and abs(known - value) <= NEAR * abs(known):
                return True
        return False

    def has_url(self, url: str) -> bool:
        url = url.lower().rstrip("/")
        if url in self.urls or any(found.startswith(url) for found in self.urls):
            return True
        parts = urlsplit(url)
        host = (parts.hostname or "").removeprefix("www.")
        return bool(host) and host in self.hosts and parts.path in ("", "/")  # a site's front page


def _split_run(run: list[str]) -> list[list[str]]:
    """The names a run joins with "and" or "&"."""
    parts: list[list[str]] = [[]]
    for token in run:
        if token.lower() in ("and", "&"):
            parts.append([])
        else:
            parts[-1].append(token)
    return [part for part in parts if part]


def _initials(run: list[str]) -> str:
    return "".join(token[0] for token in run if token.lower() not in _JOINERS and token[0].isalpha()).lower()


def _figures(text: str) -> list[tuple[float, str]]:
    """(value, as written without commas and spaces) for every figure in a text."""
    out: list[tuple[float, str]] = []
    for match in _NUMBER.finditer(text):
        digits = match.group(2).replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        tail = (match.group(3) or "").strip().lower()
        if tail and tail != "%":
            value *= _SCALE.get(tail, 1.0)
        out.append((value, digits))
    return out


def _pieces(reply: str) -> list[_Piece]:
    pieces: list[_Piece] = []
    fenced = False
    lines = reply.split("\n")
    for number, line in enumerate(lines):
        stripped = line.strip()
        below = lines[number + 1] if number + 1 < len(lines) else ""
        if stripped.startswith("```"):
            fenced = not fenced
            pieces.append(_Piece(number, line, "fence"))
        elif fenced:
            pieces.append(_Piece(number, line, "fence"))
        elif not stripped:
            pieces.append(_Piece(number, line, "blank"))
        elif _HEADING.match(line) or _is_label(stripped):
            pieces.append(_Piece(number, line, "heading"))
        elif _TABLE_RULE.match(line) and "-" in line and "|" in line:
            pieces.append(_Piece(number, line, "rule"))
        elif stripped.startswith("|") or stripped.count("|") >= 2:
            head = bool(_TABLE_RULE.match(below)) and "-" in below and "|" in below
            pieces.append(_Piece(number, line, "heading" if head else "row"))
        elif _LIST_ITEM.match(line) or stripped.startswith(">"):
            pieces.append(_Piece(number, line, "item"))
        else:
            for sentence in _SENTENCE_END.split(line):
                if sentence.strip():
                    pieces.append(_Piece(number, sentence, "sentence"))
    return pieces


def _is_label(stripped: str) -> bool:
    """A short line that is only a bold label ("**Key points:**"): it heads what follows."""
    if not (stripped.startswith(("**", "__")) and stripped.rstrip(":").endswith(("**", "__"))):
        return False
    inner = re.sub(r"^\d{1,3}[.)]\s*", "", stripped.strip("*_: "))
    return len(inner.split()) <= 10 and not re.search(r"[.!?]\s", inner)


def _names(text: str, openers: list[str] | None = None) -> list[list[str]]:
    """Runs of capitalised words that name something. A word capitalised because it opens a
    sentence or a table cell may or may not be part of a name: those are collected in
    ``openers`` so the caller can leave them out when nothing backs them."""
    runs: list[list[str]] = []
    for part in re.split(r"(?<=[.!?:;])\s+|\s+[—–|]\s+|\s*\|\s*|\n+|[\"“”()]", text):
        tokens = _NAME_WORD.findall(part)
        run: list[str] = []
        for index, token in enumerate(tokens):
            low = token.lower().strip(".'’-")
            capital = token[0].isupper() and low not in terms.STOPWORDS
            if capital and index == 0 and not _SHAPED.match(token):
                followed = len(tokens) > 1 and tokens[1][0].isupper() and tokens[1].lower() not in terms.STOPWORDS
                if not followed:
                    capital = False  # a plain sentence opener
                elif openers is not None:
                    openers.append(token)
            if capital or (run and low in _JOINERS):
                run.append(token)
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)
    cleaned = []
    for run in runs:
        while run and run[-1].lower() in _JOINERS:
            run.pop()
        if run:
            cleaned.append(run)
    return cleaned


def _missing(piece: _Piece, sources: _Sources, known: _Sources) -> list[str]:
    text = piece.text
    found: list[str] = []
    for url in _URL.findall(text):
        url = url.rstrip(".,;:!?*")
        if not (sources.has_url(url) or known.has_url(url)):
            found.append(url)
    plain = _URL.sub(" ", _plain(_MD_LINK.sub(r"\1", text)))
    plain = _LIST_ITEM.sub("", plain, count=1) if piece.kind == "item" else plain
    openers: list[str] = []
    for run in _names(plain, openers):
        lacking = []
        for position, token in enumerate(run):
            low = token.lower().strip(".'’-")
            if low in _JOINERS or low in _CALENDAR or low in _TITLES or low in _COMMON or len(low) < 2:
                continue
            if low.replace(".", "").replace(",", "").isdigit():
                continue  # figures are checked on their own
            if piece.kind == "heading" and not _SHAPED.match(token):
                continue  # title case says nothing in a heading: only what is a name by its shape counts
            if sources.has_word(token) or known.has_word(token):
                continue
            if token.isalpha() and token.isupper() and (low in sources.initials or low in known.initials):
                continue  # initials of a name the sources spell out
            if position == 0 and token in openers:
                continue  # most likely capitalised for opening the sentence
            lacking.append(token.strip(".,'’"))
        short = _initials(run)
        if lacking and len(short) >= 3 and (short in sources.acronyms or short in known.acronyms):
            lacking = []  # the sources give it by its initials ("RSNA")
        if lacking:  # "United States and China": each name of the run may be given by its initials ("US")
            for part in _split_run(run):
                short = _initials(part)
                if len(part) >= 2 and len(short) >= 2 and (short in sources.acronyms or short in known.acronyms):
                    lacking = [token for token in lacking if token not in {word.strip(".,'’") for word in part}]
        if lacking:
            shown = [token for token in run if not (token is run[0] and token in openers and token.strip(".,'’") not in lacking)]
            name = " ".join(shown).strip(".,'’ ")
            found.append(name if len(shown) <= 4 else " ".join(lacking))
    for clock in _CLOCK.findall(plain):
        if clock not in sources.figures and clock not in known.figures:
            found.append(clock)
    for match in _NUMBER.finditer(_CLOCK.sub(" ", plain)):
        digits = match.group(2).replace(",", "")
        tail = (match.group(3) or "").strip().lower()
        try:
            value = float(digits)
        except ValueError:
            continue
        scaled = value * _SCALE.get(tail, 1.0) if tail and tail != "%" else value
        plain_count = not match.group(1) and not tail and "." not in digits
        if plain_count and value <= SMALL_NUMBER:
            continue
        if sources.has_value(scaled, digits) or known.has_value(scaled, digits):
            continue
        found.append(match.group(0).strip())
    return list(dict.fromkeys(found))


_COUNTED = re.compile(r"(?<![\w.,])(\d[\d,]*(?:\.\d+)?)[\s-]+([^\W\d_]{3,})")
_COMPOUND = re.compile(r";|\b(?:and|while|whereas|but|meanwhile|also)\b", re.I)


def _counted(text: str) -> dict[str, set[str]]:
    """What each figure of a text counts: {"year": {"13"}} for "a 13-year-old"."""
    found: dict[str, set[str]] = {}
    for match in _COUNTED.finditer(text):
        word = match.group(2).lower()
        if word not in _CALENDAR and word not in _SCALE:
            found.setdefault(terms.stem(word), set()).add(match.group(1).replace(",", ""))
    return found


class _Result:
    """One search result, news item or listed line: what a single statement may be about."""

    def __init__(self, text: str) -> None:
        plain = _plain(text)
        self.sources = _Sources([text])
        self.content = {terms.stem(word) for word in terms.content_words(plain)}
        self.counted = _counted(plain)


def _apart(piece: _Piece, results: list[_Result], known: _Sources) -> list[str]:
    """What a statement says that the result it is about does not: a name or figure taken
    from another result ("ASIC approves Adobe's prospectus"), or a figure that result gives
    differently ("an 11-year-old" for "a 13-year-old"). [] when one result backs all of it,
    or two do for a statement that joins two."""
    plain = _plain(piece.text)
    about = {terms.stem(word) for word in terms.content_words(plain)}
    ranked = sorted(results, key=lambda result: len(about & result.content), reverse=True)[:2]
    ranked = [result for result in ranked if len(about & result.content) >= ITEM_SHARED]
    if not ranked:
        return []  # about no result in particular
    said = _counted(plain)

    def differs(result: _Result) -> list[str]:
        return [f"{' / '.join(sorted(values))} {word}" for word, values in said.items()
                if word in result.counted and not values & result.counted[word]]

    for result in ranked:
        if not _missing(piece, result.sources, known) and not differs(result):
            return []
    if len(ranked) == 2 and _COMPOUND.search(plain):
        both = _Sources([])
        for result in ranked:
            both.words |= result.sources.words
            both.acronyms |= result.sources.acronyms
            both.initials |= result.sources.initials
            both.values += result.sources.values
            both.figures |= result.sources.figures
            both.urls |= result.sources.urls
            both.hosts |= result.sources.hosts
        if not _missing(piece, both, known) and not (differs(ranked[0]) and differs(ranked[1])):
            return []
    first = ranked[0]
    return list(dict.fromkeys([*_missing(piece, first.sources, known), *differs(first)]))


def check(reply: str, sources: Iterable[str], known: Iterable[str] = (), results: Iterable[str] = ()) -> list[Finding]:
    """The pieces of ``reply`` that name or count something found neither in ``sources``
    (what the model was given to answer from) nor in ``known`` (the user's own words, the
    date, saved memory). With ``results`` (the same sources, one text per search result or
    news item) a statement must also hold within the result it is about: a small model's
    usual invention is to join the names of two stories into a third that never happened."""
    given = _Sources(sources)
    if not given.words:
        return []
    own = _Sources(known)
    separate = [_Result(text) for text in results if (text or "").strip()]
    findings: list[Finding] = []
    for piece in _pieces(reply or ""):
        if piece.kind not in ("item", "row", "sentence", "heading"):
            continue
        missing = _missing(piece, given, own)
        if not missing and len(separate) >= 2 and piece.kind != "heading":
            missing = _apart(piece, separate, own)
        if missing:
            findings.append(Finding(piece.line, piece.text, tuple(missing)))
    return findings


def results_in(texts: Iterable[str]) -> list[str]:
    """The separate results inside tool output that came as one text: its list lines
    ("- headline (outlet, time). first sentence") and numbered results with what follows them."""
    found: list[str] = []
    for text in texts:
        block: list[str] = []
        for line in (text or "").split("\n"):
            if _LIST_ITEM.match(line) or re.match(r"^\s*\[\d+\]\s", line):
                if block:
                    found.append(" ".join(block))
                block = [line.strip()]
            elif block and line.strip():
                block.append(line.strip())
            elif block:
                found.append(" ".join(block))
                block = []
        if block:
            found.append(" ".join(block))
    return found


def headlines(texts: Iterable[str], limit: int = 6) -> list[str]:
    """What the results are titled, as the sources wrote it (for showing them in one line each)."""
    out: list[str] = []
    for result in results_in(texts):
        line = _LIST_ITEM.sub("", result, count=1)
        linked = _MD_LINK.search(line)
        title = linked.group(1) if linked else re.split(r"\s+\(|\s+https?://|(?<=[.!?])\s", line, maxsplit=1)[0]
        title = _EMPHASIS.sub("", title).strip(" .")
        if len(title.split()) >= 3 and title not in out:
            out.append(title)
        if len(out) >= limit:
            break
    return out


def missing(findings: Iterable[Finding]) -> list[str]:
    """What the findings name, once each, in the order written."""
    out: list[str] = []
    for finding in findings:
        for item in finding.missing:
            if item not in out:
                out.append(item)
    return out


def listed(findings: Iterable[Finding]) -> str:
    items = missing(findings)
    shown = ", ".join(f"“{item}”" for item in items[:MAX_LISTED])
    return shown + (" and more" if len(items) > MAX_LISTED else "")


_POINTS_BACK = re.compile(r"^\W*(?:this|that|these|those|it|they|he|she|his|her|their|its)\b", re.I)


def strip(reply: str, findings: Iterable[Finding]) -> str:
    """``reply`` without the pieces the findings name. Headings left with nothing under them
    and tables left without rows go too. "" when what is left is no longer an answer."""
    bad: dict[int, set[str]] = {}
    for finding in findings:
        bad.setdefault(finding.line, set()).add(finding.piece)
    if not bad:
        return reply
    lines = reply.split("\n")
    total = kept = 0
    out: list[str | None] = []
    for number, line in enumerate(lines):
        content = [piece for piece in _pieces(line) if piece.kind in ("item", "row", "sentence")]
        if not content:
            out.append(None if number in bad else line)  # a heading that names what no source has
            continue
        total += len(content)
        dropped = set(bad.get(number, set()))
        for before, piece in zip(content, content[1:]):
            if before.text in dropped and _POINTS_BACK.match(piece.text):
                dropped.add(piece.text)  # "This follows …" says nothing once what it follows is gone
        left = [piece.text for piece in content if piece.text not in dropped]
        kept += len(left)
        if len(left) == len(content):
            out.append(line)
        elif not left:
            out.append(None)
        else:
            out.append(" ".join(part.strip() for part in left))
    if not total or kept < max(1, round(total * KEEP_SHARE)):
        return ""
    return _tidy([line for line in out if line is not None])


def _tidy(lines: list[str]) -> str:
    """Drop headings, labels and table heads left with nothing under them."""
    result: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        heading = bool(_HEADING.match(line)) or _is_label(stripped)
        if heading:
            below = next((later for later in lines[index + 1:] if later.strip()), "")
            if not below or _HEADING.match(below) or _is_label(below.strip()) or below.strip() in ("---", "***"):
                continue
        if stripped.startswith("|") and index + 1 < len(lines) and _TABLE_RULE.match(lines[index + 1]):
            after = lines[index + 2] if index + 2 < len(lines) else ""
            if not after.strip().startswith("|"):
                continue  # a table head whose rows are gone
        if _TABLE_RULE.match(line) and "|" in line and (not result or not result[-1].strip().startswith("|")):
            continue
        result.append(line)
    text = "\n".join(result)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*(?:---|\*\*\*)\s*$\n?(?=\s*(?:---|\*\*\*)?\s*$)", "", text)
    return text.strip()


def tidy(lines: list[str]) -> str:
    """Lines put back together as a reply, without a heading or lead-in left with nothing under it."""
    return _tidy(lines)


DIGEST_HEAD = "Here is what the sources say:"
DIGEST_ITEMS = 5
DIGEST_CHARS = 320


def digest(items: list[dict[str, Any]], head: str = DIGEST_HEAD) -> str:
    """The findings as the sources give them: one line per source with its own words and
    its link. What Kiwibro shows when no model's wording of them could be trusted."""
    lines: list[str] = []
    seen: set[str] = set()
    for item in items:
        title = " ".join(str(item.get("title") or "").split())
        url = str(item.get("url") or "").strip()
        said = " ".join(str(item.get("passage") or item.get("snippet") or "").split())
        if not (title or said) or url in seen:
            continue
        seen.add(url)
        if len(said) > DIGEST_CHARS:
            said = said[: DIGEST_CHARS - 1].rsplit(" ", 1)[0] + "…"
        name = title.replace("[", "(").replace("]", ")") or url
        label = f"[{name}]({url})" if url else f"**{name}**"
        lines.append(f"- {label}" + (f": {said}" if said else ""))
        if len(lines) >= DIGEST_ITEMS:
            break
    return f"{head}\n\n" + "\n".join(lines) if lines else ""
