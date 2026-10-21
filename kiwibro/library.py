"""On-disk document library: stream, chunk and index long files for small models.

Long documents and long tool outputs never go into the prompt or RAM whole.
Each file is extracted section by section, cut into chunks of about 1000
characters and stored in SQLite with an FTS5 index. The model then searches
the index and reads a few chunks at a time through the ``doc_*`` tools.
Outputs too large for the context are parked with :meth:`Library.stash_text`.

The library lives in its own database file (the caller passes the path,
normally ``<data_dir>/library.db``); kiwibro.db is never touched.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import hashlib
import io
import json
import logging
import os
import posixpath
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

from .extensions import ExtensionTool, data_dir_for, schema, tool_result

log = logging.getLogger(__name__)

Section = tuple[str, str, str]  # (heading, location, text)
Extractor = Callable[[Path, dict[str, str]], Iterator[Section]]

MB = 1024 * 1024
TARGET_CHARS = 1000
MAX_CHARS = 1600
OVERLAP_CHARS = 100
SECTION_LINES = 200
SECTION_CHARS = 64_000
ROW_GROUP = 100
BATCH_ROWS = 200
MAX_CHUNKS = 20_000
MAX_OUTPUTS = 200
PER_DOC_HITS = 3
MAX_BYTES_PDF = 100 * MB
MAX_BYTES_OTHER = 30 * MB
PDF_TIMEOUT_S = 120
PDF_FALLBACK_BYTES = 50 * MB
JSON_MAX_BYTES = 8 * MB
HTML_MAX_BYTES = 8 * MB  # BeautifulSoup parses whole documents
SYNC_MAX_BYTES = 8 * MB  # bigger files are left to an explicit doc_add
CSV_MAX_ROWS = 50_000
XLSX_MAX_ROWS = 20_000
HASH_BLOCK = MB
SKIP_DIRS = frozenset({"node_modules", "__pycache__", ".git", ".venv", "venv", "dist", "build", ".cache"})
SYNC_EXTS = frozenset({".pdf", ".docx", ".odt", ".epub", ".pptx", ".xlsx", ".md", ".markdown", ".txt", ".rst", ".html", ".htm"})
STOPWORDS = frozenset(
    "a an and are as at be been but by can do does for from had has have how i if in into is it its "
    "of on or so that the their then there these this to was we were what when where which who why "
    "will with you your".split()
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    display TEXT NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    size INTEGER,
    mtime REAL,
    sha256 TEXT,
    chunks INTEGER,
    chars INTEGER,
    added_at TEXT,
    indexed_at TEXT,
    source TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    location TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc_seq ON chunks(doc_id, seq);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading, content='chunks', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading) VALUES ('delete', old.id, old.text, old.heading);
END;
"""

_INSERT_CHUNK = "INSERT INTO chunks (doc_id, seq, heading, location, text) VALUES (?, ?, ?, ?, ?)"
_HANDLE_RE = re.compile(r"[dD]?(\d+)")
_MD_HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.*?)(?:\s+#+)?\s*$")
_LOC_RE = re.compile(r"^(.*?)(\d+)(?:\s*-\s*(\d+))?$")
_CODE_EXTS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb",
    ".php", ".sh", ".sql", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".css", ".scss",
)

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_ODT_TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_S_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_OPF_NS = "{http://www.idpf.org/2007/opf}"
_CONTAINER_NS = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_SLIDE_RE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
_SHEET_RE = re.compile(r"^xl/worksheets/sheet(\d+)\.xml$")
_HTML_DROP = ["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]
_HTML_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_HTML_BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "td", "th"]
_PDF_STREAM_RE = re.compile(rb"(?<!end)stream\r?\n(.*?)\r?\nendstream", re.S)
_PDF_OP_RE = re.compile(
    rb"\(((?:\\.|[^\\()])*)\)\s*Tj"
    rb"|\[((?:\\.|[^\\\[\]])*)\]\s*TJ"
    rb"|\b(?:Td|TD|Tm|ET)\b|T\*"
)
_PDF_ARRAY_RE = re.compile(rb"\(((?:\\.|[^\\()])*)\)|(-?\d*\.?\d+)")
_PDF_ESCAPES = {
    ord("n"): b"\n", ord("r"): b"\r", ord("t"): b"\t", ord("b"): b"\b", ord("f"): b"\f",
}


# --- small helpers -----------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _squash(text: str) -> str:
    return " ".join(text.split())


def _clean(text: str, limit: int = 200) -> str:
    return _squash(text)[:limit]


def _lines_label(first: int, last: int) -> str:
    return f"lines {first}-{last}"


def _label(seq: int, location: str, heading: str) -> str:
    parts = [f"chunk {seq}"]
    if location:
        parts.append(location)
    if heading:
        parts.append(heading)
    return " · ".join(parts)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _span(first: str, last: str) -> str:
    """Location covering two section locations, e.g. "page 3" + "page 4" -> "page 3-4"."""
    if first == last or not last:
        return first
    if not first:
        return last
    left, right = _LOC_RE.match(first), _LOC_RE.match(last)
    if left and right and left.group(1) == right.group(1):
        low = int(left.group(2))
        high = int(right.group(3) or right.group(2))
        return f"{left.group(1)}{low}" if low == high else f"{left.group(1)}{low}-{high}"
    return f"{first} … {last}"


# --- chunking ----------------------------------------------------------------


def _hard_wrap(text: str, maximum: int) -> Iterator[str]:
    step = maximum - OVERLAP_CHARS
    start = 0
    while True:
        piece = text[start : start + maximum].strip()
        if piece:
            yield piece
        if start + maximum >= len(text):
            return
        start += step


def _split_sentences(text: str, maximum: int) -> Iterator[str]:
    """Pack sentences up to ``maximum``; a single giant sentence is hard-wrapped."""
    buffer = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > maximum:
            if buffer:
                yield buffer
                buffer = ""
            yield from _hard_wrap(sentence, maximum)
            continue
        candidate = f"{buffer} {sentence}" if buffer else sentence
        if len(candidate) > maximum:
            yield buffer
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        yield buffer


def _split_long(text: str, maximum: int) -> Iterator[str]:
    """Split an oversized paragraph on line boundaries (logs, wrapped prose), keeping whole lines together."""
    buffer = ""
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        if len(line) > maximum:
            if buffer:
                yield buffer
                buffer = ""
            yield from _split_sentences(line, maximum)
            continue
        candidate = f"{buffer}\n{line}" if buffer else line
        if len(candidate) > maximum:
            yield buffer
            buffer = line
        else:
            buffer = candidate
    if buffer:
        yield buffer


def _pieces(text: str, maximum: int) -> Iterator[str]:
    for paragraph in re.split(r"\n\s*\n", text or ""):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= maximum:
            yield paragraph
        else:
            yield from _split_long(paragraph, maximum)


def _emit(heading: str, first: str, last: str, parts: list[str]) -> Section:
    return heading, _span(first, last), "\n\n".join(parts)


def chunk_sections(
    sections: Iterable[Section], target: int = TARGET_CHARS, maximum: int = MAX_CHARS
) -> Iterator[Section]:
    """Pack paragraphs of ``(heading, location, text)`` sections into chunks.

    A chunk closes once it reaches ``target`` characters, never grows past
    ``maximum``, and always closes when the heading changes.
    """
    heading = ""
    first = last = ""
    parts: list[str] = []
    size = 0
    for sec_heading, sec_location, sec_text in sections:
        if sec_heading != heading:
            if parts:
                yield _emit(heading, first, last, parts)
                parts, size = [], 0
            heading = sec_heading
        for piece in _pieces(sec_text, maximum):
            if parts and size + len(piece) + 2 > maximum:
                yield _emit(heading, first, last, parts)
                parts, size = [], 0
            if not parts:
                first = sec_location
            parts.append(piece)
            size += len(piece) + 2
            last = sec_location
            if size >= target:
                yield _emit(heading, first, last, parts)
                parts, size = [], 0
    if parts:
        yield _emit(heading, first, last, parts)


# --- extraction: text, markdown, code, json ------------------------------------


def _read_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            yield line.rstrip("\r\n")


def _line_sections(lines: Iterable[str], *, markdown: bool) -> Iterator[Section]:
    """Group lines into sections of up to SECTION_LINES; markdown headings start new sections."""
    heading = ""
    start = 1
    buffer: list[str] = []
    size = 0
    fence = False
    number = 0
    for number, line in enumerate(lines, start=1):
        if markdown and line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        match = _MD_HEADING.match(line) if markdown and not fence else None
        if match:
            if buffer:
                yield heading, _lines_label(start, number - 1), "\n".join(buffer)
            heading = _clean(match.group(1))
            start, buffer, size = number, [], 0
            continue
        buffer.append(line)
        size += len(line)
        if len(buffer) >= SECTION_LINES or size >= SECTION_CHARS:
            yield heading, _lines_label(start, number), "\n".join(buffer)
            start, buffer, size = number + 1, [], 0
    if buffer:
        yield heading, _lines_label(start, number), "\n".join(buffer)


def _text_file(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    return _line_sections(_read_lines(path), markdown=False)


def _markdown_file(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    return _line_sections(_read_lines(path), markdown=True)


def _json_file(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    if path.stat().st_size <= JSON_MAX_BYTES:
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            data = None
        if data is not None:
            pretty = json.dumps(data, indent=1, ensure_ascii=False)
            yield from _line_sections(pretty.splitlines(), markdown=False)
            return
    yield from _line_sections(_read_lines(path), markdown=False)


def _csv_column(header: list[str], index: int) -> str:
    if index < len(header) and header[index]:
        return header[index]
    return f"col{index + 1}"


def _csv_file(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    header: list[str] = []
    lines: list[str] = []
    first = last = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for number, row in enumerate(csv.reader(fh, delimiter=delimiter), start=1):
                if number > CSV_MAX_ROWS:
                    meta["note"] = f"only the first {CSV_MAX_ROWS} rows were indexed"
                    break
                if not header:
                    header = [cell.strip() for cell in row]
                    continue
                pairs = [f"{_csv_column(header, i)}: {cell.strip()}" for i, cell in enumerate(row) if cell.strip()]
                if not pairs:
                    continue
                first = first or number
                last = number
                lines.append("; ".join(pairs))
                if len(lines) >= ROW_GROUP:
                    yield "", f"rows {first}-{last}", "\n\n".join(lines)
                    lines, first = [], 0
    except csv.Error as exc:
        meta["note"] = f"stopped reading at row {last + 1}: {exc}"
    if lines:
        yield "", f"rows {first}-{last}", "\n\n".join(lines)


# --- extraction: html, epub --------------------------------------------------


def _html_sections(markup: str, location: str = "") -> Iterator[Section]:
    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup.find_all(_HTML_DROP):
        tag.decompose()
    heading = ""
    emitted = False
    for element in soup.find_all(_HTML_BLOCKS):
        if element.find(_HTML_BLOCKS):
            continue  # a wrapper; its nested blocks are visited on their own
        if element.name in _HTML_HEADINGS:
            text = _clean(element.get_text(" ", strip=True))
            if text:
                heading = text
                emitted = True
                yield heading, location, ""
            continue
        # Keep line structure inside <pre>; collapse whitespace everywhere else.
        text = element.get_text().strip("\n") if element.name == "pre" else _squash(element.get_text())
        if text.strip():
            emitted = True
            yield heading, location, text
    if not emitted:
        fallback = soup.get_text("\n", strip=True)
        if fallback:
            yield "", location, fallback


def _html_file(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    if path.stat().st_size > HTML_MAX_BYTES:
        meta["note"] = "large HTML indexed as plain text"
        return _line_sections(_read_lines(path), markdown=False)
    return _html_sections(path.read_text(encoding="utf-8", errors="replace"))


def _epub_sections(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    with zipfile.ZipFile(path) as zf:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        rootfile = container.find(f".//{_CONTAINER_NS}rootfile")
        if rootfile is None:
            raise ValueError("EPUB has no package file")
        package = rootfile.get("full-path", "")
        opf = ET.fromstring(zf.read(package))
        base = posixpath.dirname(package)
        manifest = {item.get("id"): item.get("href", "") for item in opf.iter(f"{_OPF_NS}item")}
        spine = [ref.get("idref") for ref in opf.iter(f"{_OPF_NS}itemref")]
        chapters = [
            posixpath.normpath(posixpath.join(base, manifest[idref]))
            for idref in spine
            if idref in manifest
        ]
        for number, name in enumerate(chapters, start=1):
            try:
                raw = zf.read(name)
            except KeyError:
                continue
            yield from _html_sections(raw.decode("utf-8", errors="replace"), f"chapter {number}")


# --- extraction: office formats ----------------------------------------------


def _docx_sections(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    heading = ""
    with zipfile.ZipFile(path) as zf, zf.open("word/document.xml") as fh:
        for _, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag != _W_NS + "p":
                continue
            parts: list[str] = []
            for node in elem.iter():
                if node.tag == _W_NS + "t":
                    parts.append(node.text or "")
                elif node.tag == _W_NS + "tab":
                    parts.append("\t")
                elif node.tag == _W_NS + "br":
                    parts.append("\n")
            text = "".join(parts).strip()
            style = elem.find(f"{_W_NS}pPr/{_W_NS}pStyle")
            style_name = style.get(f"{_W_NS}val", "") if style is not None else ""
            elem.clear()
            if not text:
                continue
            if style_name.lower().startswith(("heading", "title")):
                heading = _clean(text)
                yield heading, "", ""
            else:
                yield heading, "", text


def _odt_sections(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    heading = ""
    with zipfile.ZipFile(path) as zf, zf.open("content.xml") as fh:
        for _, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag not in (_ODT_TEXT_NS + "h", _ODT_TEXT_NS + "p"):
                continue
            text = _squash("".join(elem.itertext()))
            is_heading = elem.tag == _ODT_TEXT_NS + "h"
            elem.clear()
            if not text:
                continue
            if is_heading:
                heading = _clean(text)
                yield heading, "", ""
            else:
                yield heading, "", text


def _pptx_sections(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    with zipfile.ZipFile(path) as zf:
        slides = sorted(
            (int(match.group(1)), name)
            for name in zf.namelist()
            if (match := _SLIDE_RE.match(name))
        )
        for number, name in slides:
            lines: list[str] = []
            with zf.open(name) as fh:
                for _, elem in ET.iterparse(fh, events=("end",)):
                    if elem.tag == _A_NS + "p":
                        text = "".join(node.text or "" for node in elem.iter(_A_NS + "t")).strip()
                        elem.clear()
                        if text:
                            lines.append(text)
            if lines:
                yield "", f"slide {number}", "\n\n".join(lines)


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    strings: list[str] = []
    with zf.open("xl/sharedStrings.xml") as fh:
        for _, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag == _S_NS + "si":
                strings.append(_squash("".join(elem.itertext())))
                elem.clear()
    return strings


def _xlsx_sheet_files(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    names = set(zf.namelist())
    targets: dict[str, str] = {}
    if "xl/_rels/workbook.xml.rels" in names:
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        for rel in rels.iter(f"{_PKG_NS}Relationship"):
            target = rel.get("Target", "")
            if target.startswith("/"):
                member = target.lstrip("/")
            else:
                member = posixpath.normpath(posixpath.join("xl", target))
            targets[rel.get("Id", "")] = member
    ordered: list[tuple[str, str]] = []
    if "xl/workbook.xml" in names:
        book = ET.fromstring(zf.read("xl/workbook.xml"))
        for sheet in book.iter(f"{_S_NS}sheet"):
            member = targets.get(sheet.get(f"{_R_NS}id", ""), "")
            if member in names:
                ordered.append((sheet.get("name") or member, member))
    if not ordered:
        numbered = sorted(
            (int(match.group(1)), name)
            for name in names
            if (match := _SHEET_RE.match(name))
        )
        ordered = [(f"sheet{number}", name) for number, name in numbered]
    return ordered


def _xlsx_value(cell: ET.Element, shared: list[str]) -> str:
    kind = cell.get("t", "")
    if kind == "inlineStr":
        return _squash("".join(cell.itertext()))
    value = cell.find(f"{_S_NS}v")
    if value is None or value.text is None:
        return ""
    if kind == "s":
        index = _to_int(value.text, -1)
        return shared[index] if 0 <= index < len(shared) else ""
    if kind == "b":
        return "TRUE" if value.text == "1" else "FALSE"
    return value.text.strip()


def _xlsx_column(reference: str) -> str:
    match = re.match(r"[A-Za-z]+", reference or "")
    return match.group(0).upper() if match else "?"


def _xlsx_sheet(
    zf: zipfile.ZipFile, member: str, name: str, shared: list[str], meta: dict[str, str]
) -> Iterator[Section]:
    lines: list[str] = []
    cells: list[str] = []
    first = last = 0
    rows = 0
    with zf.open(member) as fh:
        for _, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag == _S_NS + "c":
                value = _xlsx_value(elem, shared)
                if value:
                    cells.append(f"{_xlsx_column(elem.get('r', ''))}: {value}")
                elem.clear()
            elif elem.tag == _S_NS + "row":
                rows += 1
                number = _to_int(elem.get("r"), rows)
                if cells:
                    first = first or number
                    last = number
                    lines.append(" | ".join(cells))
                cells = []
                elem.clear()
                if rows >= XLSX_MAX_ROWS:
                    meta["note"] = f"sheet {name}: only the first {XLSX_MAX_ROWS} rows were indexed"
                    break
                if len(lines) >= ROW_GROUP:
                    yield "", f"sheet {name} rows {first}-{last}", "\n\n".join(lines)
                    lines, first = [], 0
    if lines:
        yield "", f"sheet {name} rows {first}-{last}", "\n\n".join(lines)


def _xlsx_sections(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    with zipfile.ZipFile(path) as zf:
        shared = _xlsx_shared_strings(zf) if "xl/sharedStrings.xml" in zf.namelist() else []
        for name, member in _xlsx_sheet_files(zf):
            yield from _xlsx_sheet(zf, member, name, shared, meta)


# --- extraction: PDF ----------------------------------------------------------


def _pdftotext_pages(binary: str, path: Path, meta: dict[str, str]) -> Iterator[Section]:
    proc = subprocess.Popen(
        [binary, "-enc", "UTF-8", "-q", str(path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    timed_out: list[bool] = []

    def expire() -> None:
        timed_out.append(True)
        proc.kill()

    killer = threading.Timer(PDF_TIMEOUT_S, expire)
    killer.start()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    page = 0
    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(65536)
            if not raw:
                break
            pending += decoder.decode(raw)
            *complete, pending = pending.split("\f")
            for body in complete:
                page += 1
                if body.strip():
                    yield "", f"page {page}", body.strip()
        pending += decoder.decode(b"", final=True)
        if pending.strip():
            page += 1
            yield "", f"page {page}", pending.strip()
    finally:
        killer.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
        if timed_out:
            meta["note"] = f"pdftotext stopped after {PDF_TIMEOUT_S} s; kept the pages read so far"


def _pdf_unescape(raw: bytes) -> bytes:
    out = bytearray()
    index, size = 0, len(raw)
    while index < size:
        char = raw[index]
        if char != 0x5C:  # backslash
            out.append(char)
            index += 1
            continue
        index += 1
        if index >= size:
            break
        char = raw[index]
        if 0x30 <= char <= 0x37:  # octal escape, up to three digits
            end = index
            while end < size and end < index + 3 and 0x30 <= raw[end] <= 0x37:
                end += 1
            out.append(int(raw[index:end], 8) & 0xFF)
            index = end
            continue
        if char in (0x0D, 0x0A):  # line continuation
            index += 2 if char == 0x0D and index + 1 < size and raw[index + 1] == 0x0A else 1
            continue
        out += _PDF_ESCAPES.get(char, bytes([char]))
        index += 1
    return bytes(out)


def _pdf_text(raw: bytes) -> str:
    data = _pdf_unescape(raw)
    if data.startswith(b"\xfe\xff"):
        return data[2:].decode("utf-16-be", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _pdf_array_text(body: bytes) -> str:
    pieces: list[str] = []
    for match in _PDF_ARRAY_RE.finditer(body):
        if match.group(1) is not None:
            pieces.append(_pdf_text(match.group(1)))
        elif float(match.group(2).decode()) < -200:  # a large negative kern is a word gap
            pieces.append(" ")
    return "".join(pieces)


def _pdf_stream_text(content: bytes) -> str:
    lines: list[str] = []
    current: list[str] = []
    for match in _PDF_OP_RE.finditer(content):
        if match.group(1) is not None:
            current.append(_pdf_text(match.group(1)))
        elif match.group(2) is not None:
            current.append(_pdf_array_text(match.group(2)))
        elif current:
            lines.append("".join(current))
            current = []
    if current:
        lines.append("".join(current))
    return "\n".join(line for line in lines if line.strip())


def _pdf_inflate(raw: bytes) -> bytes:
    try:
        return zlib.decompressobj().decompress(raw)
    except zlib.error:
        return raw  # not Flate-compressed: the stream is already plain text


def _pdf_fallback_sections(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    """Pure-Python text recovery from content streams (no pdftotext available)."""
    meta["quality"] = "fallback"
    meta["note"] = "read with the built-in PDF fallback; text may be incomplete"
    with path.open("rb") as fh:
        data = fh.read(PDF_FALLBACK_BYTES + 1)
    if len(data) > PDF_FALLBACK_BYTES:
        data = data[:PDF_FALLBACK_BYTES]
        meta["note"] += f"; only the first {PDF_FALLBACK_BYTES // MB} MB were scanned"
    part = 0
    for match in _PDF_STREAM_RE.finditer(data):
        text = _pdf_stream_text(_pdf_inflate(match.group(1)))
        if text:
            part += 1
            yield "", f"part {part}", text


def _pdf_sections(path: Path, meta: dict[str, str]) -> Iterator[Section]:
    binary = shutil.which("pdftotext")
    emitted = False
    if binary:
        try:
            for section in _pdftotext_pages(binary, path, meta):
                emitted = True
                yield section
        except OSError as exc:
            log.warning("pdftotext could not run on %s: %s", path, exc)
    if not emitted:
        yield from _pdf_fallback_sections(path, meta)


# --- extraction dispatch -------------------------------------------------------

_KINDS: dict[str, tuple[str, Extractor]] = {
    ".pdf": ("pdf", _pdf_sections),
    ".docx": ("docx", _docx_sections),
    ".odt": ("odt", _odt_sections),
    ".epub": ("epub", _epub_sections),
    ".pptx": ("pptx", _pptx_sections),
    ".xlsx": ("xlsx", _xlsx_sections),
    ".html": ("html", _html_file),
    ".htm": ("html", _html_file),
    ".xhtml": ("html", _html_file),
    ".md": ("markdown", _markdown_file),
    ".markdown": ("markdown", _markdown_file),
    ".txt": ("text", _text_file),
    ".rst": ("text", _text_file),
    ".log": ("text", _text_file),
    ".csv": ("csv", _csv_file),
    ".tsv": ("csv", _csv_file),
    ".json": ("json", _json_file),
    ".jsonl": ("json", _text_file),
    ".ndjson": ("json", _text_file),
    **{ext: ("code", _text_file) for ext in _CODE_EXTS},
}
SUPPORTED_EXTS = frozenset(_KINDS)


def extract(path: Path, meta: dict[str, str] | None = None) -> Iterator[Section]:
    """Stream ``(heading, location, text)`` sections from one file.

    Extractors that degrade (PDF fallback) record it in ``meta``.
    """
    file_path = Path(path)
    ext = file_path.suffix.lower()
    entry = _KINDS.get(ext)
    if entry is None:
        raise ValueError(f"unsupported document type: {ext or file_path.name}")
    return entry[1](file_path, {} if meta is None else meta)


def _inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _walk(root: Path, exts: frozenset[str]) -> Iterator[Path]:
    """Yield files under ``root`` with a matching extension, in sorted order.

    Hidden and build directories are skipped, symlinked directories are not
    followed, and symlinked files count only when they resolve inside ``root``.
    """
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except OSError:
            continue
        subdirs: list[Path] = []
        for entry in entries:
            if entry.name.startswith(".") or entry.name in SKIP_DIRS:
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(Path(entry.path))
                elif entry.is_file() and Path(entry.name).suffix.lower() in exts:
                    if entry.is_symlink() and not _inside(Path(entry.path).resolve(), root):
                        continue
                    yield Path(entry.path)
            except OSError:
                continue
        stack.extend(reversed(subdirs))


# --- indexing helpers ----------------------------------------------------------


class _TitleTap:
    """Remembers the first heading that streams past, for the document title."""

    def __init__(self) -> None:
        self.title = ""

    def wrap(self, sections: Iterable[Section]) -> Iterator[Section]:
        for section in sections:
            if not self.title and section[0]:
                self.title = section[0]
            yield section


def _insert_chunks(conn: sqlite3.Connection, doc_id: int, stream: Iterable[Section]) -> tuple[int, int, bool]:
    """Insert chunks in batches; returns (count, chars, truncated)."""
    batch: list[tuple[Any, ...]] = []
    seq = chars = 0
    truncated = False
    for heading, location, text in stream:
        if seq >= MAX_CHUNKS:
            truncated = True
            break
        seq += 1
        chars += len(text)
        batch.append((doc_id, seq, heading, location, text))
        if len(batch) >= BATCH_ROWS:
            conn.executemany(_INSERT_CHUNK, batch)
            batch = []
    if batch:
        conn.executemany(_INSERT_CHUNK, batch)
    return seq, chars, truncated


def _prune_outputs(conn: sqlite3.Connection) -> None:
    conn.execute(
        "DELETE FROM documents WHERE kind = 'output' AND id NOT IN "
        "(SELECT id FROM documents WHERE kind = 'output' ORDER BY id DESC LIMIT ?)",
        (MAX_OUTPUTS,),
    )


def _handle(row: sqlite3.Row) -> str:
    return f"D{row['id']}"


def _view(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "handle": _handle(row),
        "id": row["id"],
        "title": row["title"],
        "kind": row["kind"],
        "display": row["display"],
        "path": row["path"],
        "chunks": row["chunks"] or 0,
        "chars": row["chars"] or 0,
        "size": row["size"] or 0,
        "source": row["source"] or "",
        "added_at": row["added_at"],
        "indexed_at": row["indexed_at"],
    }


def _skipped(display: str, title: str, kind: str, reason: str, handle: str | None = None) -> dict[str, Any]:
    return {
        "handle": handle,
        "title": title,
        "kind": kind,
        "chunks": 0,
        "chars": 0,
        "display": display,
        "status": "skipped",
        "reason": reason,
    }


def _unchanged(row: sqlite3.Row, display: str) -> dict[str, Any]:
    return {
        "handle": _handle(row),
        "title": row["title"],
        "kind": row["kind"],
        "chunks": row["chunks"] or 0,
        "chars": row["chars"] or 0,
        "display": display,
        "status": "unchanged",
    }


def _row_by_path(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()


def _ambiguous(text: str, rows: list[sqlite3.Row]) -> str:
    listed = "; ".join(f'{_handle(row)} "{row["title"]}"' for row in rows[:5])
    return f'"{text}" matches several documents: {listed}. Pass one handle, for example {_handle(rows[0])}.'


def _close_rows(rows: list[sqlite3.Row], lowered: str) -> list[sqlite3.Row]:
    labels: dict[str, sqlite3.Row] = {}
    for row in rows:
        for label in (row["title"], Path(row["display"] or "").name):
            labels.setdefault(label.lower(), row)
    unique: dict[int, sqlite3.Row] = {}
    for label in difflib.get_close_matches(lowered, list(labels), n=5, cutoff=0.6):
        unique.setdefault(labels[label]["id"], labels[label])
    return list(unique.values())


def _lookup(conn: sqlite3.Connection, ref: str) -> sqlite3.Row | None:
    """Find one document by handle, path, display or title; None when nothing matches."""
    text = str(ref or "").strip()
    if not text:
        raise ValueError("document is required: pass a handle such as D3, a path or a title")
    number = _HANDLE_RE.fullmatch(text)
    if number:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (int(number.group(1)),)).fetchone()
        if row is not None:
            return row
    rows = conn.execute("SELECT * FROM documents").fetchall()
    lowered = text.lower()
    exact: list[sqlite3.Row] = []
    if os.path.isabs(text):
        absolute = str(Path(text).resolve())
        exact = [row for row in rows if row["path"] == absolute]
    if not exact:
        exact = [row for row in rows if lowered in ((row["display"] or "").lower(), (row["title"] or "").lower())]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(_ambiguous(text, exact))
    partial = [
        row for row in rows if lowered in (row["title"] or "").lower() or lowered in (row["display"] or "").lower()
    ]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError(_ambiguous(text, partial))
    close = _close_rows(rows, lowered)
    if len(close) == 1:
        return close[0]
    if len(close) > 1:
        raise ValueError(_ambiguous(text, close))
    return None


def _require(conn: sqlite3.Connection, ref: str) -> sqlite3.Row:
    row = _lookup(conn, ref)
    if row is not None:
        return row
    rows = conn.execute("SELECT * FROM documents").fetchall()
    if not rows:
        raise ValueError("the library is empty: add files with doc_add first")
    text = str(ref or "").strip()
    ranked = sorted(
        rows,
        key=lambda item: difflib.SequenceMatcher(None, text.lower(), (item["title"] or "").lower()).ratio(),
        reverse=True,
    )[:5]
    listed = "; ".join(f'{_handle(item)} "{item["title"]}"' for item in ranked)
    raise ValueError(f'no document matches "{text}". Use a handle from doc_list, for example: {listed}.')


def _query_tokens(query: str) -> list[str]:
    words = re.findall(r"[^\W_]+", (query or "").lower())
    kept = [word for word in words if word not in STOPWORDS] or words
    return list(dict.fromkeys(kept))[:12]


def _fts_expr(tokens: list[str], joiner: str) -> str:
    # Tokens come from [^\W_]+, so they never contain quotes or FTS operators.
    return joiner.join(f'"{token}"*' for token in tokens)


def _fts_rows(
    conn: sqlite3.Connection, expr: str, doc_id: int | None, exclude_outputs: bool, pool: int
) -> list[sqlite3.Row]:
    sql = """
        SELECT c.id AS chunk_id, c.doc_id AS doc_id, c.seq AS seq, c.heading AS heading,
               c.location AS location, d.title AS title, d.display AS display,
               snippet(chunks_fts, 0, '[', ']', ' … ', 24) AS snippet,
               bm25(chunks_fts, 1.0, 2.0) AS bm25_rank
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN documents d ON d.id = c.doc_id
        WHERE chunks_fts MATCH ?
    """
    params: list[Any] = [expr]
    if doc_id is not None:
        sql += " AND c.doc_id = ?"
        params.append(doc_id)
    if exclude_outputs:
        sql += " AND d.kind != 'output'"
    sql += " ORDER BY bm25_rank LIMIT ?"
    params.append(pool)
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("library search failed for %r: %s", expr, exc)
        return []


def _hit(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "handle": f"D{row['doc_id']}",
        "title": row["title"],
        "display": row["display"],
        "seq": row["seq"],
        "heading": row["heading"],
        "location": row["location"],
        "snippet": _squash(row["snippet"] or ""),
        "score": round(-row["bm25_rank"], 4),
    }


def _note_text(meta: dict[str, str], truncated: bool) -> str:
    notes = [meta.get("note", "")]
    if truncated:
        notes.append(f"only the first {MAX_CHUNKS} chunks were indexed")
    return "; ".join(note for note in notes if note)


# --- the library -----------------------------------------------------------------


class Library:
    """SQLite-backed document index. One database file per library."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self._init_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._sync_lock = threading.Lock()
        self._ready = False
        self._last_sync: dict[str, float] = {}

    # connections --------------------------------------------------------------

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=15, isolation_level=None)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
            finally:
                conn.close()
            self._ready = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._ensure_schema()
        conn = sqlite3.connect(self.db_path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    # adding documents ----------------------------------------------------------

    def add_file(self, path: Path, display: str) -> dict[str, Any]:
        """Index one file. Returns a status of indexed, updated, unchanged or skipped."""
        file_path = Path(path).resolve()
        ext = file_path.suffix.lower()
        entry = _KINDS.get(ext)
        if entry is None:
            return _skipped(display, file_path.stem, "", f"unsupported type {ext or 'without extension'}")
        kind, extractor = entry
        try:
            st = file_path.stat()
        except OSError:
            return _skipped(display, file_path.stem, kind, "file cannot be read")
        if not file_path.is_file():
            return _skipped(display, file_path.stem, kind, "not a regular file")
        limit = MAX_BYTES_PDF if ext == ".pdf" else MAX_BYTES_OTHER
        if st.st_size > limit:
            return _skipped(display, file_path.stem, kind, f"too large ({st.st_size // MB} MB; limit {limit // MB} MB)")
        if st.st_size == 0:
            return _skipped(display, file_path.stem, kind, "file is empty")
        with self._connect() as conn:
            row = _row_by_path(conn, str(file_path))
            if row is not None and row["size"] == st.st_size and row["mtime"] == st.st_mtime:
                return _unchanged(row, display)
            try:
                digest = _sha256(file_path)
            except OSError:
                return _skipped(display, file_path.stem, kind, "file cannot be read", _handle(row) if row else None)
            if row is not None and row["sha256"] == digest:
                conn.execute("UPDATE documents SET size = ?, mtime = ? WHERE id = ?", (st.st_size, st.st_mtime, row["id"]))
                return _unchanged(row, display)
            try:
                with self._write_lock:
                    return self._index(conn, file_path, display, kind, extractor, st, digest, row)
            except sqlite3.Error:
                raise
            except Exception as exc:  # one unreadable file must not stop a batch
                log.warning("could not index %s: %s", file_path, exc)
                reason = f"could not extract text: {exc}"
                return _skipped(display, file_path.stem, kind, reason, _handle(row) if row else None)

    def _index(
        self,
        conn: sqlite3.Connection,
        path: Path,
        display: str,
        kind: str,
        extractor: Extractor,
        st: os.stat_result,
        digest: str,
        row: sqlite3.Row | None,
    ) -> dict[str, Any]:
        meta: dict[str, str] = {}
        tap = _TitleTap()
        now = _now()
        conn.execute("BEGIN IMMEDIATE")
        committed = False
        try:
            if row is None:
                doc_id = conn.execute(
                    "INSERT INTO documents (path, display, title, kind, added_at) VALUES (?, ?, ?, ?, ?)",
                    (str(path), display, path.stem, kind, now),
                ).lastrowid
            else:
                doc_id = row["id"]
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            with closing(chunk_sections(tap.wrap(extractor(path, meta)))) as stream:
                count, chars, truncated = _insert_chunks(conn, doc_id, stream)
            if count == 0:
                return _skipped(display, path.stem, kind, "no extractable text", _handle(row) if row else None)
            title = _clean(tap.title) or path.stem
            conn.execute(
                "UPDATE documents SET path = ?, display = ?, title = ?, kind = ?, size = ?, mtime = ?, "
                "sha256 = ?, chunks = ?, chars = ?, indexed_at = ? WHERE id = ?",
                (str(path), display, title, kind, st.st_size, st.st_mtime, digest, count, chars, now, doc_id),
            )
            conn.execute("COMMIT")
            committed = True
        finally:
            if not committed and conn.in_transaction:
                conn.execute("ROLLBACK")
        result = {
            "handle": f"D{doc_id}",
            "title": title,
            "kind": kind,
            "chunks": count,
            "chars": chars,
            "display": display,
            "status": "indexed" if row is None else "updated",
        }
        note = _note_text(meta, truncated)
        if note:
            result["note"] = note
        if meta.get("quality"):
            result["quality"] = meta["quality"]
        return result

    def add_folder(
        self,
        root: Path,
        display_fn: Callable[[Path], str],
        *,
        max_files: int = 500,
        time_budget: float = 30.0,
    ) -> list[dict[str, Any]]:
        """Index every supported file under ``root``; stops at ``max_files`` or the time budget."""
        root_path = Path(root).resolve()
        if root_path.is_file():
            return [self.add_file(root_path, display_fn(root_path))]
        deadline = time.monotonic() + time_budget
        rows: list[dict[str, Any]] = []
        for path in _walk(root_path, SUPPORTED_EXTS):
            display = display_fn(path)
            if len(rows) >= max_files:
                rows.append(_skipped(display, "", "", f"file limit of {max_files} reached; run doc_add on a narrower folder"))
                break
            if time.monotonic() >= deadline:
                rows.append(_skipped(display, "", "", "time budget reached; run doc_add again to continue"))
                break
            rows.append(self.add_file(path, display))
        return rows

    def sync(
        self,
        root: Path,
        display_fn: Callable[[Path], str],
        *,
        time_budget: float = 2.0,
        max_files: int = 300,
        min_interval: float = 20.0,
    ) -> dict[str, int]:
        """Index new or changed document files under ``root`` and drop entries whose file is gone."""
        counts = {"indexed": 0, "updated": 0, "removed": 0, "skipped": 0}
        root_path = Path(root).resolve()
        key = str(root_path)
        now = time.monotonic()
        with self._sync_lock:
            last = self._last_sync.get(key)
            if last is not None and now - last < min_interval:
                return counts
            self._last_sync[key] = now
        deadline = now + time_budget
        worked = 0
        for path in _walk(root_path, SYNC_EXTS):
            if worked >= max_files or time.monotonic() >= deadline:
                break
            status = self._sync_one(path, display_fn(path))
            if status is not None:
                worked += 1
                counts[status if status in ("indexed", "updated") else "skipped"] += 1
        counts["removed"] = self._prune_missing(root_path)
        return counts

    def _sync_one(self, path: Path, display: str) -> str | None:
        """Index a new or changed file; None when there was nothing to do."""
        try:
            st = path.stat()
        except OSError:
            return None
        if st.st_size > SYNC_MAX_BYTES:
            return None
        with self._connect() as conn:
            row = _row_by_path(conn, str(path.resolve()))
        if row is not None and row["size"] == st.st_size and row["mtime"] == st.st_mtime:
            return None
        return self.add_file(path, display)["status"]

    def _prune_missing(self, root: Path) -> int:
        prefix = str(root).rstrip(os.sep) + os.sep
        with self._connect() as conn:
            rows = conn.execute("SELECT id, path FROM documents WHERE path IS NOT NULL").fetchall()
            gone = [row["id"] for row in rows if row["path"].startswith(prefix) and not os.path.exists(row["path"])]
            if not gone:
                return 0
            with self._write_lock:
                conn.executemany("DELETE FROM documents WHERE id = ?", [(doc_id,) for doc_id in gone])
        return len(gone)

    def stash_text(self, text: str, *, title: str, source: str = "") -> dict[str, Any]:
        """Store arbitrary long text (e.g. a huge tool output) as a searchable output document."""
        if not text or not text.strip():
            raise ValueError("nothing to stash: the text is empty")
        label = _clean(title) or "tool output"
        now = _now()
        size = len(text.encode("utf-8", errors="replace"))
        lines = (line.rstrip("\r\n") for line in io.StringIO(text))
        with self._connect() as conn, self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            committed = False
            try:
                doc_id = conn.execute(
                    "INSERT INTO documents (path, display, title, kind, size, chunks, chars, added_at, indexed_at, source) "
                    "VALUES (NULL, ?, ?, 'output', ?, 0, 0, ?, ?, ?)",
                    (label, label, size, now, now, _clean(source, 200)),
                ).lastrowid
                with closing(chunk_sections(_line_sections(lines, markdown=False))) as stream:
                    count, chars, truncated = _insert_chunks(conn, doc_id, stream)
                conn.execute("UPDATE documents SET chunks = ?, chars = ? WHERE id = ?", (count, chars, doc_id))
                _prune_outputs(conn)
                conn.execute("COMMIT")
                committed = True
            finally:
                if not committed and conn.in_transaction:
                    conn.execute("ROLLBACK")
        result: dict[str, Any] = {"handle": f"D{doc_id}", "chunks": count, "chars": chars}
        if truncated:
            result["note"] = f"only the first {MAX_CHUNKS} chunks were stored"
        return result

    # managing and reading ------------------------------------------------------

    def remove(self, ref: str) -> bool:
        with self._connect() as conn, self._write_lock:
            row = _lookup(conn, ref)
            if row is None:
                return False
            conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            return True

    def resolve(self, ref: str) -> dict[str, Any]:
        with self._connect() as conn:
            return _view(_require(conn, ref))

    def list_documents(self, limit: int = 50, include_outputs: bool = False) -> list[dict[str, Any]]:
        where = "" if include_outputs else "WHERE kind != 'output'"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents {where} ORDER BY id DESC LIMIT ?", (max(1, int(limit)),)
            ).fetchall()
        return [_view(row) for row in rows]

    def outline(self, ref: str, limit: int = 200) -> list[dict[str, Any]]:
        """One row per distinct heading (the first chunk always counts)."""
        with self._connect() as conn:
            row = _require(conn, ref)
            rows: list[dict[str, Any]] = []
            current: str | None = None
            for chunk in conn.execute(
                "SELECT seq, heading, location FROM chunks WHERE doc_id = ? ORDER BY seq", (row["id"],)
            ):
                if chunk["heading"] == current:
                    continue
                current = chunk["heading"]
                rows.append({"seq": chunk["seq"], "heading": chunk["heading"], "location": chunk["location"]})
                if len(rows) >= limit:
                    break
        return rows

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT (SELECT COUNT(*) FROM documents WHERE kind != 'output') AS documents, "
                "(SELECT COUNT(*) FROM chunks) AS chunks, "
                "(SELECT COUNT(*) FROM documents WHERE kind = 'output') AS outputs, "
                "(SELECT COALESCE(SUM(size), 0) FROM documents) AS bytes"
            ).fetchone()
        return {key: int(row[key]) for key in ("documents", "chunks", "outputs", "bytes")}

    def search(
        self,
        query: str,
        *,
        limit: int = 6,
        document: str | None = None,
        include_outputs: bool = False,
    ) -> list[dict[str, Any]]:
        """Ranked passages for ``query``. Tokens are AND-ed first, then OR-ed to fill up the list.

        ``score`` is higher-is-better (the negated bm25 rank).
        """
        tokens = _query_tokens(query)
        if not tokens:
            return []
        limit = max(1, min(int(limit or 6), 50))
        pool = min(400, max(limit * 40, 120))
        with self._connect() as conn:
            doc_id = None
            exclude_outputs = not include_outputs
            if document:
                doc_id = _require(conn, document)["id"]
                exclude_outputs = False
            cap = None if doc_id is not None else PER_DOC_HITS
            hits: list[dict[str, Any]] = []
            seen: set[int] = set()
            per_doc: dict[int, int] = {}

            def take(rows: list[sqlite3.Row]) -> None:
                for row in rows:
                    if len(hits) >= limit:
                        return
                    if row["chunk_id"] in seen:
                        continue
                    used = per_doc.get(row["doc_id"], 0)
                    if cap is not None and used >= cap:
                        continue
                    seen.add(row["chunk_id"])
                    per_doc[row["doc_id"]] = used + 1
                    hits.append(_hit(row))

            take(_fts_rows(conn, _fts_expr(tokens, " "), doc_id, exclude_outputs, pool))
            if len(hits) < limit and len(tokens) > 1:
                take(_fts_rows(conn, _fts_expr(tokens, " OR "), doc_id, exclude_outputs, pool))
        return hits

    def read(
        self,
        ref: str,
        *,
        chunk: int | None = None,
        count: int = 3,
        section: str | None = None,
    ) -> dict[str, Any]:
        """Chunks ``chunk``..``chunk+count-1`` (or from the first heading containing ``section``)."""
        count = max(1, min(int(count or 3), 8))
        with self._connect() as conn:
            row = _require(conn, ref)
            doc = _view(row)
            total = int(row["chunks"] or 0)
            if total == 0:
                return {"doc": doc, "chunks": [], "total": 0, "next": None}
            start = 1
            if chunk is not None:
                start = int(chunk)
                if not 1 <= start <= total:
                    raise ValueError(f"{doc['handle']} has {total} chunks; chunk must be between 1 and {total}")
            elif section and section.strip():
                needle = section.strip().lower()
                for heading_row in conn.execute(
                    "SELECT seq, heading FROM chunks WHERE doc_id = ? ORDER BY seq", (row["id"],)
                ):
                    if needle in heading_row["heading"].lower():
                        start = heading_row["seq"]
                        break
                else:
                    raise ValueError(
                        f'no heading in {doc["handle"]} contains "{section.strip()}"; '
                        f'use doc_list with document="{doc["handle"]}" to see the outline'
                    )
            rows = conn.execute(
                "SELECT seq, heading, location, text FROM chunks WHERE doc_id = ? AND seq BETWEEN ? AND ? ORDER BY seq",
                (row["id"], start, start + count - 1),
            ).fetchall()
        chunks = [dict(item) for item in rows]
        last = chunks[-1]["seq"] if chunks else start
        return {
            "doc": doc,
            "chunks": chunks,
            "total": total,
            "next": last + 1 if last < total else None,
        }


_LIBRARIES: dict[str, Library] = {}
_LIBRARIES_LOCK = threading.Lock()


def get_library(db_path: Path) -> Library:
    key = str(Path(db_path).expanduser().resolve())
    with _LIBRARIES_LOCK:
        library = _LIBRARIES.get(key)
        if library is None:
            library = _LIBRARIES[key] = Library(Path(key))
        return library


def library_for(ctx: Any) -> Library:
    return get_library(data_dir_for(ctx) / "library.db")


# --- paging long tool outputs ---------------------------------------------------------

NO_PAGING = frozenset({"doc_search", "doc_read", "doc_list", "doc_add"})


def page_limit(context_tokens: int) -> int:
    """Largest tool output (chars) kept inline: about 15% of the context window."""
    return max(1500, min(8000, int(context_tokens * 0.15 * 3.6)))


def page_output(library: Library, name: str, result: Any, limit_chars: int) -> Any:
    """Stash an over-long tool output in the library; keep head + tail + a navigation hint.

    Nothing is lost (the full text stays searchable and readable by handle) and
    a small context window is not flooded. Call after guard.postprocess, so the
    stored text is already redacted.
    """
    content = getattr(result, "content", None)
    if not isinstance(content, str) or len(content) <= limit_chars or name in NO_PAGING:
        return result
    try:
        stored = library.stash_text(content, title=f"{name} output", source=name)
    except Exception:  # paging is an optimisation; the inline output still works
        log.warning("could not stash long %s output", name, exc_info=True)
        return result
    head = content[: int(limit_chars * 0.6)]
    tail = content[-int(limit_chars * 0.25) :]
    if "\n" in head[-300:]:
        head = head[: head.rfind("\n")]
    if "\n" in tail[:300]:
        tail = tail[tail.find("\n") + 1 :]
    handle = stored["handle"]
    hint = (
        f"\n… [{len(content) - len(head) - len(tail):,} more characters stored as {handle} "
        f"({stored['chunks']} chunks). Continue reading: doc_read(document=\"{handle}\", "
        f"chunk={len(head) // MAX_CHARS + 1}) · search inside: doc_search(query=\"...\", document=\"{handle}\")] …\n"
    )
    meta = dict(getattr(result, "meta", None) or {})
    meta["paged"] = {"handle": handle, "chunks": stored["chunks"], "chars": len(content)}
    return replace(result, content=head + hint + tail, meta=meta)


# --- tools -------------------------------------------------------------------------


def _text_arg(args: dict[str, Any], name: str, *, required: bool = False, limit: int = 400) -> str:
    value = args.get(name)
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()[:limit]
    if required and not text:
        raise ValueError(f"{name} is required")
    return text


def _int_arg(args: dict[str, Any], name: str, low: int, high: int, default: int | None) -> int | None:
    value = args.get(name)
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return max(low, min(high, number))


def _validate_search(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": _text_arg(args, "query", required=True),
        "limit": _int_arg(args, "limit", 1, 10, 5),
        "document": _text_arg(args, "document", limit=200) or None,
    }


def _validate_read(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "document": _text_arg(args, "document", required=True, limit=200),
        "chunk": _int_arg(args, "chunk", 1, 10**9, None),
        "count": _int_arg(args, "count", 1, 8, 3),
        "section": _text_arg(args, "section", limit=200) or None,
    }


def _validate_add(args: dict[str, Any]) -> dict[str, Any]:
    return {"path": _text_arg(args, "path", required=True, limit=1024)}


def _validate_list(args: dict[str, Any]) -> dict[str, Any]:
    return {"document": _text_arg(args, "document", limit=200) or None}


def _exec_search(args: dict[str, Any], ctx: Any) -> Any:
    library = library_for(ctx)
    library.sync(ctx.workspace.root, ctx.workspace.public)
    query = args["query"]
    try:
        hits = library.search(query, limit=args["limit"], document=args.get("document"))
    except ValueError as exc:
        return tool_result(False, str(exc))
    if not hits:
        return tool_result(
            True,
            f'No matches for "{query}" in the indexed documents. Try doc_list to see what is available, '
            "or different words.",
            meta={"hits": 0},
        )
    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        where = f" · {hit['location']}" if hit["location"] else ""
        lines.append(f"[{index}] {hit['handle']} · {hit['title']}{where} — {hit['snippet']}")
    top = hits[0]
    lines.append(f'Read more: doc_read(document="{top["handle"]}", chunk={top["seq"]})')
    return tool_result(True, "\n".join(lines), meta={"hits": len(hits)})


def _exec_read(args: dict[str, Any], ctx: Any) -> Any:
    try:
        data = library_for(ctx).read(
            args["document"],
            chunk=args.get("chunk"),
            count=args["count"],
            section=args.get("section"),
        )
    except ValueError as exc:
        return tool_result(False, str(exc))
    doc = data["doc"]
    chunks = data["chunks"]
    if not chunks:
        return tool_result(True, f"{doc['handle']} · {doc['title']} — the document has no content.")
    first, last = chunks[0]["seq"], chunks[-1]["seq"]
    span = f"chunk {first}" if first == last else f"chunks {first}-{last}"
    blocks = [f"{doc['handle']} · {doc['title']} — {span} of {data['total']}"]
    for item in chunks:
        blocks.append(f"[{_label(item['seq'], item['location'], item['heading'])}]\n{item['text']}")
    if data["next"]:
        blocks.append(f'Next: doc_read(document="{doc["handle"]}", chunk={data["next"]})')
    else:
        blocks.append("(end of document)")
    return tool_result(True, "\n\n".join(blocks))


def _exec_add(args: dict[str, Any], ctx: Any) -> Any:
    target = ctx.workspace.resolve(args["path"], must_exist=True)  # SandboxError propagates
    library = library_for(ctx)
    try:
        if target.is_dir():
            rows = library.add_folder(target, ctx.workspace.public)
        else:
            rows = [library.add_file(target, ctx.workspace.public(target))]
    except ValueError as exc:
        return tool_result(False, str(exc))
    if not rows:
        return tool_result(True, f"No supported documents found in {ctx.workspace.public(target)}.")
    lines: list[str] = []
    totals = {"indexed": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    for row in rows:
        status = row["status"]
        totals[status] += 1
        if status == "skipped":
            lines.append(f"{row['display']} · skipped: {row['reason']}")
            continue
        line = f"{row['handle']} · {row['title']} · {row['kind']} · {row['chunks']} chunks · {status}"
        if row.get("note"):
            line += f" ({row['note']})"
        lines.append(line)
    lines.append(
        f"Total: {totals['indexed']} indexed, {totals['updated']} updated, "
        f"{totals['unchanged']} unchanged, {totals['skipped']} skipped."
    )
    return tool_result(True, "\n".join(lines))


def _exec_list(args: dict[str, Any], ctx: Any) -> Any:
    library = library_for(ctx)
    document = args.get("document")
    try:
        if document:
            doc = library.resolve(document)
            outline = library.outline(document, limit=200)
            lines = [f"{doc['handle']} · {doc['title']} · {doc['kind']} · {doc['chunks']} chunks · {doc['display']}"]
            if not outline:
                lines.append("(no headings; read the chunks in order with doc_read)")
            for item in outline:
                lines.append(_label(item["seq"], item["location"], item["heading"]))
            if len(outline) >= 200:
                lines.append("(outline truncated; use doc_read with section=... to jump)")
            return tool_result(True, "\n".join(lines))
        docs = library.list_documents(limit=50)
        stats = library.stats()
    except ValueError as exc:
        return tool_result(False, str(exc))
    lines = [
        f"{doc['handle']} · {doc['title']} · {doc['kind']} · {doc['chunks']} chunks · {doc['display']}"
        for doc in docs
    ] or ["No documents indexed yet. Use doc_add with a file or folder in the workspace."]
    lines.append(
        f"{stats['documents']} documents, {stats['chunks']} chunks, {stats['outputs']} stashed outputs"
    )
    return tool_result(True, "\n".join(lines))


def _search_summary(args: dict[str, Any]) -> str:
    return f'Search documents for "{args.get("query", "")}"'


def _read_summary(args: dict[str, Any]) -> str:
    return f"Read document {args.get('document', '')}".strip()


def _add_summary(args: dict[str, Any]) -> str:
    return f"Index documents at {args.get('path', '')}".strip()


def _list_summary(args: dict[str, Any]) -> str:
    return f"List documents {args.get('document', '')}".strip()


DOC_SEARCH = ExtensionTool(
    name="doc_search",
    description=(
        "Search the user's documents (PDF, Word, ebooks, slides, notes and other files in the workspace are "
        "indexed automatically) and stashed long tool outputs. Returns the best passages with handles such as "
        "D3; read around a passage with doc_read."
    ),
    parameters=schema(
        properties={
            "query": {"type": "string", "description": "Words to look for."},
            "limit": {"type": "integer", "description": "Number of passages (1-10, default 5).", "minimum": 1, "maximum": 10},
            "document": {"type": "string", "description": "Optional handle (D3) or title to search inside one document."},
        },
        required=["query"],
    ),
    risk="read",
    phase="reading",
    label="Searching documents",
    summary=_search_summary,
    validate=_validate_search,
    execute=_exec_search,
    category="core",  # paged tool output points here, so it must always be offered
)

DOC_READ = ExtensionTool(
    name="doc_read",
    description=(
        "Read chunks of an indexed document by handle (e.g. D3), a few at a time. Use chunk to continue "
        "from a chunk number, or section to jump to a heading."
    ),
    parameters=schema(
        properties={
            "document": {"type": "string", "description": "Handle such as D3, or a title."},
            "chunk": {"type": "integer", "description": "First chunk number to read (1 or more).", "minimum": 1},
            "count": {"type": "integer", "description": "How many chunks (1-8, default 3).", "minimum": 1, "maximum": 8},
            "section": {"type": "string", "description": "Case-insensitive heading text to start from."},
        },
        required=["document"],
    ),
    risk="read",
    phase="reading",
    label="Reading a document",
    summary=_read_summary,
    validate=_validate_read,
    execute=_exec_read,
    category="core",  # paged tool output points here, so it must always be offered
)

DOC_ADD = ExtensionTool(
    name="doc_add",
    description=(
        "Index a file or folder inside the workspace so doc_search and doc_read can use it. Supports PDF, "
        "DOCX, ODT, EPUB, PPTX, XLSX, HTML, Markdown, text, code, CSV and JSON."
    ),
    parameters=schema(
        properties={"path": {"type": "string", "description": "File or folder path inside the workspace."}},
        required=["path"],
    ),
    risk="read",
    phase="reading",
    label="Indexing documents",
    summary=_add_summary,
    validate=_validate_add,
    execute=_exec_add,
    category="research",
)

DOC_LIST = ExtensionTool(
    name="doc_list",
    description="List indexed documents with their handles, or the heading outline of one document.",
    parameters=schema(
        properties={"document": {"type": "string", "description": "Optional handle or title to show its outline."}},
    ),
    risk="read",
    phase="reading",
    label="Listing documents",
    summary=_list_summary,
    validate=_validate_list,
    execute=_exec_list,
    category="research",
)

TOOLS = (DOC_SEARCH, DOC_READ, DOC_ADD, DOC_LIST)
