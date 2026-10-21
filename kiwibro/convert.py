"""Document conversion with the converters already installed on this computer.

Pandoc handles the text formats (Markdown, HTML, DOCX, ODT, EPUB, RST, LaTeX,
text), pdftotext turns PDF into text, LibreOffice handles Office files, and
weasyprint lets pandoc write PDF. Nothing is installed and no shell is used:
every converter runs from a fixed argv list with a timeout, and writes into a
private scratch folder that is moved into place only when it succeeded. Existing
files are never overwritten; a conversion gets a ``-2``, ``-3`` ... name instead.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .extensions import ExtensionTool, schema, tool_result

PANDOC_TIMEOUT_S = 90
PDFTOTEXT_TIMEOUT_S = 90
LIBREOFFICE_TIMEOUT_S = 180
MAX_TAIL_CHARS = 300
MAX_NAME_TRIES = 1000

ROUTE_PDFTOTEXT = "pdftotext"
ROUTE_PANDOC = "pandoc"
ROUTE_PANDOC_PDF = "pandoc+weasyprint"
ROUTE_LIBREOFFICE = "LibreOffice"

OFFICE_INPUTS = frozenset({"docx", "doc", "odt", "rtf", "xlsx", "xls", "ods", "pptx", "ppt", "odp"})
OFFICE_OUTPUTS = frozenset({"pdf", "docx", "odt", "xlsx", "ods", "pptx", "odp", "csv", "html", "txt"})

# extension -> pandoc reader/writer name
PANDOC_FORMATS: dict[str, str] = {
    "md": "markdown",
    "markdown": "markdown",
    "html": "html",
    "htm": "html",
    "docx": "docx",
    "odt": "odt",
    "epub": "epub",
    "rst": "rst",
    "tex": "latex",
    "txt": "plain",
    "org": "org",
    "rtf": "rtf",
}
PANDOC_STANDALONE = frozenset({"html", "latex", "rtf", "epub"})

# Spellings a model may use, mapped to the canonical extension.
ALIASES = {"markdown": "md", "htm": "html", "text": "txt", "word": "docx", "latex": "tex"}
FORMAT_NAME = re.compile(r"^[a-z0-9]{1,10}$")


def _canonical(ext: str) -> str:
    return ALIASES.get(ext, ext)


PANDOC_TARGETS = tuple(dict.fromkeys(_canonical(ext) for ext in PANDOC_FORMATS))
PANDOC_SOURCES = frozenset(_canonical(ext) for ext in PANDOC_FORMATS)


def normalize_target(raw: str) -> str:
    """Lowercase, drop a leading dot and map common spellings (Markdown -> md)."""
    value = raw.strip().lower().lstrip(".")
    return _canonical(value)


def _find(name: str) -> str | None:
    return shutil.which(name)


def _soffice() -> str | None:
    for name in ("soffice", "libreoffice", "/snap/bin/libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _env() -> dict[str, str]:
    return {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}


def _dot(ext: str) -> str:
    return f".{ext}" if ext else "files with no extension"


def _list(exts: list[str]) -> str:
    return ", ".join(f".{ext}" for ext in exts)


def available_routes() -> dict[str, list[str]]:
    """Source extension -> target extensions possible with the converters installed here."""
    pandoc = _find("pandoc") is not None
    weasyprint = _find("weasyprint") is not None
    office = _soffice() is not None
    routes: dict[str, list[str]] = {}
    if _find("pdftotext") is not None:
        routes["pdf"] = ["txt"]
    sources = sorted(PANDOC_SOURCES | OFFICE_INPUTS)
    for source in sources:
        targets: list[str] = []
        if pandoc and source in PANDOC_SOURCES:
            targets += [t for t in PANDOC_TARGETS if t != source]
            if weasyprint:
                targets.append("pdf")
        if office and source in OFFICE_INPUTS:
            targets += [t for t in sorted(OFFICE_OUTPUTS) if t != source and t not in targets]
        targets = list(dict.fromkeys(targets))
        if targets:
            routes[source] = targets
    return routes


def _unsupported(source: str, target: str) -> str:
    routes = available_routes()
    if not routes:
        return "No document converters are installed on this computer, so nothing can be converted."
    if source in routes:
        return (
            f"Cannot convert {_dot(source)} to .{target} with the converters on this computer. "
            f"Converters here can turn {_dot(source)} into: {_list(routes[source])}."
        )
    return (
        f"No converter on this computer reads {_dot(source)}. "
        f"Sources that can be converted here: {_list(sorted(routes))}."
    )


def _route(source: str, target: str) -> str:
    if source == target:
        raise ValueError(
            f"The file is already .{source}. Choose a different target such as pdf, docx, md, html or txt."
        )
    if source == "pdf":
        if target != "txt":
            raise ValueError(
                f"PDF files can only be converted to txt here. Convert the PDF to txt first, "
                f"then convert that txt file to .{target}."
            )
        if _find("pdftotext") is None:
            raise ValueError("pdftotext is not installed on this computer, so PDF text extraction is unavailable.")
        return ROUTE_PDFTOTEXT
    if target == "pdf":
        if source in OFFICE_INPUTS:
            if _soffice() is None:
                raise ValueError(f"Making a PDF from {_dot(source)} needs LibreOffice, which is not installed here.")
            return ROUTE_LIBREOFFICE
        if source in PANDOC_SOURCES:
            missing = [name for name in ("pandoc", "weasyprint") if _find(name) is None]
            if missing:
                raise ValueError(
                    f"Making a PDF from {_dot(source)} needs {' and '.join(missing)}, "
                    "which is not installed on this computer."
                )
            return ROUTE_PANDOC_PDF
        raise ValueError(_unsupported(source, target))
    pandoc_pair = source in PANDOC_SOURCES and target in PANDOC_TARGETS
    office_pair = source in OFFICE_INPUTS and target in OFFICE_OUTPUTS
    if pandoc_pair and _find("pandoc") is not None:
        return ROUTE_PANDOC
    if office_pair and _soffice() is not None:
        return ROUTE_LIBREOFFICE
    if pandoc_pair:
        raise ValueError(f"pandoc is not installed on this computer, so {_dot(source)} cannot become .{target} here.")
    if office_pair:
        raise ValueError(f"LibreOffice is not installed on this computer, so {_dot(source)} cannot become .{target} here.")
    raise ValueError(_unsupported(source, target))


def _plan(route: str, src: Path, target: str, work: Path, produced_name: str) -> tuple[str, list[str], Path, int]:
    """Return (converter label, argv, file the converter writes, timeout)."""
    if route == ROUTE_PDFTOTEXT:
        produced = work / produced_name
        argv = ["pdftotext", "-enc", "UTF-8", "-layout", str(src), str(produced)]
        return "pdftotext", argv, produced, PDFTOTEXT_TIMEOUT_S
    if route == ROUTE_LIBREOFFICE:
        produced = work / f"{src.stem}.{target}"
        soffice = _soffice() or "soffice"
        argv = [soffice, "--headless", "--norestore", "--convert-to", target, "--outdir", str(work), str(src)]
        return "LibreOffice", argv, produced, LIBREOFFICE_TIMEOUT_S
    produced = work / produced_name
    reader = PANDOC_FORMATS[_canonical(src.suffix.lower().lstrip("."))]
    if route == ROUTE_PANDOC_PDF:
        argv = ["pandoc", "-f", reader, "--pdf-engine=weasyprint", "-o", str(produced), str(src)]
        return "pandoc+weasyprint", argv, produced, PANDOC_TIMEOUT_S
    writer = PANDOC_FORMATS[target]
    argv = ["pandoc", "-f", reader, "-t", writer]
    if target in PANDOC_STANDALONE:
        argv.append("--standalone")
    argv += ["-o", str(produced), str(src)]
    return "pandoc", argv, produced, PANDOC_TIMEOUT_S


def _run(tool: str, argv: list[str], *, cwd: Path, timeout: int, label: str) -> None:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{tool} took longer than {timeout} seconds on {label}. Try a smaller file.") from exc
    except OSError as exc:
        raise ValueError(f"{tool} could not be started: {exc}") from exc
    if proc.returncode == 0:
        return
    stderr = proc.stderr.decode("utf-8", "replace").strip()
    stdout = proc.stdout.decode("utf-8", "replace").strip()
    detail = "\n".join(part for part in (stderr, stdout) if part)[-MAX_TAIL_CHARS:]
    message = f"{tool} could not convert {label}: {detail}"
    if "source file could not be loaded" in (stderr + stdout).lower():
        message += (
            " LibreOffice may already be open with another document, or it cannot read that location. "
            "Close other LibreOffice windows and keep the file in a normal (non-hidden) folder."
        )
    raise ValueError(message)


def convert_file(src: Path, to: str, out: Path) -> str:
    """Convert ``src`` to ``out`` (which must not exist yet) and return the converter used."""
    source = _canonical(src.suffix.lower().lstrip("."))
    target = _canonical(to)
    route = _route(source, target)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="convert-", dir=out.parent) as scratch:
        work = Path(scratch)
        label, argv, produced, timeout = _plan(route, src, target, work, out.name)
        _run(label, argv, cwd=work, timeout=timeout, label=src.name)
        if not produced.is_file():
            raise ValueError(f"{label} finished but wrote no file for {src.name}. Check that the input is not damaged.")
        os.replace(produced, out)
    return label


def _free_name(path: Path) -> Path:
    """Return ``path`` or the first ``<stem>-N<suffix>`` that does not exist yet."""
    if not path.exists():
        return path
    for number in range(2, MAX_NAME_TRIES):
        candidate = path.with_name(f"{path.stem}-{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Too many files named {path.stem}{path.suffix} already exist; choose an output name.")


def _text(args: dict[str, Any], name: str, *, required: bool = False, limit: int = 1024) -> str:
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


def _validate_convert(args: dict[str, Any]) -> dict[str, Any]:
    path = _text(args, "path", required=True)
    to = normalize_target(_text(args, "to", required=True, limit=40))
    if not to:
        raise ValueError("to is required, for example pdf, docx, md, html or txt")
    if not FORMAT_NAME.match(to):
        raise ValueError("to must be a format name such as pdf, docx, md, html, txt or csv")
    return {"path": path, "to": to, "output": _text(args, "output") or None}


def _paths(args: dict[str, Any], workspace: Any) -> tuple[Path, Path]:
    """(source, output) for a validated call; the output never overwrites an existing file."""
    src = workspace.resolve(args["path"], must_exist=True)  # SandboxError propagates
    if not src.is_file():
        raise ValueError(f"{workspace.public(src)} is a folder. Pass the path of one document.")
    to = args["to"]
    source = _canonical(src.suffix.lower().lstrip("."))
    if source == to:
        _route(source, to)  # raises the same-format message before any name is chosen
    if args.get("output"):
        out = workspace.resolve(args["output"])  # must stay inside the workspace
        if out.suffix.lower() != f".{to}":
            raise ValueError(f"output must end with .{to} so the file type is clear, for example {out.stem}.{to}")
    else:
        out = src.with_suffix(f".{to}")
    return src, _free_name(out)


def _touches(args: dict[str, Any], workspace: Any) -> list[Path]:
    """The file a conversion will create, so a checkpoint can remove it on undo."""
    try:
        return [_paths(args, workspace)[1]]
    except Exception:  # an invalid call creates nothing; execute reports the error
        return []


def _exec_convert(args: dict[str, Any], ctx: Any) -> Any:
    src, out = _paths(args, ctx.workspace)
    shown = ctx.workspace.public(src)
    to = args["to"]
    converter = convert_file(src, to, out)
    size = out.stat().st_size
    shown_out = ctx.workspace.public(out)
    hint = "" if to == "csv" else "\nRead it with doc_read or doc_search."
    return tool_result(
        True,
        f"Converted {shown} → {shown_out} ({size:,} bytes) with {converter}.{hint}",
        meta={"output": shown_out, "converter": converter},
    )


CONVERT_DOCUMENT = ExtensionTool(
    name="convert_document",
    description=(
        "Convert a document to another format with the converters installed on this computer: "
        "Markdown/HTML/DOCX/ODT/EPUB/RST/LaTeX/text with pandoc, PDF output, Office files (Word, Excel, "
        "PowerPoint, LibreOffice) with LibreOffice, and PDF to text. Example: path='report.md', to='pdf'. "
        "The original is never changed."
    ),
    parameters=schema(
        properties={
            "path": {"type": "string", "description": "Workspace-relative path of the document to convert."},
            "to": {
                "type": "string",
                "description": 'Target format, such as "pdf", "docx", "md", "html", "txt" or "csv".',
            },
            "output": {
                "type": "string",
                "description": (
                    "Optional workspace-relative output path ending in the new extension. "
                    "Default: same folder and name with the new extension."
                ),
            },
        },
        required=["path", "to"],
    ),
    risk="write",
    phase="writing",
    label="Converting a document",
    summary=lambda a: f"Convert {a.get('path', '')} to {a.get('to', '')}",
    validate=_validate_convert,
    execute=_exec_convert,
    category="research",
    touches=_touches,
)

TOOLS = (CONVERT_DOCUMENT,)
