"""Tests for kiwibro/convert.py (document conversion with installed converters).

Fixtures are built inside each test (tmp_path, hand-written PDF and DOCX bytes).
Converters that are not installed skip their tests. LibreOffice runs only when
KIWIBRO_TEST_LIBREOFFICE=1, and then its input lives in a non-hidden folder under
$HOME because the snap-confined LibreOffice cannot read /tmp.
"""

from __future__ import annotations

import os
import shutil
import sys
import types
import uuid
import zipfile
from pathlib import Path

import pytest

from kiwibro import convert, tools
from kiwibro.sandbox import SandboxError, Workspace

HAS_PANDOC = convert._find("pandoc") is not None
HAS_WEASYPRINT = convert._find("weasyprint") is not None
HAS_PDFTOTEXT = convert._find("pdftotext") is not None
RUN_LIBREOFFICE = os.environ.get("KIWIBRO_TEST_LIBREOFFICE") == "1"


def _ctx(root: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(workspace=Workspace(root), memory=None, search=None, conversation_id="c1")


def _convert(ctx, path: str, to: str, output: str | None = None):
    args = {"path": path, "to": to}
    if output is not None:
        args["output"] = output
    return convert.CONVERT_DOCUMENT.execute(convert.CONVERT_DOCUMENT.validate(args), ctx)


def _minimal_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def _minimal_docx(path: Path, text: str) -> None:
    prolog = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            prolog
            + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        zf.writestr(
            "_rels/.rels",
            prolog
            + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        )
        zf.writestr(
            "word/document.xml",
            prolog
            + '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )


@pytest.mark.skipif(not HAS_PANDOC, reason="pandoc is not installed")
def test_markdown_to_docx_and_html(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "report.md").write_text("# Heading Text\n\nSome **bold** words.\n", encoding="utf-8")
    ctx = _ctx(ws)

    html = _convert(ctx, "report.md", "html")
    assert html.ok, html.content
    assert html.meta["output"] == "report.html"
    assert html.meta["converter"] == "pandoc"
    assert "Heading Text" in (ws / "report.html").read_text(encoding="utf-8")

    docx = _convert(ctx, "report.md", "docx")
    assert docx.ok, docx.content
    with zipfile.ZipFile(ws / "report.docx") as zf:
        assert "word/document.xml" in zf.namelist()
    assert "Read it with doc_read or doc_search." in docx.content


@pytest.mark.skipif(not (HAS_PANDOC and HAS_WEASYPRINT), reason="needs pandoc and weasyprint")
def test_markdown_to_pdf(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "report.md").write_text("# Heading Text\n\nBody.\n", encoding="utf-8")
    result = _convert(_ctx(ws), "report.md", "pdf")
    assert result.ok, result.content
    assert (ws / "report.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.skipif(not HAS_PDFTOTEXT, reason="pdftotext is not installed")
def test_pdf_to_txt(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "hello.pdf").write_bytes(_minimal_pdf("Hello Convert"))
    result = _convert(_ctx(ws), "hello.pdf", "txt")
    assert result.ok, result.content
    assert "Hello Convert" in (ws / "hello.txt").read_text(encoding="utf-8")


@pytest.mark.skipif(not RUN_LIBREOFFICE, reason="set KIWIBRO_TEST_LIBREOFFICE=1 to run LibreOffice")
def test_docx_to_pdf_with_libreoffice():
    if convert._soffice() is None:
        pytest.skip("LibreOffice is not installed")
    base = Path.home() / f"kiwibro-convert-test-{uuid.uuid4().hex[:8]}"
    try:
        base.mkdir()
        _minimal_docx(base / "letter.docx", "Hello LibreOffice")
        result = _convert(_ctx(base), "letter.docx", "pdf")
        assert result.ok, result.content
        assert result.meta["converter"] == "LibreOffice"
        assert (base / "letter.pdf").read_bytes().startswith(b"%PDF")
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.skipif(not HAS_PANDOC, reason="pandoc is not installed")
def test_never_overwrites_existing_output(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "report.md").write_text("# First\n", encoding="utf-8")
    ctx = _ctx(ws)

    first = _convert(ctx, "report.md", "docx")
    assert first.ok, first.content
    original = (ws / "report.docx").read_bytes()

    second = _convert(ctx, "report.md", "docx")
    assert second.ok, second.content
    assert second.meta["output"] == "report-2.docx"
    assert (ws / "report-2.docx").exists()
    assert (ws / "report.docx").read_bytes() == original

    third = _convert(ctx, "report.md", "docx")
    assert third.meta["output"] == "report-3.docx"


def test_same_format_is_refused(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.pdf").write_bytes(_minimal_pdf("x"))
    with pytest.raises(ValueError, match="already .pdf"):
        _convert(_ctx(ws), "a.pdf", "pdf")
    assert not (ws / "a-2.pdf").exists()


def test_pdf_only_converts_to_txt(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.pdf").write_bytes(_minimal_pdf("x"))
    with pytest.raises(ValueError, match="Convert the PDF to txt first"):
        _convert(_ctx(ws), "a.pdf", "docx")


def test_unsupported_source_lists_what_is_possible(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "photo.jpg").write_bytes(b"\xff\xd8\xff")
    with pytest.raises(ValueError, match="No converter on this computer reads .jpg"):
        _convert(_ctx(ws), "photo.jpg", "pdf")


def test_missing_pandoc_is_named(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.md").write_text("# Notes\n", encoding="utf-8")
    real = convert._find
    monkeypatch.setattr(convert, "_find", lambda name: None if name == "pandoc" else real(name))
    with pytest.raises(ValueError, match="pandoc is not installed"):
        _convert(_ctx(ws), "notes.md", "docx")


def test_missing_weasyprint_is_named_for_pdf(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.md").write_text("# Notes\n", encoding="utf-8")
    real = convert._find
    monkeypatch.setattr(convert, "_find", lambda name: None if name == "weasyprint" else real(name))
    with pytest.raises(ValueError, match="needs weasyprint"):
        _convert(_ctx(ws), "notes.md", "pdf")


def test_missing_libreoffice_is_named(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "sheet.xlsx").write_bytes(b"PK")
    monkeypatch.setattr(convert, "_soffice", lambda: None)
    with pytest.raises(ValueError, match="LibreOffice is not installed"):
        _convert(_ctx(ws), "sheet.xlsx", "csv")


def test_path_outside_workspace_is_refused(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "outside.md").write_text("# Outside\n", encoding="utf-8")
    with pytest.raises(SandboxError):
        _convert(_ctx(ws), "../outside.md", "html")


def test_output_outside_workspace_is_refused(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.md").write_text("# Notes\n", encoding="utf-8")
    with pytest.raises(SandboxError):
        _convert(_ctx(ws), "notes.md", "html", output="../escape.html")


def test_output_must_end_with_target_extension(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.md").write_text("# Notes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must end with .html"):
        _convert(_ctx(ws), "notes.md", "html", output="notes.txt")


def test_validate_requires_path_and_to():
    spec = convert.CONVERT_DOCUMENT
    with pytest.raises(ValueError, match="path is required"):
        spec.validate({"to": "pdf"})
    with pytest.raises(ValueError, match="to is required"):
        spec.validate({"path": "a.md"})
    with pytest.raises(ValueError, match="to is required"):
        spec.validate({"path": "a.md", "to": "   "})


def test_validate_normalises_target():
    spec = convert.CONVERT_DOCUMENT
    assert spec.validate({"path": "a.pdf", "to": "Markdown"})["to"] == "md"
    assert spec.validate({"path": "a.md", "to": ".PDF"})["to"] == "pdf"
    assert spec.validate({"path": "a.md", "to": "htm"})["to"] == "html"
    assert spec.validate({"path": "a.md", "to": "Word"})["to"] == "docx"
    assert spec.validate({"path": "a.md", "to": "text"})["to"] == "txt"
    assert spec.validate({"path": "a.md", "to": "LaTeX"})["to"] == "tex"
    with pytest.raises(ValueError, match="format name"):
        spec.validate({"path": "a.md", "to": "p/df"})


def test_available_routes_shape():
    routes = convert.available_routes()
    assert isinstance(routes, dict)
    assert all(isinstance(targets, list) for targets in routes.values())
    if HAS_PDFTOTEXT:
        assert routes["pdf"] == ["txt"]


def test_failed_converter_reports_stderr_tail(tmp_path):
    script = "import sys; sys.stderr.write('x' * 500 + 'boom'); sys.exit(3)"
    with pytest.raises(ValueError) as info:
        convert._run("fake", [sys.executable, "-c", script], cwd=tmp_path, timeout=30, label="a.md")
    message = str(info.value)
    assert message.endswith("boom")
    assert len(message) < 400


def test_libreoffice_load_failure_gets_hint(tmp_path):
    script = "import sys; print('source file could not be loaded'); sys.exit(1)"
    with pytest.raises(ValueError, match="cannot read that location"):
        convert._run("LibreOffice", [sys.executable, "-c", script], cwd=tmp_path, timeout=30, label="a.docx")


def test_tool_is_registered():
    assert "convert_document" in tools.TOOL_SPECS
    assert tools.TOOL_SPECS["convert_document"].category == "research"
