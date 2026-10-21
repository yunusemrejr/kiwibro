"""Offline tests for the on-disk document library (kiwibro/library.py).

Every fixture is built inside the test (tmp_path, zipfile, hand-written PDF bytes).
No network and no real kiwibro.db is touched.
"""

from __future__ import annotations

import os
import re
import shutil
import types
import zipfile
from pathlib import Path

import pytest

from kiwibro import library
from kiwibro.library import (
    DOC_ADD,
    DOC_LIST,
    DOC_READ,
    DOC_SEARCH,
    TOOLS,
    Library,
    chunk_sections,
    extract,
    get_library,
    library_for,
)
from kiwibro.sandbox import SandboxError, Workspace


# --- fixtures and builders -------------------------------------------------------


def _lib(tmp_path: Path) -> Library:
    return Library(tmp_path / "data" / "library.db")


def _ctx(tmp_path: Path) -> types.SimpleNamespace:
    (tmp_path / "data").mkdir(exist_ok=True)
    return types.SimpleNamespace(
        workspace=Workspace(tmp_path / "ws"),
        memory=types.SimpleNamespace(db=types.SimpleNamespace(path=str(tmp_path / "data" / "kiwibro.db"))),
        search=None,
        conversation_id="c1",
    )


def _docx(path: Path) -> None:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Quarterly Report</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>Revenue grew strongly.</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Costs stayed flat.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", body)


def _pdf(path: Path, text: str = "Hello Kiwibro PDF") -> None:
    """A minimal, valid one-page PDF with an uncompressed content stream."""
    stream = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def _xlsx(path: Path) -> None:
    sheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{sheet_ns}" xmlns:r="{rel_ns}"><sheets>'
            '<sheet name="Sales" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        zf.writestr("xl/sharedStrings.xml", f'<sst xmlns="{sheet_ns}"><si><t>Region</t></si><si><t>Kiwis</t></si></sst>')
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{sheet_ns}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>42</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" t="inlineStr"><is><t>exotic</t></is></c></row>'
            "</sheetData></worksheet>",
        )


def _long_text(paragraphs: int = 12, size: int = 600) -> str:
    return "\n\n".join(f"Paragraph {index} " + ("filler " * size)[: size - 12] for index in range(paragraphs))


# --- extraction ------------------------------------------------------------------


def test_markdown_sections_follow_hash_headings(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Alpha\n\nFirst paragraph about zebras.\n\n# Beta\n\nSecond about rivers.\n", encoding="utf-8")
    sections = list(extract(path))
    assert [(heading, location) for heading, location, _ in sections] == [
        ("Alpha", "lines 1-4"),
        ("Beta", "lines 5-7"),
    ]
    assert "zebras" in sections[0][2] and "rivers" not in sections[0][2]


def test_markdown_ignores_hash_lines_inside_code_fences(tmp_path: Path) -> None:
    path = tmp_path / "fenced.md"
    path.write_text("# Real\n\n```\n# not a heading\n```\n", encoding="utf-8")
    assert [heading for heading, _, _ in extract(path)] == ["Real"]


def test_text_and_code_yield_line_sections(tmp_path: Path) -> None:
    text = tmp_path / "log.txt"
    text.write_text("first line\nsecond line\n", encoding="utf-8")
    code = tmp_path / "tool.py"
    code.write_text("def run():\n    return 1\n", encoding="utf-8")
    assert list(extract(text)) == [("", "lines 1-2", "first line\nsecond line")]
    assert list(extract(code)) == [("", "lines 1-2", "def run():\n    return 1")]


def test_html_skips_script_and_uses_headings(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Page</title><script>var secret = 1;</script></head><body>"
        "<nav>menu words</nav><h2>Install</h2><p>Run the installer now.</p>"
        "<h2>Usage</h2><ul><li>Item one</li></ul></body></html>",
        encoding="utf-8",
    )
    sections = list(extract(path))
    text = " ".join(section[2] for section in sections)
    assert "secret" not in text and "menu words" not in text
    assert "Run the installer now." in text and "Item one" in text
    headings = [heading for heading, _, body in sections if body == ""]
    assert headings == ["Install", "Usage"]


def test_docx_heading_and_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "report.docx"
    _docx(path)
    sections = list(extract(path))
    assert sections[0][0] == "Quarterly Report"
    body = [text for heading, _, text in sections if text]
    assert body == ["Revenue grew strongly.", "Costs stayed flat."]
    assert all(heading == "Quarterly Report" for heading, _, _ in sections)


def test_csv_rows_become_named_pairs(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,city\nAda,London\nGrace,New York\n", encoding="utf-8")
    ((heading, location, text),) = list(extract(path))
    assert location == "rows 2-3"
    assert text == "name: Ada; city: London\n\nname: Grace; city: New York"


def test_pdf_fallback_parser_reads_uncompressed_stream(tmp_path: Path) -> None:
    path = tmp_path / "hello.pdf"
    _pdf(path)
    meta: dict[str, str] = {}
    sections = list(library._pdf_fallback_sections(path, meta))
    assert sections == [("", "part 1", "Hello Kiwibro PDF")]
    assert meta["quality"] == "fallback"


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_pdftotext_path_reports_pages(tmp_path: Path) -> None:
    path = tmp_path / "hello.pdf"
    _pdf(path)
    meta: dict[str, str] = {}
    assert list(extract(path, meta)) == [("", "page 1", "Hello Kiwibro PDF")]
    assert "quality" not in meta


def test_odt_pptx_xlsx_epub_extractors(tmp_path: Path) -> None:
    odt = tmp_path / "memo.odt"
    with zipfile.ZipFile(odt, "w") as zf:
        zf.writestr(
            "content.xml",
            '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>'
            "<text:h>Odt Head</text:h><text:p>Odt body text.</text:p></office:text></office:body>"
            "</office:document-content>",
        )
    assert list(extract(odt)) == [("Odt Head", "", ""), ("Odt Head", "", "Odt body text.")]

    pptx = tmp_path / "deck.pptx"
    with zipfile.ZipFile(pptx, "w") as zf:
        zf.writestr(
            "ppt/slides/slide1.xml",
            '<p:sld xmlns:p="urn:p" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            "<a:p><a:r><a:t>Hello slide</a:t></a:r></a:p></p:sld>",
        )
    assert list(extract(pptx)) == [("", "slide 1", "Hello slide")]

    xlsx = tmp_path / "sales.xlsx"
    _xlsx(xlsx)
    ((_, location, text),) = list(extract(xlsx))
    assert location == "sheet Sales rows 1-2"
    assert text == "A: Region | B: 42\n\nA: Kiwis | B: exotic"

    epub = tmp_path / "book.epub"
    with zipfile.ZipFile(epub, "w") as zf:
        zf.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
        )
        zf.writestr(
            "OEBPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
            '<item id="c1" href="ch1.xhtml"/></manifest><spine><itemref idref="c1"/></spine></package>',
        )
        zf.writestr("OEBPS/ch1.xhtml", "<html><body><h1>Chapter One</h1><p>Epub text.</p></body></html>")
    assert list(extract(epub)) == [
        ("Chapter One", "chapter 1", ""),
        ("Chapter One", "chapter 1", "Epub text."),
    ]


def test_unsupported_extension_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported document type: .xyz"):
        extract(tmp_path / "file.xyz")


# --- chunking --------------------------------------------------------------------


def test_chunks_never_exceed_maximum_and_split_on_sentences() -> None:
    text = " ".join(f"Sentence number {index} is here." for index in range(300))
    chunks = list(chunk_sections([("", "lines 1-1", text)]))
    assert len(chunks) > 3
    assert all(0 < len(body) <= 1600 for _, _, body in chunks)


def test_hard_wrap_overlaps_giant_word_runs() -> None:
    chunks = list(chunk_sections([("", "", "x" * 5000)]))
    assert len(chunks) >= 4
    assert all(len(body) <= 1600 for _, _, body in chunks)


def test_heading_change_starts_new_chunk_and_locations_merge() -> None:
    chunks = list(
        chunk_sections(
            [
                ("A", "page 3", "first para"),
                ("A", "page 4", "second para"),
                ("B", "page 5", "other section"),
            ]
        )
    )
    assert chunks == [("A", "page 3-4", "first para\n\nsecond para"), ("B", "page 5", "other section")]


def test_whitespace_only_sections_produce_no_chunks() -> None:
    assert list(chunk_sections([("", "lines 1-1", "   \n\n  ")])) == []


# --- indexing and search -------------------------------------------------------------


def test_add_file_statuses_cover_unchanged_and_updated(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    path = tmp_path / "notes.md"
    path.write_text("# Alpha\n\nzebras\n", encoding="utf-8")
    first = lib.add_file(path, "notes.md")
    assert first["status"] == "indexed" and first["handle"] == "D1" and first["title"] == "Alpha"
    assert lib.add_file(path, "notes.md")["status"] == "unchanged"
    os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
    assert lib.add_file(path, "notes.md")["status"] == "unchanged"
    path.write_text("# Alpha\n\nzebras and giraffes\n", encoding="utf-8")
    assert lib.add_file(path, "notes.md")["status"] == "updated"
    assert lib.search("giraffes")[0]["handle"] == "D1"


def test_add_file_skips_empty_unsupported_and_outside_names(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert lib.add_file(empty, "empty.txt")["status"] == "skipped"
    binary = tmp_path / "image.png"
    binary.write_bytes(b"\x89PNG")
    result = lib.add_file(binary, "image.png")
    assert result["status"] == "skipped" and "unsupported" in result["reason"]


def test_search_ranks_the_right_document_first(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    (tmp_path / "a.md").write_text("# Animals\n\nThe zebras graze on the plain.\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Rivers carry sediment to the sea.\n", encoding="utf-8")
    (tmp_path / "c.html").write_text("<h1>Cooking</h1><p>Simmer the stock slowly.</p>", encoding="utf-8")
    for name in ("a.md", "b.txt", "c.html"):
        lib.add_file(tmp_path / name, name)
    hits = lib.search("zebras")
    assert hits[0]["handle"] == "D1"
    assert hits[0]["title"] == "Animals"
    assert "[zebras]" in hits[0]["snippet"]
    assert lib.search("sediment")[0]["handle"] == "D2"


def test_fts_hostile_queries_never_raise(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    (tmp_path / "a.txt").write_text("plain words about kiwis\n", encoding="utf-8")
    lib.add_file(tmp_path / "a.txt", "a.txt")
    for query in ['"unbalanced AND (or *', "-", "NOT", "NEAR(a b)", "title:x", "''", "OR"]:
        assert isinstance(lib.search(query), list)
    assert lib.search("-") == []
    assert lib.search("kiwis OR (")[0]["handle"] == "D1"


def test_and_query_falls_back_to_or_to_fill_results(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    (tmp_path / "one.txt").write_text("kiwibro is a local assistant\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("zebras roam the savanna\n", encoding="utf-8")
    lib.add_file(tmp_path / "one.txt", "one.txt")
    lib.add_file(tmp_path / "two.txt", "two.txt")
    hits = lib.search("kiwibro zebras", limit=6)
    assert {hit["handle"] for hit in hits} == {"D1", "D2"}


def test_default_search_caps_hits_per_document(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    (tmp_path / "many.txt").write_text(_long_text(12, 600), encoding="utf-8")
    lib.add_file(tmp_path / "many.txt", "many.txt")
    assert len(lib.search("filler", limit=6)) == 3
    assert len(lib.search("filler", limit=6, document="D1")) == 6


def test_read_paginates_and_reports_next(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    (tmp_path / "long.txt").write_text(_long_text(20, 600), encoding="utf-8")
    lib.add_file(tmp_path / "long.txt", "long.txt")
    total = lib.resolve("D1")["chunks"]
    assert total >= 5
    page = lib.read("D1", chunk=1, count=2)
    assert [item["seq"] for item in page["chunks"]] == [1, 2]
    assert page["next"] == 3 and page["total"] == total
    tail = lib.read("D1", chunk=total, count=3)
    assert [item["seq"] for item in tail["chunks"]] == [total]
    assert tail["next"] is None
    assert len(lib.read("D1", count=99)["chunks"]) == 8
    with pytest.raises(ValueError, match="chunk must be between"):
        lib.read("D1", chunk=total + 1)


def test_read_section_jumps_to_heading(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    (tmp_path / "guide.md").write_text(
        "# Overview\n\nIntro text here.\n\n## Installation\n\nRun pip install.\n\n## Usage\n\nCall run().\n",
        encoding="utf-8",
    )
    lib.add_file(tmp_path / "guide.md", "guide.md")
    found = lib.read("D1", section="install")
    assert found["chunks"][0]["heading"] == "Installation"
    assert "pip install" in found["chunks"][0]["text"]
    with pytest.raises(ValueError, match="no heading"):
        lib.read("D1", section="billing")
    outline = lib.outline("D1")
    assert [row["heading"] for row in outline] == ["Overview", "Installation", "Usage"]


def test_resolve_accepts_handle_number_title_and_path(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    path = tmp_path / "alpha.md"
    path.write_text("# Alpha\n\nbody\n", encoding="utf-8")
    lib.add_file(path, "alpha.md")
    for ref in ("D1", "d1", "1", "alpha", "Alpha", str(path), "alpha.md"):
        assert lib.resolve(ref)["handle"] == "D1", ref
    with pytest.raises(ValueError, match="doc_list"):
        lib.resolve("nothing like this")


def test_ambiguous_titles_list_candidates(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    (tmp_path / "report-2023.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "report-2024.txt").write_text("beta\n", encoding="utf-8")
    lib.add_file(tmp_path / "report-2023.txt", "report-2023.txt")
    lib.add_file(tmp_path / "report-2024.txt", "report-2024.txt")
    with pytest.raises(ValueError, match="D1") as info:
        lib.resolve("report")
    assert "D2" in str(info.value)


def test_remove_drops_document_and_chunks(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    (tmp_path / "gone.txt").write_text("ephemeral text\n", encoding="utf-8")
    lib.add_file(tmp_path / "gone.txt", "gone.txt")
    assert lib.remove("D1") is True
    assert lib.remove("D1") is False
    assert lib.stats()["chunks"] == 0
    assert lib.search("ephemeral") == []


def test_add_folder_skips_hidden_and_build_dirs(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "docs").mkdir(parents=True)
    (root / "node_modules").mkdir()
    (root / ".secret").mkdir()
    (root / "docs" / "guide.md").write_text("# Guide\n\nhello\n", encoding="utf-8")
    (root / "node_modules" / "dep.md").write_text("# Dep\n\nhidden\n", encoding="utf-8")
    (root / ".secret" / "key.txt").write_text("hidden\n", encoding="utf-8")
    rows = _lib(tmp_path).add_folder(root, lambda path: path.relative_to(root).as_posix())
    assert [(row["display"], row["status"]) for row in rows] == [("docs/guide.md", "indexed")]


def test_add_folder_ignores_symlinks_that_leave_the_workspace(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("# Secret\n\ndo not index\n", encoding="utf-8")
    (root / "inside.md").write_text("# Inside\n\nindexed\n", encoding="utf-8")
    (root / "link.md").symlink_to(outside)
    rows = _lib(tmp_path).add_folder(root, lambda path: path.name)
    assert [row["display"] for row in rows] == ["inside.md"]


def test_corrupt_documents_are_skipped_with_a_reason(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"%PDF-1.4 truncated garbage")
    assert lib.add_file(broken_pdf, "broken.pdf")["status"] == "skipped"
    bad_docx = tmp_path / "bad.docx"
    bad_docx.write_bytes(b"not a zip archive")
    result = lib.add_file(bad_docx, "bad.docx")
    assert result["status"] == "skipped" and "could not extract" in result["reason"]
    assert lib.stats()["documents"] == 0


def test_sync_indexes_new_files_removes_deleted_and_throttles(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    root = tmp_path / "ws"
    root.mkdir()
    display = lambda path: path.relative_to(root).as_posix()  # noqa: E731
    (root / "one.md").write_text("# One\n\nfirst\n", encoding="utf-8")
    assert lib.sync(root, display, min_interval=0)["indexed"] == 1
    (root / "two.txt").write_text("second\n", encoding="utf-8")
    assert lib.sync(root, display, min_interval=0)["indexed"] == 1
    (root / "one.md").unlink()
    counts = lib.sync(root, display, min_interval=0)
    assert counts["removed"] == 1
    assert [doc["display"] for doc in lib.list_documents()] == ["two.txt"]
    (root / "three.txt").write_text("third\n", encoding="utf-8")
    assert lib.sync(root, display)["indexed"] == 0  # throttled by min_interval


def test_stash_text_is_readable_but_hidden_from_default_search(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    output = "\n".join(f"row {index}: value {index * 3}" for index in range(400))
    stashed = lib.stash_text(output, title="big tool output", source="shell")
    handle = stashed["handle"]
    assert stashed["chunks"] >= 2
    assert abs(stashed["chars"] - len(output)) <= len(output) * 0.02
    stored = lib.read(handle, chunk=1, count=8)
    assert stored["doc"]["kind"] == "output"
    assert all(re.fullmatch(r"row \d+: value \d+", line) for item in stored["chunks"] for line in item["text"].split("\n"))
    assert lib.read(handle, chunk=1)["doc"]["kind"] == "output"
    assert lib.search("value") == []
    assert lib.search("value", include_outputs=True)[0]["handle"] == handle
    assert lib.search("value", document=handle)[0]["handle"] == handle
    assert lib.stats()["outputs"] == 1
    assert lib.list_documents() == []


def test_output_cap_prunes_oldest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "MAX_OUTPUTS", 3)
    lib = _lib(tmp_path)
    handles = [lib.stash_text(f"payload {index}", title=f"out {index}")["handle"] for index in range(5)]
    assert lib.stats()["outputs"] == 3
    with pytest.raises(ValueError):
        lib.resolve(handles[0])
    assert lib.resolve(handles[-1])["title"] == "out 4"


def test_get_library_is_cached_per_path(tmp_path: Path) -> None:
    db = tmp_path / "data" / "library.db"
    assert get_library(db) is get_library(db)


def test_library_for_uses_data_dir_of_context(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert library_for(ctx).db_path == (tmp_path / "data" / "library.db").resolve()


# --- tools -----------------------------------------------------------------------------


def test_tool_registry_names() -> None:
    assert [tool.name for tool in TOOLS] == ["doc_search", "doc_read", "doc_add", "doc_list"]
    assert all(tool.risk == "read" and tool.permission == "auto" for tool in TOOLS)


@pytest.mark.parametrize("tool", [DOC_SEARCH, DOC_READ, DOC_ADD])
def test_validate_rejects_missing_required_args(tool) -> None:
    with pytest.raises(ValueError):
        tool.validate({})


def test_validate_clamps_numbers_and_defaults() -> None:
    assert DOC_SEARCH.validate({"query": "zebra", "limit": 99})["limit"] == 10
    assert DOC_READ.validate({"document": "D1"}) == {
        "document": "D1",
        "chunk": None,
        "count": 3,
        "section": None,
    }
    assert DOC_LIST.validate({}) == {"document": None}


def test_tool_execute_round_trip(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.workspace.root.mkdir(parents=True, exist_ok=True)
    (ctx.workspace.root / "notes.md").write_text(
        "# Alpha\n\nThe zebras graze.\n\n## Beta\n\nRivers flow.\n", encoding="utf-8"
    )

    added = DOC_ADD.execute(DOC_ADD.validate({"path": "notes.md"}), ctx)
    assert added.ok and "indexed" in added.content

    found = DOC_SEARCH.execute(DOC_SEARCH.validate({"query": "zebras"}), ctx)
    assert found.ok
    assert "[1] D1" in found.content
    assert 'doc_read(document="D1", chunk=1)' in found.content

    missing = DOC_SEARCH.execute(DOC_SEARCH.validate({"query": "quasar"}), ctx)
    assert missing.ok and "No matches" in missing.content

    read = DOC_READ.execute(DOC_READ.validate({"document": "D1", "count": 1}), ctx)
    assert read.ok
    assert "chunk 1 of 2" in read.content or "chunks 1-1 of 2" in read.content
    assert 'Next: doc_read(document="D1", chunk=2)' in read.content

    outline = DOC_LIST.execute(DOC_LIST.validate({"document": "D1"}), ctx)
    assert outline.ok and "Beta" in outline.content

    listing = DOC_LIST.execute(DOC_LIST.validate({}), ctx)
    assert listing.ok and "D1 · Alpha" in listing.content

    bad = DOC_READ.execute(DOC_READ.validate({"document": "D99"}), ctx)
    assert bad.ok is False and "doc_list" in bad.content


def test_doc_add_refuses_paths_outside_workspace(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.workspace.root.mkdir(parents=True, exist_ok=True)
    (tmp_path / "x.md").write_text("# Outside\n", encoding="utf-8")
    with pytest.raises(SandboxError):
        DOC_ADD.execute(DOC_ADD.validate({"path": "../x.md"}), ctx)
