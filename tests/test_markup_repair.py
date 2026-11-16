"""A markup file whose first tag lost its '<' is repaired at the call; a page that shows markup as text is named."""

from __future__ import annotations

from pathlib import Path

import pytest

from kiwibro import pagecheck
from kiwibro import toolcall as tc
from kiwibro import tools
from kiwibro.sandbox import Workspace

SPECS = tools.TOOL_SPECS


def _write(path: str, content: str) -> tc.PreparedCall:
    call = {"id": "w1", "function": {"name": "write_file", "arguments": {"path": path, "content": content}}}
    return tc.prepare_call(call, SPECS)


@pytest.mark.parametrize(
    "content, fixed",
    [
        ('html lang="en">\n<head><title>Kiwi</title></head>', '<html lang="en">\n<head><title>Kiwi</title></head>'),
        ('!DOCTYPE html>\n<html>', '<!DOCTYPE html>\n<html>'),
        ('  head>\n<body></body>', '  <head>\n<body></body>'),
        ('svg xmlns="http://www.w3.org/2000/svg"><circle r="4"/></svg>', '<svg xmlns="http://www.w3.org/2000/svg"><circle r="4"/></svg>'),
    ],
)
def test_a_markup_file_whose_first_tag_lost_its_bracket_gets_it_back(content: str, fixed: str) -> None:
    prepared = _write("kiwitree.html", content)
    assert prepared.args["content"] == fixed
    assert any("added the '<' that the first tag lost" in note for note in prepared.notes)


def test_content_that_already_opens_with_a_bracket_is_not_touched() -> None:
    content = '<!DOCTYPE html>\n<html lang="en">'
    prepared = _write("index.html", content)
    assert prepared.args["content"] == content
    assert not any("first tag" in note for note in prepared.notes)


def test_a_file_that_is_not_markup_is_not_touched() -> None:
    prepared = _write("notes.py", 'html lang="en">\n')
    assert prepared.args["content"] == 'html lang="en">\n'
    assert not prepared.notes


def test_plain_text_in_a_markup_file_is_not_touched() -> None:
    prepared = _write("page.html", "Heading text for the page")
    assert prepared.args["content"] == "Heading text for the page"


def test_the_repair_is_recorded_in_the_notes_the_model_sees() -> None:
    prepared = _write("kiwitree.html", 'html lang="en">')
    assert prepared.notes[-1] == "added the '<' that the first tag lost (html)"


@pytest.mark.parametrize(
    "shown",
    ['html lang="en">', "<div class=\"tree\">", "</div>", "<!DOCTYPE html>", "<script src=x.js>", "body style=\"margin:0\""],
)
def test_markup_shown_as_text_is_named(shown: str) -> None:
    found = pagecheck._from_report({"vis": f"Kiwi walks\n{shown}\n", "tl": 9, "cv": 1, "e": [], "f": []}, Path("/x/k.html"), False)
    markup = [item for item in found if item.code == "markup"]
    assert len(markup) == 1
    assert markup[0].text == f'Part of the page\'s code shows as text: "{shown}". The file\'s markup is broken there.'


def test_ordinary_visible_text_is_not_markup() -> None:
    report = {"vis": "Kiwi walks to the tree. Speed < fast and a/b </ c", "tl": 40, "cv": 1, "e": [], "f": []}
    assert not [item for item in pagecheck._from_report(report, Path("/x/k.html"), False) if item.code == "markup"]


@pytest.mark.skipif(pagecheck.find_chrome() is None, reason="Chrome is not installed here")
def test_the_real_browser_names_markup_shown_as_text_and_not_a_code_example(tmp_path: Path) -> None:
    (tmp_path / "broken.html").write_text(
        'html lang="en">\n<head><meta charset="utf-8"></head><body><p>Kiwi walks to the tree.</p></body>\n', encoding="utf-8")
    (tmp_path / "example.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"></head><body><h1>Tags</h1>'
        '<pre><code>&lt;div class="tree"&gt;&lt;/div&gt;</code></pre><p>Use <code>&lt;svg&gt;</code> for art.</p>'
        '<p>Kiwi walks to the tree.</p></body></html>\n',
        encoding="utf-8")
    workspace = Workspace(tmp_path)
    broken = pagecheck.look(workspace, "broken.html")
    assert [item.code for item in broken.findings] == ["markup"]
    assert broken.findings[0].text.startswith('Part of the page\'s code shows as text: "html lang="en">')
    example = pagecheck.look(workspace, "example.html")
    assert not [item for item in example.findings if item.code == "markup"]
