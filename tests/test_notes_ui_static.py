"""Static checks for the Notes panel: no browser, no network."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "js"
CSS = ROOT / "static" / "css"

# Shared classes the notes panel may style or reuse without the nt- prefix.
SHARED_CLASSES = {
    "md",
    "panel-empty",
    "panel-error",
    "btn",
    "icon-btn",
    "search-field",
    "spinner",
    "text-btn",
}


def read(path):
    return path.read_text(encoding="utf-8")


def test_notes_module_exports_create_notes_view():
    src = read(JS / "notes.js")
    assert re.search(r"export\s+function\s+createNotesView\b", src)


def test_html_is_only_assigned_from_markdown_renderer():
    src = read(JS / "notes.js")
    assert not re.search(r"\bhtml\s*:", src), "el() must not take an html attribute"
    lines = [line for line in src.splitlines() if "innerHTML" in line]
    assert len(lines) <= 1, "one markdown render path is enough"
    for line in lines:
        assert "renderMarkdown(" in line


def test_every_api_call_exists_in_api_js():
    src = read(JS / "notes.js")
    api_src = read(JS / "api.js")
    block = api_src[api_src.index("export const api = {") :]
    keys = set(re.findall(r"^\s{2}(\w+)\s*:", block, flags=re.M))
    calls = set(re.findall(r"\bapi\.(\w+)\(", src))
    assert calls, "notes.js should call the API"
    missing = calls - keys
    assert not missing, f"api methods missing from api.js: {sorted(missing)}"


def test_notes_css_class_selectors_use_nt_prefix_or_shared_names():
    css = re.sub(r"/\*.*?\*/", "", read(CSS / "notes.css"), flags=re.S)
    bad = []
    for prelude in re.findall(r"([^{}]+)\{", css):
        if prelude.lstrip().startswith("@"):
            continue
        for name in re.findall(r"\.([A-Za-z_][\w-]*)", prelude):
            if not (name.startswith("nt-") or name in SHARED_CLASSES):
                bad.append(name)
    assert not bad, f"selectors outside the nt- namespace: {sorted(set(bad))}"
