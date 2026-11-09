"""Static checks for the Models panel: no network, no browser."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_JS = ROOT / "static" / "js" / "models.js"
MOCK_JS = ROOT / "static" / "js" / "mock_models.js"
API_JS = ROOT / "static" / "js" / "api.js"
MODELS_CSS = ROOT / "static" / "css" / "models.css"

# Shared classes from shell.css, app.css and the sheet that the panel reuses.
SHARED_CLASSES = {
    "panel-section",
    "panel-title",
    "bar",
    "panel-empty",
    "panel-error",
    "btn",
    "btn-primary",
    "btn-ghost",
    "btn-danger",
    "text-btn",
    "icon-btn",
    "icon",
    "spinner",
    "search-field",
}


def read(path):
    return path.read_text(encoding="utf-8")


def api_keys():
    source = read(API_JS)
    block = source[source.index("export const api = {") :]
    keys = set()
    for line in block.splitlines():
        if line.startswith("};"):
            break
        match = re.match(r"^  (\w+)\s*[:(]", line)
        if match:
            keys.add(match.group(1))
    return keys


def test_models_module_exports_the_contract():
    source = read(MODELS_JS)
    assert "export function createModelsView(" in source
    assert "return { open, refreshIfOpen }" in source


def test_mock_module_exports_mock_models():
    assert "export function mockModels(" in read(MOCK_JS)


def test_no_innerhtml_and_no_lm_link():
    source = read(MODELS_JS)
    assert "innerHTML" not in source
    assert "LM Link" not in source


def test_every_api_call_exists_in_api_js():
    source = read(MODELS_JS)
    called = set(re.findall(r"\bapi\.(\w+)\(", source))
    assert called, "the panel should call the API"
    missing = sorted(called - api_keys())
    assert not missing, f"api methods missing from api.js: {missing}"


def test_css_classes_are_panel_or_shared():
    css = read(MODELS_CSS)
    # Class selectors only; strip comments first so words in them are ignored.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    classes = set(re.findall(r"\.([A-Za-z_][\w-]*)", css))
    assert classes, "models.css should style the panel"
    stray = sorted(c for c in classes if not c.startswith("mdl-") and c not in SHARED_CLASSES)
    assert not stray, f"classes outside mdl- and the shared set: {stray}"


def test_css_uses_no_new_colours_gradients_or_shadows():
    css = read(MODELS_CSS)
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css), "hex colour in models.css"
    assert not re.search(r"rgba?\(", css), "literal rgb colour in models.css"
    assert "gradient" not in css
    assert "box-shadow" not in css
    assert "filter:" not in css


def test_no_emoji_in_panel_text():
    emoji = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
    for path in (MODELS_JS, MOCK_JS, MODELS_CSS):
        assert not emoji.search(read(path)), f"emoji in {path.name}"
