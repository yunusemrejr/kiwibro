"""The music kit: one finished page that composes a loop with the Web Audio API.

The page itself is checked in a headless browser when Chrome is there (playing, the grid moving and the
WAV download); the rest is the manifest and the phrases that call the kit from a message.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiwibro import components  # noqa: E402
from kiwibro.sandbox import Workspace  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LIBRARY = REPO / "kiwibro" / "components" / "library"
KIT = LIBRARY / "web" / "music"


def manifest() -> dict:
    return json.loads((KIT / "component.json").read_text(encoding="utf-8"))


def test_the_kit_is_a_finished_page_with_five_blanks():
    data = manifest()
    assert data["name"] == "web/music"
    assert data["kind"] == components.KIT
    assert data["finished"] is True
    assert data["change"] and "{file}" in data["change"]
    assert list(data["blanks"]) == ["title", "style", "tempo", "key", "bars"]
    assert components.Library(LIBRARY).get("web/music").file_paths() == ["index.html"]


def test_the_page_has_the_settings_lines_the_model_edits_later():
    page = (KIT / "files" / "index.html").read_text(encoding="utf-8")
    block = page.split('<script type="text/plain" id="settings">')[1].split("</script>")[0]
    assert [line.split(" ")[0] for line in block.strip().splitlines()] == [
        "title", "style", "tempo", "key", "bars"]
    assert '{{style}}' in block and '<script type="text/plain" id="scene">' not in page
    # the page stays a single file with no library and no internet
    assert "http://" not in page and "https://" not in page
    assert "@import" not in page and "fonts.googleapis" not in page


def test_the_defaults_fill_into_a_playable_page(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    text = components.use("web/music", workspace=ws, path="tune")
    assert text.splitlines()[0] == "Used web/music in tune: 1 file written"
    page = (ws.root / "tune" / "index.html").read_text(encoding="utf-8")
    assert "<title>Night loop</title>" in page
    assert "style lofi\n" in page and "tempo own\n" in page and "key own\n" in page and "bars 8\n" in page
    assert "{{" not in page and page.rstrip().endswith("</html>")


def test_a_requested_style_and_key_reach_the_settings(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    components.use("web/music", {"title": "Chip run", "style": "a bouncy chiptune tune", "tempo": "144",
                                 "key": "E minor", "bars": "4"}, workspace=ws, path="chip")
    page = (ws.root / "chip" / "index.html").read_text(encoding="utf-8")
    for line in ("title Chip run", "style a bouncy chiptune tune", "tempo 144", "key E minor", "bars 4"):
        assert line + "\n" in page


@pytest.mark.parametrize("message, name", [
    ("make me a lofi beat", "web/music"),
    ("make a chiptune song", "web/music"),
    ("write a chiptune tune", "web/music"),
    ("make relaxing ambient music", "web/music"),
    ("make a piano melody in A minor", "web/music"),
    ("make a lo-fi beat to study to", "web/music"),
    ("write a melody for my game", "web/music"),
    ("build me a music page", "web/music"),
    ("create a soundtrack", "web/music"),
    ("make me a beat", "web/music"),
    ("write me a little tune", "web/music"),
    ("want some lofi", "web/music"),
])
def test_a_message_about_music_gets_the_music_kit(message: str, name: str):
    found = components.suggest(message)
    assert found is not None and found.name == name


@pytest.mark.parametrize("message", [
    "compose a chiptune song",
    "a piano melody in A minor",
    "chiptune",
    "ambient music",
    "a beat",
    "soundtrack",
    "music page",
    "web audio",
])
def test_the_match_phrases_are_read_even_without_a_making_verb(message: str):
    """suggest only offers a kit to a message that asks for something to be made (components.py `_MAKES`);
    a bare "chiptune" names something but asks for nothing, so the kit is not suggested. Its phrases are
    still read by called_for, which is what the test above needs them for."""
    assert components.called_for(components.Library(LIBRARY).get("web/music"), message)



@pytest.mark.parametrize("message", [
    "make a bouncing balls animation",
    "make a 3d scene of a dog walking to a house",
    "make a dashboard",
])
def test_a_message_that_is_not_music_gets_no_music_kit(message: str):
    found = components.suggest(message)
    assert found is None or found.name != "web/music"


@pytest.mark.parametrize("message", [
    "make a music player for my files",
    "add my spotify playlist to the music page",
    "make a playlist of my music",
    "build a music visualizer for my songs",
    "make a game with chiptune music",
    "write sheet music for twinkle twinkle",
])
def test_a_music_player_or_a_playlist_is_not_a_music_kit(message: str):
    found = components.suggest(message)
    assert found is None or found.name != "web/music"


def test_the_advice_tells_the_model_the_page_is_finished():
    advice = components.advice(components.Library(LIBRARY).get("web/music"))
    assert advice.startswith(components.SUGGEST_HEADER)
    assert 'component(action="use", name="web/music"' in advice
    assert "- style: " in advice and "chiptune" in advice


def test_the_music_page_is_named_a_big_kit():
    """The page is one finished file over the 16 KB starter-file ceiling, like the 2D animation:
    test_components.BIG_KITS is where the library says so, and this test keeps the entry honest."""
    import test_components

    assert "web/music" in test_components.BIG_KITS
    size = sum(item.stat().st_size for item in KIT.rglob("*") if item.is_file())
    assert test_components.BYTES < size <= test_components.BIG_BYTES
    lines = (KIT / "files" / "index.html").read_text(encoding="utf-8").splitlines()
    assert test_components.LINES < len(lines) <= test_components.BIG_LINES
