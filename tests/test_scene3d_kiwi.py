"""The web/scene-3d kit reads "walking towards ... and hugging it" and coloured skies, and still
picks its kit for the requests it picked before."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from kiwibro import components

KIT = components.LIBRARY_ROOT / "web" / "scene-3d"
HTML = KIT / "files" / "index.html"
REQUEST = (
    "create kiwitree.html and in it write code in the html file that animates a kiwi walking "
    "towards a tree and hugging it. sky should be purple."
)

# The reader is the kit's own code: the words-and-lines part of index.html, run by node.
READER = r"""
const fs = require("fs");
const [file, ...sentences] = process.argv.slice(2);
const html = fs.readFileSync(file, "utf8");
const start = html.indexOf("// ---------------------------------------------------------------- words");
const end = html.indexOf("// ---------------------------------------------------------------- three.js");
const KINDS = {};
for (const match of html.matchAll(/^  (\w+): \{ r:/gm)) KINDS[match[1]] = { r: 1 };
const api = new Function(html.slice(start, end) + "\nreturn { readScene, skyPalette, brightness };")();
const read = sentences.map((sentence) => {
  const { setup, specs, problems } = api.readScene(sentence, KINDS);
  const things = specs.map((spec) => ({ id: spec.id, kind: spec.kind,
    acts: spec.acts.map((act) => ({ type: act.type, to: act.to ? act.to.id : null })) }));
  return { sentence, setup, things, problems };
});
const palettes = { day: api.skyPalette({ dark: false }, "#8b68c2"), night: api.skyPalette({ dark: true }, "#8b68c2") };
console.log(JSON.stringify({ read, palettes, brightness: api.brightness("#8b68c2") }));
"""


@pytest.fixture(scope="module")
def reader(tmp_path_factory):
    if shutil.which("node") is None:
        pytest.skip("node is needed to run the kit's reader")
    script = tmp_path_factory.mktemp("scene3d") / "reader.js"
    script.write_text(READER, encoding="utf-8")

    def read(*sentences: str) -> dict:
        done = subprocess.run(["node", str(script), str(HTML), *sentences],
                              capture_output=True, text=True, check=True)
        return json.loads(done.stdout)

    return read


def _first_act(reader, sentence):
    """The acts of the first thing a sentence names that has any, as (type, target id) pairs."""
    result = reader(sentence)["read"][0]
    thing = next(item for item in result["things"] if item["acts"])
    return [(act["type"], act["to"]) for act in thing["acts"]]


# --- suggest() ----------------------------------------------------------------------------

def test_the_request_gets_the_kit():
    found = components.suggest(REQUEST)
    assert found is not None and found.name == "web/scene-3d"
    assert components.called_for(components.default_library().get("web/scene-3d"), REQUEST)


@pytest.mark.parametrize(("message", "kit"), [
    ("make a bouncing balls animation", "web/animation-2d"),
    ("make a 3d bar chart of my sales", "web/chart-3d"),
    ("make a starfield of particles", "web/particles-3d"),
    ("make a turning 3d cube", "web/object-3d"),
])
def test_other_requests_keep_their_kits(message, kit):
    found = components.suggest(message)
    assert found is not None and found.name == kit


# --- the scene reader ---------------------------------------------------------------------

@pytest.mark.parametrize(("sentence", "kind"), [
    ("kiwi walks towards tree", "walk"),
    ("kiwi walking to tree", "walk"),
    ("kiwi is walking to tree", "walk"),
    ("kiwi walks to tree", "walk"),
    ("kiwi runs to tree", "run"),
    ("bird flies to tree", "fly"),
    ("car drives to tree", "glide"),
])
def test_walk_words_mean_walks_to(reader, sentence, kind):
    assert _first_act(reader, sentence)[0] == (kind, "tree")


def test_walking_towards_and_hugging_it(reader):
    assert _first_act(reader, "a kiwi walking towards a tree and hugging it") == [
        ("walk", "tree"), ("hug", "tree")]


@pytest.mark.parametrize("sentence", ["kiwi dances", "a kiwi dancing", "kiwi dances for 3 seconds"])
def test_dance_is_a_subject_action(reader, sentence):
    assert _first_act(reader, sentence) == [("dance", None)]


@pytest.mark.parametrize("sentence", ["kiwi hugs tree", "kiwi walks to tree and hugs it",
                                      "kiwi walks to tree and hugging it"])
def test_hug_walks_up_then_presses(reader, sentence):
    acts = _first_act(reader, sentence)
    assert acts[-1] == ("hug", "tree")
    assert acts[-2][0] == "walk"


def test_dog_walks_to_house_and_stops_still_reads(reader):
    assert _first_act(reader, "dog walks to house and stops") == [("walk", "house"), ("stop", None)]


@pytest.mark.parametrize(("sentence", "hue", "sky"), [
    ("sky purple", "#8b68c2", "day"),
    ("a purple sky", "#8b68c2", "day"),
    ("sky #6b3fa0", "#6b3fa0", "day"),
    ("sky pink sunset", "#e690b2", "sunset"),
])
def test_sky_colour_words_and_hex(reader, sentence, hue, sky):
    setup = reader(sentence)["read"][0]["setup"]
    assert setup["skyHue"] == hue and setup["sky"] == sky


def test_plain_sky_words_still_set_the_sky(reader):
    setup = reader("sky night")["read"][0]["setup"]
    assert setup["sky"] == "night" and setup["skyHue"] is None


def test_a_purple_sky_is_purple_and_lighter_at_the_horizon(reader):
    def channels(hex_):
        return [int(hex_[at:at + 2], 16) for at in (1, 3, 5)]

    palettes = reader("sky purple")["palettes"]
    red, green, blue = channels(palettes["day"]["top"])
    assert blue > green and red > green  # the top is a purple, not a grey
    assert sum(channels(palettes["day"]["low"])) > sum(channels(palettes["day"]["top"]))  # paler at the horizon
    assert sum(channels(palettes["night"]["top"])) < sum(channels(palettes["day"]["top"]))  # night stays dark
