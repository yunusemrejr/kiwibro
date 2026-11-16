"""Looking at a page: what a page's problems are called, with a stand-in for Chrome, and once with the real one."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from kiwibro import pagecheck, pageloop, prompts
from kiwibro.pagecheck import Options, Render
from kiwibro.sandbox import SandboxError, Workspace

CLEAN = {"e": [], "f": [], "c": [], "raf": 30, "an": 0, "text": "Bouncing ball", "tl": 13, "cv": 1, "md": 0, "el": 3}


def write_png(path: Path, mark: int) -> None:
    """A small valid picture; two different marks give two different pictures. Needs no imaging library."""
    rows = b"".join(b"\x00" + bytes(([40, 90, 140] * 16)) for _ in range(16))
    rows = bytearray(rows)
    rows[1 + (mark % 16) * 3] = 250  # one bright pixel on the first row
    raw = bytes(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class Fake:
    """Stands in for Chrome. ``reports`` holds one probe report per look at desktop width (the last one repeats);
    a page that ``moves`` draws a different picture at a different time."""

    def __init__(self, *reports: dict[str, Any], moves: bool = True, error: str = "") -> None:
        self.reports = list(reports) or [dict(CLEAN)]
        self.moves = moves
        self.error = error
        self.looks = -1
        self.sizes: list[tuple[int, int, int]] = []

    def __call__(self, page: Path, png: Path, width: int, height: int, budget_ms: int) -> Render:
        self.sizes.append((width, height, budget_ms))
        if self.error:
            return Render(error=self.error)
        if width == pagecheck.DESKTOP[0] and budget_ms == pagecheck.BUDGET_MS:
            self.looks += 1
        report = dict(self.reports[min(max(self.looks, 0), len(self.reports) - 1)])
        report.setdefault("w", width)
        report.setdefault("sw", width)
        if "motion: { planned:" in page.read_text(encoding="utf-8"):
            report.setdefault("motion", {"planned": True, "rendered": 30, "observed": self.moves, "waiting": False})
        write_png(png, budget_ms // 100 if self.moves else 0)
        return Render(report=report)

    @property
    def widths(self) -> list[int]:
        return [size[0] for size in self.sizes]


def script_error(line: int = 2) -> dict[str, Any]:
    return {**CLEAN, "e": [{"m": "Uncaught ReferenceError: spin is not defined", "l": line}]}


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "index.html").write_text("<!doctype html><html><body><canvas></canvas></body></html>\n", encoding="utf-8")
    return Workspace(tmp_path)


def look(workspace: Workspace, fake: Fake, options: Options | None = None, **kwargs: Any) -> pagecheck.Look:
    return pagecheck.look(workspace, "index.html", options, render=fake, **kwargs)


def test_a_script_error_is_named_with_the_line_it_is_on(workspace: Workspace) -> None:
    found = look(workspace, Fake(script_error(2)))
    assert [item.code for item in found.findings] == ["script"]
    assert 'A script error stops the page: "Uncaught ReferenceError: spin is not defined", line 2.' in found.report()
    assert "Fix each of these in index.html." in found.report()
    assert found.picture == "screenshots/look-index.png" and (workspace.root / found.picture).is_file()


def test_a_page_that_loads_draws_and_moves_has_nothing_to_fix(workspace: Workspace) -> None:
    found = look(workspace, Fake(), Options(moves=True))
    assert found.findings == []
    assert found.report().splitlines()[-1] == "No problems found: it loads without errors, it draws and it moves. Nothing more is to be done."
    assert found.report().splitlines()[0] == "Looked at index.html at 1280x800, and again 2 seconds later."


def test_a_page_that_asks_for_motion_and_shows_the_same_picture_is_named(workspace: Workspace) -> None:
    still = {**CLEAN, "raf": 0, "an": 0}
    found = look(workspace, Fake(still, moves=False), Options(moves=True))
    assert [item.code for item in found.findings] == ["still"]
    assert look(workspace, Fake(still, moves=False), Options(moves=False)).findings == []  # not asked: not raised


def test_an_animation_callback_without_visible_motion_does_not_pass(workspace: Workspace) -> None:
    found = look(workspace, Fake(moves=False), Options(moves=True))
    assert [item.code for item in found.findings] == ["still"]
    assert "it moves" not in found.confirmed


def test_a_running_callback_needs_pixels_to_move_in_the_live_check(workspace: Workspace, monkeypatch) -> None:
    class Virtual(Fake):
        def __call__(self, *args) -> Render:
            report = super().__call__(*args).report
            return Render({**report, "live": True}, virtual_time=True)

    def motion(_page, first, later):
        write_png(first, 0)
        write_png(later, 0)
        return True

    monkeypatch.setattr(pagecheck, "_live_motion_frames", motion)
    found = look(workspace, Virtual(moves=False), Options(moves=True))
    assert [item.code for item in found.findings] == ["still"]
    assert "it moves" not in found.confirmed


def test_scene_motion_is_checked_on_the_subject(workspace: Workspace) -> None:
    (workspace.root / "index.html").write_text(
        '<!doctype html><html><body><canvas></canvas>'
        '<script id="scene" type="text/plain">kiwi dances</script>'
        '<script>window.sceneReport={motion: { planned: true }};</script></body></html>',
        encoding="utf-8",
    )
    motion = {"planned": True, "observed": False, "rendered": 30, "waiting": False}
    found = look(workspace, Fake({**CLEAN, "motion": motion}, moves=True), Options(moves=True))
    assert [item.code for item in found.findings] == ["still"]  # a background change is insufficient
    fake = Fake({**CLEAN, "motion": {**motion, "observed": True}})
    found = look(workspace, fake, Options(moves=True))
    assert "it moves" in found.confirmed
    assert fake.widths == [pagecheck.DESKTOP[0]]


def test_text_left_unfilled_is_named(workspace: Workspace) -> None:
    found = look(workspace, Fake({**CLEAN, "text": "Hello {{title}}", "tl": 15}))
    assert [item.code for item in found.findings] == ["unfilled"]
    assert "{{title}}" in found.findings[0].text


def test_a_kit_notice_on_the_page_is_a_problem(workspace: Workspace) -> None:
    shown = {**CLEAN, "text": 'Not understood: "kiwi runs"\nscene', "tl": 30}
    found = look(workspace, Fake(shown), Options(not_text=('Not understood: "',)))
    assert [item.code for item in found.findings] == ["notice"]
    assert look(workspace, Fake(shown)).findings == []


def test_the_phone_look_names_a_page_wider_than_the_screen(workspace: Workspace) -> None:
    class Wide(Fake):
        def __call__(self, page: Path, png: Path, width: int, height: int, budget_ms: int) -> Render:
            done = super().__call__(page, png, width, height, budget_ms)
            if width == pagecheck.PHONE[0] and done.report is not None:
                done.report["sw"] = 600
            return done

    fake = Wide()
    found = look(workspace, fake, Options(phone=True))
    assert pagecheck.PHONE[0] in fake.widths and found.sizes == ("1280x800", "360x740")
    assert [item.code for item in found.findings] == ["wide"]
    assert pagecheck.PHONE[0] not in look(workspace, fake := Fake(), Options(phone=False)).sizes and 360 not in fake.widths


def test_a_failed_phone_view_is_not_confirmed(workspace: Workspace) -> None:
    class MissingPhone(Fake):
        def __call__(self, page: Path, png: Path, width: int, height: int, budget_ms: int) -> Render:
            if width == pagecheck.PHONE[0]:
                return Render(error="Chrome stopped")
            return super().__call__(page, png, width, height, budget_ms)

    found = look(workspace, MissingPhone(), Options(phone=True))
    assert found.sizes == ("1280x800",)
    assert "it fits a phone screen" not in found.confirmed
    assert found.could_not == "the phone viewport could not be checked"


def test_in_goal_mode_the_picture_is_asked_whether_it_shows_the_goal(workspace: Workspace) -> None:
    asked: list[str] = []

    def eyes(picture: str, question: str) -> str:
        asked.append(question)
        return "No. It shows a gray square."

    found = look(workspace, Fake(), Options(goal="a red ball"), eyes=eyes)
    assert "a red ball" in asked[0]
    assert [item.code for item in found.findings] == ["unlike"]
    assert found.findings[0].short == "it does not look like what was asked: It shows a gray square."
    assert found.seen == "It shows a gray square."
    yes = look(workspace, Fake(), Options(goal="a red ball"), eyes=lambda picture, question: "Yes. A red ball on blue.")
    assert yes.findings == [] and "it looks like what was asked" in yes.confirmed


def test_a_model_that_cannot_look_at_the_picture_does_not_fail_the_look(workspace: Workspace) -> None:
    def blind(picture: str, question: str) -> str:
        raise RuntimeError("no vision model")

    assert look(workspace, Fake(), Options(goal="a red ball"), eyes=blind).findings == []


def test_a_page_that_could_not_be_rendered_is_not_called_wrong(workspace: Workspace) -> None:
    found = look(workspace, Fake(error="Chrome did not start"))
    assert found.could_not and found.findings == []
    assert found.report().splitlines() == [
        "Could not look at index.html: Chrome did not start.",
        "The render check did not finish.",
    ]


def test_only_html_files_of_the_workspace_are_looked_at(workspace: Workspace) -> None:
    (workspace.root / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SandboxError):
        pagecheck.look(workspace, "notes.txt", render=Fake())
    with pytest.raises(SandboxError):
        pagecheck.look(workspace, "../outside.html", render=Fake())


def test_a_file_the_page_names_and_that_is_not_there_is_named(workspace: Workspace) -> None:
    (workspace.root / "index.html").write_text(
        '<!doctype html><html><head><script src="app.js"></script></head><body></body></html>\n', encoding="utf-8")
    found = look(workspace, Fake())
    assert any(item.code == "missing" and "app.js" in item.text for item in found.findings)


def test_a_goal_message_is_read_by_its_first_word() -> None:
    assert pageloop.parse_goal("/goal a page with a bouncing ball") == (True, "a page with a bouncing ball")
    assert pageloop.parse_goal("/GOAL   a page\nwith two lines ") == (True, "a page\nwith two lines")
    assert pageloop.parse_goal("/goal") == (True, "")
    assert pageloop.parse_goal("  /goal  ") == (True, "")
    assert pageloop.parse_goal("/goalpost") == (False, "")
    assert pageloop.parse_goal("make it /goal") == (False, "")


def test_what_kiwibro_says_after_a_look_follows_how_the_problems_changed() -> None:
    error = {"short": "a script error stops the page", "fixed": "the script runs"}
    still = {"short": "nothing moves", "fixed": "it moves now"}
    assert pageloop.verdict(None, [error]) == prompts.LOOK_BAD.format(problems="a script error stops the page")
    assert pageloop.verdict([error], [error]) == prompts.LOOK_SAME.format(problems="a script error stops the page")
    assert pageloop.verdict([error], [still]) == prompts.LOOK_BAD.format(problems="nothing moves")
    assert pageloop.verdict([error, still], [still]) == prompts.LOOK_BETTER.format(fixed="the script runs", problems="nothing moves")
    assert pageloop.verdict([still], [still, error]) == prompts.LOOK_WORSE.format(problems="nothing moves; a script error stops the page")


@pytest.mark.skipif(pagecheck.find_chrome() is None, reason="Chrome is not installed here")
def test_the_real_browser_finds_a_script_error_and_a_page_that_moves(tmp_path: Path, monkeypatch) -> None:
    # Chrome's virtual-time screenshots can be the same compositor frame. Force that first
    # comparison here so the test verifies the real-time fallback on every Chrome version.
    compare = pagecheck._same
    monkeypatch.setattr(pagecheck, "_same", lambda first, later: first.name == "first.png" or compare(first, later))
    (tmp_path / "bad.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"></head><body><canvas id="c"></canvas><script>\n'
        'document.getElementById("c").getContext("2d").fillRect(0,0,10,10); spin();\n</script></body></html>\n',
        encoding="utf-8")
    (tmp_path / "ball.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0"><canvas id="c" width="300" height="200"></canvas><script>\n'
        'const x=document.getElementById("c").getContext("2d");let y=20,v=0;\n'
        'function f(){v+=.4;y+=v;if(y>170){y=170;v=-v*.9}x.fillStyle="#123";x.fillRect(0,0,300,200);x.fillStyle="#fc0";'
        'x.beginPath();x.arc(150,y,20,0,7);x.fill();requestAnimationFrame(f)}f();\n</script></body></html>\n',
        encoding="utf-8")
    workspace = Workspace(tmp_path)
    bad = pagecheck.look(workspace, "bad.html")
    assert [item.code for item in bad.findings] == ["script"] and "spin is not defined" in bad.findings[0].text
    assert "line 2" in bad.findings[0].text  # the page's own line, not one shifted by what the look adds
    good = pagecheck.look(workspace, "ball.html", Options(moves=True))
    assert good.findings == [] and (tmp_path / good.picture).stat().st_size > 200
    assert "it moves" in good.confirmed and not good.could_not


def test_a_request_that_names_movement_in_plain_words_expects_motion():
    for said in ("a page where a red square moves from left to right forever", "a ball that bounces",
                 "a logo sliding in", "make an html page with an animation of a bouncing ball", "a duck dances"):
        assert pagecheck.wants_motion(said), said
    for said in ("a page with my opening hours", "a table of prices", "remove the footer", "a 3D model of a cup"):
        assert not pagecheck.wants_motion(said), said
