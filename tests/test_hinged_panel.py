"""The hinged panel only publishes new pages and preserves its actual pivots."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from kiwibro import checkpoints, components, pagecheck
from kiwibro.sandbox import Workspace
from scripts.mind import hinge_shard
from scripts.mind.interactive_artifact_shard import chrome_session


KIT = components.HINGED_PANEL


@pytest.mark.parametrize(("values", "error"), [
    ({"edge": "center"}, "edge must be"),
    ({"direction": "sideways"}, "direction must be"),
    ({"angle": "55.5"}, "whole number"),
    ({"angle": "81"}, "whole number"),
    ({"filename": "../custom.html"}, "single .html"),
    ({"leaf_id": 'bad" onclick="x'}, "IDs of up to"),
    ({"leaf_id": "frame"}, "Each element ID"),
    ({"title": "multi\nline"}, "single line"),
    ({"css": "body{}"}, "Left out"),
])
def test_invalid_component_values_leave_no_file(tmp_path: Path, values: dict, error: str) -> None:
    with pytest.raises(components.ComponentError, match=error):
        components.use(KIT, values, workspace=Workspace(tmp_path))
    assert not list(tmp_path.iterdir())


def test_kit_escapes_text_and_undoes_the_successful_write(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    data = tmp_path / "data"
    label = '"</script><img src=/probe>'
    reply = components.use(
        KIT, {"title": label, "status_id": "hint", "open_label": label, "filename": "motion.htm"},
        workspace=workspace, path="demo", data_dir=data, conversation_id="test-hinge",
    )
    page = workspace.root / "demo/motion.htm"
    assert "demo/motion.htm" in reply
    source = page.read_text(encoding="utf-8")
    assert source.count("</script>") == 1
    assert source.count("&quot;&lt;/script&gt;&lt;img src=/probe&gt;") == 3
    assert 'document.getElementById("hint")' in source
    with pytest.raises(components.ComponentError, match="already exists"):
        components.use(KIT, {"filename": "motion.htm"}, workspace=workspace, path="demo",
                       replace={"demo/motion.htm"})
    assert page.read_text(encoding="utf-8") == source
    restored, = checkpoints.store_for(data).undo("test-hinge")
    assert restored["path"] == "demo/motion.htm"
    assert restored["action"] == "removed (it did not exist before)"
    assert not page.exists()


def test_concurrent_creator_wins_without_a_checkpoint_or_partial_artifact(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(tmp_path / "workspace")
    data = tmp_path / "data"
    real_link = os.link

    def competing_link(source, name, **kwargs):
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=kwargs["dst_dir_fd"])
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write("different creator")
        return real_link(source, name, **kwargs)

    monkeypatch.setattr(components.os, "link", competing_link)
    with pytest.raises(components.ComponentError, match="already exists"):
        components.use(KIT, workspace=workspace, data_dir=data, conversation_id="test-hinge")
    assert [entry.name for entry in workspace.root.iterdir()] == ["index.html"]
    assert (workspace.root / "index.html").read_text(encoding="utf-8") == "different creator"
    assert checkpoints.store_for(data).history("test-hinge") == []


def test_a_symbolic_destination_or_folder_is_never_followed(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace.root / "index.html").symlink_to(workspace.root / "missing.html")
    (workspace.root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(components.ComponentError, match="already exists"):
        components.use(KIT, workspace=workspace)
    with pytest.raises(components.ComponentError, match="Could not create"):
        components.use(KIT, workspace=workspace, path="redirect")
    with pytest.raises(components.ComponentError, match="relative to the workspace"):
        components.use(KIT, workspace=workspace, path="../outside")
    assert not (workspace.root / "missing.html").exists()
    assert not list(outside.iterdir())


def test_real_panel_has_fixed_anchors_and_accessible_pose_controls(tmp_path: Path) -> None:
    chrome = pagecheck.find_chrome()
    if not chrome:
        pytest.skip("Chrome is required to measure the hinge geometry")
    pytest.importorskip("websocket")
    workspace = Workspace(tmp_path / "workspace")
    # Reuse the authored geometry oracle; build each page through the actual kit.
    with chrome_session(chrome, tmp_path) as run:
        for case in hinge_shard.TRAIN[:4]:
            values = {
                "filename": case.filename,
                "edge": case.edge,
                "button_label": case.button,
                "closed_label": case.closed,
                "open_label": case.opened,
                "detail_id": case.ids["inside"],
                **{f"{key}_id": case.ids[key] for key in ("leaf", "frame", "toggle", "status")},
            }
            panel = workspace.root / case.filename
            components.use(KIT, values, workspace=workspace)
            for size in hinge_shard.VIEWPORTS:
                pass_result = hinge_shard.inspect(run, case, panel, size)
                assert pass_result["frame_fixed"]
            # Compare the real transform matrix for the reverse swing at the same hinge.
            away = "away-" + case.filename
            components.use(KIT, {**values, "filename": away, "direction": "away", "angle": "63"},
                           workspace=workspace)
            panel_uri = (workspace.root / away).as_uri()
            assert not run("Page.navigate", url=panel_uri).get("errorText")
            result = run(
                "Runtime.evaluate",
                expression="""new Promise(resolve => {
                  const paint = () => requestAnimationFrame(() => {
                    document.querySelector("button").click();
                    const style = getComputedStyle(document.getElementById(%r));
                    const matrix = new DOMMatrix(style.transform);
                    resolve([matrix.m13, matrix.m23]);
                  });
                  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", paint, {once:true});
                  else paint();
                })""" % case.ids["leaf"],
                awaitPromise=True, returnByValue=True,
            )
            assert "exceptionDetails" not in result
            y_depth, x_depth = result["result"]["value"]
            depth, off_axis = (x_depth, y_depth) if case.edge in ("top", "bottom") else (y_depth, x_depth)
            sign = -1 if case.edge in ("top", "left") else 1
            assert depth == pytest.approx(sign * math.sin(math.radians(63)), abs=.001)
            assert off_axis == pytest.approx(0, abs=.001)
