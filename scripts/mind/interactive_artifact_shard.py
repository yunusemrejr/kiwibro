#!/usr/bin/env python3
"""Build a training-only HTML interaction shard using local, offline Chrome.

Run from the repository root:
    python -m scripts.mind.interactive_artifact_shard /tmp/kiwi-interactive-artifacts

Keep the images/ tree with train.jsonl, or copy it alongside a merged training
file. This generator reads no held-out material and leaves source pages in scratch.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from unittest.mock import patch

from kiwibro import pagecheck, tools
from kiwibro.browser_render import _page_socket
from kiwibro.sandbox import Workspace
from scripts.mind import agent_corpus as corpus

WIDTH, HEIGHT = pagecheck.PHONE
CHROME_FLAGS = (
    "--headless=new", "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
    "--disable-extensions", "--disable-sync", "--disable-quic", "--no-pings",
    "--host-resolver-rules=MAP * ~NOTFOUND", "--proxy-server=http://127.0.0.1:9",
    "--proxy-bypass-list=<-loopback>", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
)


@dataclass(frozen=True)
class Case:
    slug: str
    title: str
    code_or_axis: str
    initial: str
    activated: str
    button: str
    kind: str
    request: str

    @property
    def path(self) -> str:
        return f"{self.slug}.html"

    @property
    def ids(self) -> dict[str, str]:
        return {part: f"{self.slug}-{part}" for part in ("control", "signal", "support", "moving", "extra")}

    @property
    def aria(self) -> str:
        return "aria-expanded" if self.kind == "tilt" else "aria-pressed"


DASHBOARDS = (
    Case("crew-card", "Rail crew desk", "RLY-731", "Crew call due", "Crew call logged", "Log crew call", "build",
         "Make an offline shift card from the attached panel. Copy the lane and cycle accurately."),
    Case("intake-card", "Workshop intake", "WKS-064", "Sample awaiting intake", "Sample accepted", "Accept sample",
         "build", "Turn this workstation capture into a single local intake page. Keep the pictured lane and cycle."),
    Case("parcel-card", "Parcel staging", "PKG-208", "Cage awaiting scan", "Cage scanned", "Record scan", "build",
         "Save a compact parcel checklist from the attached snapshot. Include the pictured lane and cycle."),
    Case("print-card", "Label proofing", "LPR-451", "Proof outstanding", "Proof recorded", "Record proof", "build",
         "Convert this status screenshot to a phone-sized proof card. Use its shown lane and cycle."),
    Case("freeze-card", "Cold room intake", "ICE-590", "Seal inspection due", "Seal cleared", "Clear seal", "repair",
         "The attached image is the source for the cold room card. Repair its initial live status in the existing file."),
    Case("stock-card", "Parts inventory", "INV-327", "Bin audit due", "Bin verified", "Verify bin", "repair",
         "The switch works, but the parts page announces the wrong startup text. Fix it using the attached image."),
    Case("fibre-card", "Fibre splice bench", "FBR-803", "Test pending", "Test signed", "Sign test", "repair",
         "Use the fibre bench snapshot to correct the existing page's status when it first loads."),
    Case("kit-card", "Tool kit checkout", "KIT-246", "Return date needed", "Return date entered", "Record return", "repair",
         "Repair the checkout card from the supplied image. The live region should announce the exact initial signal."),
)
MECHANISMS = (
    Case("rotary-relay", "Relay selector", "Y", "Contacts parked", "Contacts selected", "Turn selector", "rotary",
         "Build a small offline 3D relay model whose contact block turns about a vertical pin. Keep its mount still."),
    Case("rotary-station", "Calibration turntable", "Y", "Face forward", "Side exposed", "Rotate carrier", "rotary",
         "Create a CSS perspective view of a bench turntable. Its carrier rotates around a vertical axis on a fixed plate."),
    Case("rotary-jig", "Optical alignment jig", "Y", "Guide aligned", "Guide turned", "Turn guide", "rotary",
         "Illustrate a bench-top alignment jig. Use real 3D rotation for its adjustable guide and leave the support fixed."),
    Case("rotary-holder", "Sample holder", "Y", "Holder straight", "Holder indexed", "Index holder", "rotary",
         "Show a sample holder in perspective with two stable orientations. The sample rotates, and the base stays put."),
    Case("fold-lamp", "Bench lamp reflector", "X", "Reflector folded", "Reflector raised", "Raise reflector", "tilt",
         "Make a tiny 3D lamp demonstrator. The reflector pivots on its bottom edge above a stationary mount."),
    Case("fold-guard", "Roller guard", "X", "Guard down", "Guard raised", "Raise guard", "tilt",
         "Build an offline roller guard demo with a real 3D bottom pivot. Opening it reveals a short description."),
    Case("fold-mirror", "Alignment mirror", "X", "Mirror stowed", "Mirror tilted", "Tilt mirror", "tilt",
         "Create an offline inset mirror model with a tilt control. Tilt the mirror around its lower edge in perspective."),
    Case("fold-brace", "Service brace", "X", "Brace parked", "Brace raised", "Raise brace", "tilt",
         "Illustrate a brace standing up from a fixed support. The two poses should show actual depth as the brace tilts."),
)

BASE_STYLE = """
*{box-sizing:border-box}body{margin:0;background:#17242d;color:#eef4f8;font:16px system-ui}
main{max-width:44rem;margin:auto;padding:1rem}h1{font-size:1.35rem}
section{background:#243640;border:1px solid #80919d;padding:1rem;border-radius:.6rem}
button{min-height:44px;min-width:44px;padding:.5rem .85rem;border:1px solid #bac8cd;
border-radius:.35rem;background:#edf2f4;color:#192832;font:inherit}
button:focus-visible{outline:3px solid #ffd17f;outline-offset:3px}
[role=status]{font-weight:700;overflow-wrap:anywhere}small{display:block;margin-top:.7rem}
"""
MECHANISM_STYLE = """
.stage{position:relative;min-height:215px;perspective:560px;isolation:isolate}
.base{position:absolute;left:15%;right:15%;height:98px;bottom:15px;transform:rotateX(38deg);
border-radius:10px;border:4px solid #718391;background:
radial-gradient(circle at 9% 17%,#20303a 0 4px,transparent 5px),
radial-gradient(circle at 91% 17%,#20303a 0 4px,transparent 5px),
radial-gradient(circle at 9% 83%,#20303a 0 4px,transparent 5px),
radial-gradient(circle at 91% 83%,#20303a 0 4px,transparent 5px),#a4b7c2;
box-shadow:0 11px #455967,0 16px 14px #101a21}
.part{position:absolute;left:26%;right:26%;height:104px;bottom:80px;
transform-style:preserve-3d;border:4px solid #97aab6;border-radius:8px;background:
repeating-linear-gradient(90deg,#526978 0 10px,#5b7280 10px 14px);
transition:transform 170ms ease,background-color 170ms ease}
.part:before{content:"";position:absolute;inset:8px;background:#293a45;border:2px solid #78909e;
border-radius:4px;transform:translateZ(5px)}
.part:after{content:"";position:absolute;left:25%;right:25%;top:35%;bottom:35%;
background:#9aadb8;border-radius:8px;transform:translateZ(9px)}
.stage[data-active=true] .part:after{background:#ffd17f;box-shadow:0 0 10px #ffd17f88}
@media(prefers-reduced-motion:reduce){.part{transition:none}}
"""
WRONG_SIGNAL = "Slate indicates a pending item."


def html(case: Case, *, broken: bool = False) -> str:
    ids = case.ids
    mechanism = case.kind in ("tilt", "rotary")
    controlled = ids["extra"] if case.kind == "tilt" else ids["signal"]
    initial = WRONG_SIGNAL if broken else case.initial
    detail = (f'<p id="{ids["extra"]}" hidden>The raised panel exposes its support.</p>'
              if case.kind == "tilt" else "")
    if mechanism:
        pivot = "50% 100%" if case.kind == "tilt" else "50% 50%"
        angle = -49 if case.kind == "tilt" else 56
        style = BASE_STYLE + MECHANISM_STYLE + (
            f".part{{transform-origin:{pivot};transform:rotate{case.code_or_axis}(0deg)}}"
            f".stage[data-active=true] .part{{transform:rotate{case.code_or_axis}({angle}deg)}}"
        )
        body = (f'<div class="stage" data-active="false" aria-hidden="true">'
                f'<div class="base" id="{ids["support"]}"></div><div class="part" id="{ids["moving"]}"></div></div>')
        note = ""
    else:
        style = BASE_STYLE
        body = f"<p>Cycle <strong>{escape(case.code_or_axis)}</strong></p>"
        note = f"<small>{WRONG_SIGNAL}</small>" if case.kind == "repair" else ""
    update_pose = 'document.querySelector(".stage").dataset.active = String(active);' if mechanism else ""
    update_detail = f'document.getElementById("{ids["extra"]}").hidden = !active;' if detail else ""
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(case.title)}</title>
<style>{style}</style>
<main><h1>{escape(case.title)}</h1><section>
{body}
<p id="{ids["signal"]}" role="status">{escape(initial)}</p>
<button type="button" id="{ids["control"]}" {case.aria}="false" aria-controls="{controlled}">{escape(case.button)}</button>
{detail}
{note}
</section></main>
<script>
const button = document.getElementById("{ids["control"]}");
const status = document.getElementById("{ids["signal"]}");
const initialText = status.textContent;
button.addEventListener("click", () => {{
  const active = button.getAttribute("{case.aria}") === "false";
  button.setAttribute("{case.aria}", String(active));
  status.textContent = active ? {json.dumps(case.activated)} : initialText;
  {update_pose}
  {update_detail}
}});
</script>
</html>
"""


def observe(episode: corpus.Episode, context: tools.ToolContext, tool: str, **args) -> tools.ToolResult:
    result = tools.execute_tool(tool, tools.validate_args(tool, args), context)
    if not result.ok:
        raise RuntimeError(f"{episode.id}: {result.content}")
    episode.observe(tool, result.content, **args)
    return result


def make_episode(case: Case, folder: Path, out: Path) -> corpus.Episode:
    mechanism = case.kind in ("tilt", "rotary")
    repair = case.kind == "repair"
    family = "interactive_mechanism" if mechanism else "interactive_dashboard"
    names = ("read_file", "apply_patch", "look_at_page") if repair else ("write_file", "look_at_page")
    caption = (
        f'Save {case.path}. Name the native button "{case.button}" with ID {case.ids["control"]}. '
        f'Put the initial text in the visible status node {case.ids["signal"]}, and '
        f'set {case.aria}="false" on the button. Activation must set it true and show "{case.activated}". '
        "A second activation restores the initial state. Keep the button focused, fit a 360 px phone width, "
        "and check the saved render. Use inline styles and JavaScript only."
    )
    if mechanism:
        caption += (f' Start with "{case.initial}". Preserve the initial pose until activation, '
                    "and honor reduced motion.")
    elif repair:
        caption += " Keep the existing color note outside the live status region and make no unrelated edits."
    episode = corpus.Episode(f"ia-{case.slug}", family, f"{case.request} {caption}", names)
    if not mechanism:
        corpus.draw_panel(out, episode, case.title,
                          [("Lane", case.title), ("Cycle", case.code_or_axis), ("Signal", case.initial)],
                          "Operator view")
    context = tools.ToolContext(Workspace(folder), None, None, episode.id, look=pagecheck.Options(phone=True))
    if repair:
        context.workspace.write_file(case.path, html(case, broken=True))
        observe(episode, context, "read_file", path=case.path)
        node = f'<p id="{case.ids["signal"]}" role="status">'
        old, new = (f"{node}{escape(value)}</p>" for value in (WRONG_SIGNAL, case.initial))
        observe(episode, context, "apply_patch", path=case.path, edits=[{"old": old, "new": new}])
    else:
        observe(episode, context, "write_file", path=case.path, content=html(case))
    if (folder / case.path).read_text(encoding="utf-8") != html(case):
        raise RuntimeError(f"{episode.id}: file did not match the authored target")
    render = observe(episode, context, "look_at_page", path=case.path).meta["look"]
    checked = {"it loads without errors", "it draws", "it fits a phone screen"}
    if render["findings"] or render["could_not"] or not checked <= set(render["confirmed"]):
        raise RuntimeError(f"{episode.id}: render check failed: {render}")
    return episode.answer(
        f'Saved and checked `{case.path}` at phone width. The status starts as "{case.initial}". '
        f'Activating {case.button} changes it to "{case.activated}".',
        (case.path, case.initial, case.activated),
    )


@contextmanager
def chrome_session(chrome: str, scratch: Path):
    import websocket

    profile = scratch / "cdp-profile"
    with subprocess.Popen(
        [chrome, *CHROME_FLAGS, f"--user-data-dir={profile}", "--remote-debugging-port=0", "about:blank"],
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) as process:
        socket = _page_socket(profile, process, time.monotonic() + 12)
        try:
            if socket is None:
                raise RuntimeError("Chrome did not start its private control endpoint")
            sequence = 0

            def command(method: str, **params) -> dict:
                nonlocal sequence
                sequence += 1
                socket.send(json.dumps({"id": sequence, "method": method, "params": params}))
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    try:
                        event = json.loads(socket.recv())
                    except websocket.WebSocketTimeoutException:
                        continue
                    if event.get("method") == "Runtime.exceptionThrown":
                        raise RuntimeError("The authored page threw a browser exception")
                    if event.get("method") == "Network.requestWillBeSent":
                        url = event["params"]["request"]["url"]
                        if url.startswith(("http://", "https://", "ws://", "wss://")):
                            raise RuntimeError("An offline target requested an external resource")
                    if event.get("id") == sequence:
                        if "error" in event:
                            raise RuntimeError(f"Chrome {method}: {event['error']}")
                        return event["result"]
                raise RuntimeError(f"Chrome timed out on {method}")

            command("Runtime.enable")
            command("Network.enable")
            command("Network.setBlockedURLs", urls=["http://*", "https://*", "ws://*", "wss://*"])
            command("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=True)
            command("Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": "reduce"}])
            yield command
        finally:
            if socket is not None:
                socket.close(timeout=0)
            pagecheck._quit(process)
            process.wait(timeout=3)


def browser_verify(command, case: Case, path: Path, *, broken: bool = False) -> dict:
    ids = case.ids

    def evaluate(expression: str, *, wait: bool = False):
        result = command("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=wait)
        if "exceptionDetails" in result:
            raise RuntimeError(f"{case.slug}: browser probe failed")
        return result["result"].get("value")

    def key(value: str, code: str, number: int) -> None:
        args = {"key": value, "code": code, "windowsVirtualKeyCode": number}
        command("Input.dispatchKeyEvent", type="rawKeyDown", **args)
        command("Input.dispatchKeyEvent", type="keyUp", **args)

    def snapshot() -> dict:
        return evaluate(f"""(() => {{
  const ids = {json.dumps(ids)};
  const byPart = name => document.getElementById(ids[name]);
  const box = node => node?.getBoundingClientRect().toJSON();
  const button = byPart("control"), status = byPart("signal"), moving = byPart("moving");
  const styles = getComputedStyle(status);
  const partStyle = moving && getComputedStyle(moving);
  const pose = partStyle && new DOMMatrix(partStyle.transform);
  return {{
    width: innerWidth, scroll: document.documentElement.scrollWidth, button: box(button),
    focus: document.activeElement.id, name: button.innerText, native: button.tagName === "BUTTON",
    unique: Object.values(ids).every(id => document.querySelectorAll("#" + id).length <= 1),
    role: status.getAttribute("role"), signal: status.textContent, visible: Boolean(status.getClientRects().length) &&
      styles.visibility === "visible" && styles.opacity !== "0",
    aria: button.getAttribute("{case.aria}"), controlled: button.getAttribute("aria-controls"),
    extraHidden: byPart("extra")?.hidden, poseX: pose?.m23, poseY: pose?.m13,
    perspective: moving && getComputedStyle(moving.parentElement).perspective !== "none",
    transition: partStyle?.transitionDuration,
    originY: partStyle && parseFloat(partStyle.transformOrigin.split(" ")[1]),
    partHeight: moving?.offsetHeight,
    moving: box(moving), base: box(byPart("support")), text: document.body.innerText
  }};
}})()""")

    def require(condition: bool, name: str) -> None:
        if not condition:
            raise RuntimeError(f"{case.slug}: browser did not verify {name}")

    url = path.resolve().as_uri()
    require(not command("Page.navigate", url=url).get("errorText"), "local page loading")
    deadline = time.monotonic() + 5
    while not evaluate(f"location.href === {json.dumps(url)} && document.readyState === 'complete'"):
        require(time.monotonic() < deadline, "document readiness")
        time.sleep(.02)
    initial = snapshot()
    bounds = initial["button"]
    expected_text = WRONG_SIGNAL if broken else case.initial
    control = ids["extra"] if case.kind == "tilt" else ids["signal"]
    require(initial["unique"] and initial["role"] == "status" and initial["visible"] and
            initial["signal"] == expected_text and initial["aria"] == "false" and initial["controlled"] == control,
            "exact visible initial live text and ARIA")
    require(initial["native"] and initial["name"] == case.button and min(bounds["width"], bounds["height"]) >= 44,
            "the named native control and touch target")
    require(initial["width"] == WIDTH and initial["scroll"] <= WIDTH and
            0 <= bounds["left"] < bounds["right"] <= WIDTH and 0 <= bounds["top"] < bounds["bottom"] <= HEIGHT,
            "360 px layout and on-screen control")
    if case.kind in ("build", "repair"):
        require(case.title in initial["text"] and case.code_or_axis in initial["text"], "exact screenshot labels")
    if case.kind == "repair":
        require(WRONG_SIGNAL in initial["text"], "the preserved color note")
    if broken:
        require(initial["signal"] != case.initial, "the original wrong live text")
        return {}
    key("Tab", "Tab", 9)
    focused = snapshot()
    require(focused["focus"] == ids["control"] and focused["signal"] == expected_text, "keyboard focus")
    key(" ", "Space", 32)
    evaluate("new Promise(resolve => requestAnimationFrame(resolve))", wait=True)
    activated = snapshot()
    require(activated["aria"] == "true" and activated["signal"] == case.activated and
            activated["visible"] and activated["focus"] == ids["control"], "keyboard activation and visible update")
    bounds = activated["button"]
    x = (bounds["left"] + bounds["right"]) / 2
    y = (bounds["top"] + bounds["bottom"]) / 2
    for action in ("mousePressed", "mouseReleased"):
        command("Input.dispatchMouseEvent", type=action, x=x, y=y, button="left", clickCount=1)
    evaluate("new Promise(resolve => requestAnimationFrame(resolve))", wait=True)
    restored = snapshot()
    require(restored["aria"] == "false" and restored["signal"] == expected_text and
            restored["focus"] == ids["control"], "pointer activation and restoration")
    if case.kind == "tilt":
        require(initial["extraHidden"] and not activated["extraHidden"] and restored["extraHidden"],
                "visible content controlled by aria-expanded")
    if case.kind in ("tilt", "rotary"):
        axis = f"pose{case.code_or_axis}"
        expected_origin = initial["partHeight"] / (1 if case.kind == "tilt" else 2)
        require(initial["perspective"] and initial["transition"] == "0s" and
                abs(initial["originY"] - expected_origin) < .5 and
                abs(initial[axis] - activated[axis]) > .5 and abs(initial[axis] - restored[axis]) < .01 and
                abs(activated["moving"]["width"] - initial["moving"]["width"]) > 4 and
                initial["base"] == activated["base"] == restored["base"], "3D pose, fixed base, and reduced motion")
        return {"axis": case.code_or_axis, "rest_active_restored": [
                    round(state[axis], 4) for state in (initial, activated, restored)],
                "origin_y_px": initial["originY"], "fixed_base": True, "reduced_motion": True}
    return {}


def build(out: Path) -> dict:
    if any((out / name).exists() for name in ("heldout.jsonl", "probes.jsonl")):
        raise ValueError("Use a separate output directory for this training shard")
    chrome = pagecheck.find_chrome()
    if not chrome:
        raise RuntimeError("Chrome is required to verify the authored training targets")
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = (*DASHBOARDS, *MECHANISMS)
    catalog = {item["function"]["name"]: item for item in tools.tool_defs()}
    poses = {}
    with tempfile.TemporaryDirectory(prefix=".interactive-shard-", dir=out.parent) as temporary:
        scratch = Path(temporary)
        export = scratch / "export"
        export.mkdir()
        wrapper = scratch / "offline-chrome"
        wrapper.write_text(f'#!/bin/sh\nexec {shlex.join([chrome, *CHROME_FLAGS])} "$@"\n', encoding="utf-8")
        wrapper.chmod(0o700)
        with patch.object(pagecheck, "find_chrome", return_value=str(wrapper)):
            episodes = [make_episode(case, scratch / case.slug, export) for case in cases]
        with chrome_session(chrome, scratch) as command:
            for case in cases:
                page = scratch / case.slug / case.path
                if case.kind == "repair":
                    broken = page.with_name("initial-bug.html")
                    broken.write_text(html(case, broken=True), encoding="utf-8")
                    browser_verify(command, case, broken, broken=True)
                if pose := browser_verify(command, case, page):
                    poses[case.slug] = pose
        training = export / "train.jsonl"
        corpus.jsonl(training, [episode.record(catalog) for episode in episodes])
        summary = corpus.check(export, training_only=True)
        summary["chrome_verified"] = {"initial_and_keyboard_and_click": len(cases),
                                     "original_failures": sum(case.kind == "repair" for case in cases),
                                     "reduced_motion_and_3d": len(poses), "width_px": WIDTH, "poses": poses}
        files = ["train.jsonl", *(image for episode in episodes for image in episode.images)]
        summary["sha256"] = {name: sha256((export / name).read_bytes()).hexdigest() for name in files}
        (export / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name in [*files, "manifest.json"]:
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            (export / name).replace(target)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    print(json.dumps(build(parser.parse_args().out), indent=2))


if __name__ == "__main__":
    main()
