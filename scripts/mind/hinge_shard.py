#!/usr/bin/env python3
"""Build edge-hinge first-write data and check separate held-out HTML outcomes.

From the KiwiBro checkout:
    python -m scripts.mind.hinge_shard prepare OUT
    python -m scripts.mind.hinge_shard verify OUT
    python -m scripts.mind.hinge_shard check CASE_ID GENERATED_HTML

Only train/train.jsonl enters training. The held-out prompts and reference HTML
stay in heldout/. A browser receipt verifies the saved examples after preparation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import tempfile
import time

from kiwibro import pagecheck, tools
from kiwibro.sandbox import Workspace
from scripts.mind import agent_corpus as corpus
from scripts.mind.interactive_artifact_shard import BASE_STYLE, chrome_session, observe

EDGES = ("top", "bottom", "left", "right")
POSE = {"top": ("50% 0%", "X", 56), "bottom": ("50% 100%", "X", -57),
        "left": ("0% 50%", "Y", -54), "right": ("100% 50%", "Y", 60)}
VIEWPORTS = ((360, 740), (800, 760))


@dataclass(frozen=True)
class Case:
    family: str
    edge: str
    slug: str
    title: str
    part: str
    reveal: str
    brief: str
    paint: str

    @property
    def id(self):
        return f"edge-{self.slug}"

    @property
    def filename(self):
        return f"{self.slug}-motion.html"

    @property
    def button(self):
        return f"Move {self.part}"

    @property
    def closed(self):
        return f"{self.part.capitalize()} seated."

    @property
    def opened(self):
        return f"{self.reveal.capitalize()} visible."

    @property
    def ids(self):
        return {key: f"{self.slug}-{key}" for key in ("leaf", "frame", "toggle", "status", "inside")}

    def request(self):
        return (
            f"{self.brief} Save a self-contained demonstration as {self.filename}. "
            f"The {self.part} hinges at its {self.edge} edge on a stationary frame. "
            "Swing its free edge toward the viewer and keep both ends of the hinge fixed. "
            f"Use IDs {self.ids['leaf']} for the moving part, {self.ids['frame']} for the fixed frame, "
            f"and {self.ids['toggle']} for the native button named “{self.button}”. "
            f"Use aria-expanded and aria-controls to toggle the hidden detail {self.ids['inside']}. "
            f"Show “{self.closed}” then “{self.opened}” in the visible status {self.ids['status']}. "
            "Click or press Space again to restore the closed pose. Keep focus on the control, "
            "work at 360 px and desktop width, and make the pose switch immediately under reduced motion."
        )


def group(family, paint, rows):
    return tuple(Case(family, edge, *row, paint) for edge, row in zip(EDGES, rows, strict=True))


TRAIN = (
    *group("ventilation", "#879ca1", (
        ("intake-louver", "Intake louver", "louver", "intake mesh", "Explain how a pinned intake blade can expose the grille behind it."),
        ("plenum-access", "Plenum access", "access plate", "plenum lining", "Make a visual maintenance card for an underfloor plenum."),
        ("filter-grille", "Filter grille", "grille", "filter cartridge", "Show an intake grille opening for a cartridge change."),
        ("fan-shield", "Fan shield", "shield", "stationary fan guard", "Illustrate service access on a fictional desk fan."),
    )),
    *group("instrumentation", "#a1a684", (
        ("meter-shade", "Meter shade", "shade", "meter face", "Demonstrate how to lift the glare shade on a bench meter."),
        ("probe-rack", "Probe rack", "retainer", "probe slots", "Show a probe rack retainer opening to expose the empty slots."),
        ("dial-window", "Dial window", "window frame", "dial mount", "Make a miniature guide to opening a dial enclosure."),
        ("logger-cover", "Logger cover", "cover", "logger bay", "Show where a technician can reach the sample logger bay."),
    )),
    *group("workshop-storage", "#b49476", (
        ("wiring-cap", "Wiring cap", "rain cap", "terminal opening", "Illustrate how a raised rain cap shields a fictional wiring box."),
        ("fastener-bin", "Fastener bin", "bin front", "fastener pocket", "Show the lower-folding front of a small fastener bin."),
        ("cabinet-leaf", "Tool cabinet", "door", "tool rack", "Make a short illustrated instruction for a cabinet door."),
        ("clamp-rack", "Clamp rack", "side cover", "clamp rests", "Show a clamp rest enclosure opening for routine cleaning."),
    )),
    *group("laboratory", "#aba0aa", (
        ("optics-shutter", "Optics shutter", "shutter", "optics well", "Explain access to a small optical sample well."),
        ("sample-retainer", "Sample retainer", "retaining plate", "sample recess", "Illustrate how the plate folds out of the way when unloading."),
        ("rf-screen", "RF screen", "screen", "shielded recess", "Show the access screen on an illustrative RF enclosure."),
        ("reagent-locker", "Reagent locker", "locker leaf", "inner rack", "Make an offline equipment demonstration of the empty locker bay."),
    )),
    *group("museum-fixtures", "#8c9aad", (
        ("display-hood", "Display hood", "hood", "model cradle", "Demonstrate a hood over a museum model cradle."),
        ("ticket-gate", "Ticket tray", "tray gate", "ticket slot", "Explain the fold-down access gate of a specimen-free ticket tray."),
        ("plinth-service", "Plinth access", "panel", "fixing plate", "Make a service illustration showing how to open the plinth cladding."),
        ("label-pocket", "Label pocket", "pocket cover", "label recess", "Show how a holder opens to change its paper label."),
    )),
)
HELDOUT = (
    *group("vessel-hardware", "#a89986", (
        ("spray-deflector", "Spray deflector", "deflector", "deck vent", "Make an offline instruction for reaching a vessel vent beneath its spray deflector."),
        ("mooring-chute", "Mooring chute", "chute gate", "rope channel", "Demonstrate the chute gate of a fictional vessel without adding a powered motor."),
        ("locker-bulkhead", "Bulkhead locker", "locker front", "locker recess", "Illustrate the swing of a locker front near a vessel bulkhead."),
        ("cleat-keeper", "Cleat keeper", "keeper", "cleat bed", "Show how a swing-away keeper reveals a mooring cleat bed."),
    )),
    *group("building-openings", "#a6a08d", (
        ("clerestory-awning", "Clerestory awning", "awning", "window opening", "Show how an awning above an opening lifts clear on a fixed header."),
        ("parcel-drop", "Parcel drop", "drop face", "receiving slot", "Demonstrate the staged pose of a parcel drop face over a receiving slot."),
        ("transom-access", "Transom cabinet", "access leaf", "latch recess", "Create a local maintenance illustration of the side-mounted access leaf."),
        ("latch-shield", "Latch shield", "shield", "latch mount", "Show access to the latch mount in an offline architectural hardware detail."),
    )),
)
CASES = {case.id: case for case in (*TRAIN, *HELDOUT)}


STYLE = """
.scene{display:grid;place-items:center;min-height:224px;perspective:620px}
.assembly{position:relative;width:min(68%,238px);aspect-ratio:8/5;transform-style:preserve-3d}
.fixed,.leaf{position:absolute;inset:0;border-radius:.25rem}
.fixed{background:
 radial-gradient(circle at 11% 16%,#a4afb6 0 2px,transparent 3px),
 radial-gradient(circle at 89% 16%,#a4afb6 0 2px,transparent 3px),
 radial-gradient(circle at 11% 84%,#a4afb6 0 2px,transparent 3px),
 radial-gradient(circle at 89% 84%,#a4afb6 0 2px,transparent 3px),
 repeating-linear-gradient(90deg,#263842 0 18px,#344a56 18px 20px);
 box-shadow:0 0 0 10px #687d89,0 0 0 12px #293940;transform:translateZ(-3px)}
.leaf{background:linear-gradient(148deg,#ffffff31,transparent 62%),var(--paint);
 box-shadow:inset 0 0 0 2px #e4ecef66,inset 0 -6px #27364044;
 transform-origin:var(--pivot);transform:var(--rest);transition:transform 160ms ease;
 transform-style:preserve-3d}
.leaf:after{position:absolute;content:"";inset:23% 29%;border:1px solid #1b293588;border-radius:3px;
 background:repeating-linear-gradient(0deg,transparent 0 7px,#fdffff44 8px 9px)}
.assembly.open .leaf{transform:var(--open)}
.barrel{position:absolute;background:linear-gradient(0deg,#566870,#e4ecef 52%,#566870);
 box-shadow:0 0 0 1px #2c3b46;transform:translateZ(0);pointer-events:none;border-radius:2px}
.barrel.horizontal{left:12%;right:12%;height:6px}
.barrel.vertical{top:12%;bottom:12%;width:6px;background:linear-gradient(90deg,#566870,#e4ecef 52%,#566870)}
.barrel.top{top:-3px}.barrel.bottom{bottom:-3px}.barrel.left{left:-3px}.barrel.right{right:-3px}
@media(prefers-reduced-motion:reduce){.leaf{transition:none}}
"""


def page(case: Case) -> str:
    pivot, axis, angle = POSE[case.edge]
    ids = case.ids
    barrel = "horizontal" if axis == "X" else "vertical"
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(case.title)}</title>
<style>{BASE_STYLE}{STYLE}</style>
<main><h1>{escape(case.title)}</h1><section>
<div class="scene" aria-hidden="true">
  <div class="assembly" style="--pivot:{pivot};--rest:rotate{axis}(0deg);--open:rotate{axis}({angle}deg);--paint:{case.paint}">
    <div class="fixed" id="{ids['frame']}"></div>
    <div class="leaf" id="{ids['leaf']}"></div>
    <div class="barrel {barrel} {case.edge}"></div>
  </div>
</div>
<p id="{ids['status']}" role="status">{escape(case.closed)}</p>
<button type="button" id="{ids['toggle']}" aria-expanded="false" aria-controls="{ids['inside']}">{escape(case.button)}</button>
<p id="{ids['inside']}" hidden>{escape(case.reveal.capitalize())} sits behind the {escape(case.part)}.</p>
</section></main>
<script>
const button = document.getElementById("{ids['toggle']}");
const leaf = document.getElementById("{ids['leaf']}");
const detail = document.getElementById("{ids['inside']}");
const status = document.getElementById("{ids['status']}");
button.addEventListener("click", () => {{
  const open = button.getAttribute("aria-expanded") === "false";
  leaf.parentElement.classList.toggle("open", open);
  button.setAttribute("aria-expanded", String(open));
  detail.hidden = !open;
  status.textContent = open ? {json.dumps(case.opened)} : {json.dumps(case.closed)};
}});
</script>
</html>
"""


def family(case):
    return f"edge_hinge_{case.family}"


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    if len(CASES) != len(TRAIN) + len(HELDOUT) or {c.family for c in TRAIN} & {c.family for c in HELDOUT}:
        raise ValueError("Expected unique cases and disjoint mechanism families")
    catalog = {item["function"]["name"]: item for item in tools.tool_defs()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hinge-shard-", dir=out.parent) as folder:
        stage = Path(folder) / "release"
        train = stage / "train"
        heldout = stage / "heldout"
        train.mkdir(parents=True)
        heldout.mkdir()
        rows = []
        for case in TRAIN:
            episode = corpus.Episode(case.id, family(case), case.request(), ("write_file", "look_at_page"))
            scratch = Path(folder) / case.slug
            context = tools.ToolContext(Workspace(scratch), None, None, episode.id)
            source = page(case)
            observe(episode, context, "write_file", path=case.filename, content=source)
            if (scratch / case.filename).read_text(encoding="utf-8") != source:
                raise RuntimeError(f"{case.id}: unexpected write result")
            episode.answer(f"Saved {case.filename}. {case.closed} The button opens the {case.part} to expose the {case.reveal}.",
                           (case.filename, case.part))
            rows.append(episode.record(catalog))
        corpus.jsonl(train / "train.jsonl", rows)
        prompts = []
        for case in HELDOUT:
            (heldout / case.filename).write_text(page(case), encoding="utf-8")
            prompts.append({"id": case.id, "family": family(case), "message": case.request(),
                            "expected_file": case.filename})
        corpus.jsonl(heldout / "requests.jsonl", prompts)
        result = corpus.check(train, training_only=True)
        result.update(heldout={"episodes": len(HELDOUT), "families": sorted({c.family for c in HELDOUT})})
        result["sha256"] = {
            str(path.relative_to(stage)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        (stage / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        stage.rename(out)
    return result


def inspect(command, case, path, viewport):
    def require(condition, message):
        if not condition:
            raise RuntimeError(f"{case.id} at {viewport[0]} px: {message}")

    def js(expression, *, wait=False):
        reply = command("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=wait)
        require("exceptionDetails" not in reply, "the page or check threw an exception")
        return reply["result"].get("value")

    def key(value, code, number):
        args = {"key": value, "code": code, "windowsVirtualKeyCode": number}
        for action in ("rawKeyDown", "keyUp"):
            command("Input.dispatchKeyEvent", type=action, **args)

    def next_frame():
        js("new Promise(resolve => requestAnimationFrame(resolve))", wait=True)

    def snapshot():
        return js("""recordPose()""")

    width, height = viewport
    command("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=width < 500)
    url = path.resolve(strict=True).as_uri()
    require(not command("Page.navigate", url=url).get("errorText"), "could not load the local HTML")
    deadline = time.monotonic() + 4
    while not js(f"location.href === {json.dumps(url)} && document.readyState === 'complete'"):
        require(time.monotonic() < deadline, "the document did not finish loading")
        time.sleep(.02)
    js(f"""(() => {{
      const ids = {json.dumps(case.ids)}, edge = {json.dumps(case.edge)};
      const nodes = Object.fromEntries(Object.entries(ids).map(([name, id]) => [name, document.getElementById(id)]));
      const {{leaf, frame, toggle, status, inside}} = nodes;
      const w = leaf.offsetWidth, h = leaf.offsetHeight, style = getComputedStyle(leaf);
      const insetX = parseFloat(style.borderLeftWidth), insetY = parseFloat(style.borderTopWidth);
      const horizontal = edge === "top" || edge === "bottom";
      const near = edge === "top" || edge === "left";
      const edgeX = near ? 0 : w, edgeY = near ? 0 : h;
      const locations = horizontal ? [[0,edgeY],[w,edgeY],[w/2,h-edgeY]]
                                   : [[edgeX,0],[edgeX,h],[w-edgeX,h/2]];
      const points = locations.map(([x,y]) => {{
        const mark = document.createElement("i");
        mark.style.cssText = "position:absolute;width:0;height:0;visibility:hidden;pointer-events:none;" +
          "margin:0;padding:0;border:0;transform:none;left:" + (x-insetX) + "px;top:" + (y-insetY) + "px";
        leaf.append(mark);
        return mark;
      }});
      const rect = node => node.getBoundingClientRect().toJSON();
      const visible = node => {{
        const s = getComputedStyle(node);
        return !!node.getClientRects().length && s.visibility === "visible" && Number(s.opacity) > 0;
      }};
      window.recordPose = () => {{
        const style = getComputedStyle(leaf);
        const m = style.transform === "none" ? new DOMMatrix() : new DOMMatrix(style.transform);
        let perspective = m.m34 !== 0;
        for (let parent = leaf.parentElement; parent; parent = parent.parentElement)
          perspective ||= getComputedStyle(parent).perspective !== "none";
        return {{
          width: innerWidth, scroll: document.documentElement.scrollWidth,
          unique: Object.values(ids).every(id => document.querySelectorAll("#"+CSS.escape(id)).length === 1),
          button: rect(toggle), name: toggle.getAttribute("aria-label") || toggle.innerText.trim(),
          native: toggle.tagName === "BUTTON", focus: document.activeElement.id,
          expanded: toggle.getAttribute("aria-expanded"), controlled: toggle.getAttribute("aria-controls"),
          role: status.getAttribute("role"), status: status.innerText.trim(), statusVisible: visible(status),
          detail: visible(inside), frame: rect(frame), leaf: rect(leaf),
          pivot: style.transformOrigin.split(" ").map(parseFloat), size: [w,h], perspective,
          reduced: matchMedia("(prefers-reduced-motion: reduce)").matches &&
            [style.transitionDuration,style.animationDuration].every(s=>s.split(",").every(x=>parseFloat(x)===0)),
          depth: [m.m13,m.m23], flatMix: [m.m12,m.m21], points: points.map(p=>[rect(p).x,rect(p).y])
        }};
      }};
    }})()""")
    before = snapshot()
    button = before["button"]
    require(before["unique"] and before["native"] and before["name"] == case.button
            and before["controlled"] == case.ids["inside"] and before["role"] == "status",
            "native accessible control and unique named nodes")
    require(before["width"] == width and before["scroll"] <= width
            and min(button["width"], button["height"]) >= 44
            and 0 <= button["left"] < button["right"] <= width
            and 0 <= button["top"] < button["bottom"] <= height,
            "viewport width, visible control, and 44 px touch target")
    key("Tab", "Tab", 9)
    require(js("document.activeElement.id") == case.ids["toggle"], "Tab did not focus the control")
    key(" ", "Space", 32)
    next_frame()
    opened = snapshot()
    target = opened["button"]
    for action in ("mousePressed", "mouseReleased"):
        command("Input.dispatchMouseEvent", type=action, button="left", clickCount=1,
                x=(target["left"] + target["right"])/2, y=(target["top"] + target["bottom"])/2)
    next_frame()
    restored = snapshot()
    states = [before, opened, restored]
    require([s["expanded"] for s in states] == ["false", "true", "false"]
            and [s["detail"] for s in states] == [False, True, False]
            and [s["status"] for s in states] == [case.closed, case.opened, case.closed]
            and all(s["statusVisible"] for s in states)
            and all(s["focus"] == case.ids["toggle"] for s in (opened, restored)),
            "keyboard opening, pointer restoration, disclosure text and retained focus")
    return check_poses(case, states, require)


def check_poses(case, states, require):
    before, opened, restored = states
    horizontal = case.edge in ("top", "bottom")
    w, h = before["size"]
    if horizontal:
        wanted = (w / 2, 0 if case.edge == "top" else h)
        depth, off_axis = opened["depth"][::-1]
    else:
        wanted = (0 if case.edge == "left" else w, h / 2)
        depth, off_axis = opened["depth"]
    near = case.edge in ("top", "left")
    require(w > 30 and h > 30 and before["perspective"] and all(s["reduced"] for s in states),
            "visible perspective geometry and an immediate reduced-motion pose")
    require(all(math.dist(state["pivot"][:2], wanted) < 0.75 for state in states),
            f"the transform origin is off the {case.edge} hinge line")
    require((depth if near else -depth) > 0.5 and abs(off_axis) < 0.01
            and all(abs(v) < 0.01 for v in opened["flatMix"]),
            "the free edge did not swing toward the viewer about the required single axis")
    require(all(abs(v) < 0.01 for state in (before, restored) for v in state["depth"]),
            "the panel did not restore to its closed plane")
    anchors = [math.dist(a, b) for a, b in zip(before["points"], opened["points"], strict=True)]
    returned = [math.dist(a, b) for a, b in zip(before["points"], restored["points"], strict=True)]
    require(max(anchors[:2]) < 0.75 and anchors[2] > 8 and max(returned) < 0.75,
            "the projected hinge endpoints must stay still while the free edge moves and returns")
    require(before["frame"] == opened["frame"] == restored["frame"], "the mounting frame moved")
    return {"axis": "X" if horizontal else "Y", "pivot_px": before["pivot"][:2],
            "open_depth": round(depth, 4), "open_anchor_shift_px": [round(v, 3) for v in anchors],
            "reduced_motion": True, "frame_fixed": True}


def browser():
    chrome = pagecheck.find_chrome()
    if not chrome:
        raise RuntimeError("Chrome is required for the independent hinge check")
    return chrome


def verify(out):
    manifest_path = out / "manifest.json"
    stamp = manifest_path.read_bytes()
    for name, digest in json.loads(stamp)["sha256"].items():
        target = (out / name).resolve(strict=True)
        if not target.is_relative_to(out.resolve()) or sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Prepared hinge artifact changed: {name}")
    traces = {row["id"]: row for row in corpus.read_jsonl(out / "train/train.jsonl")}
    proof = {}
    with tempfile.TemporaryDirectory(prefix="hinge-browser-") as folder:
        scratch = Path(folder)
        with chrome_session(browser(), scratch) as command:
            for case in (*TRAIN, *HELDOUT):
                if case.id in traces:
                    call, = traces[case.id]["messages"][2]["tool_calls"]
                    args = call["function"]["arguments"]
                    if call["function"]["name"] != "write_file" or args["path"] != case.filename:
                        raise ValueError(f"{case.id}: first tool action must write the requested HTML")
                    path = scratch / case.filename
                    path.write_text(args["content"], encoding="utf-8")
                else:
                    path = out / "heldout" / case.filename
                proof[case.id] = {str(width): inspect(command, case, path, (width, height))
                                  for width, height in VIEWPORTS}
    receipt = {"manifest_sha256": sha256(stamp).hexdigest(), "train": len(TRAIN),
               "heldout": len(HELDOUT), "viewports": [width for width, _ in VIEWPORTS], "cases": proof}
    target = out / "browser-verification.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "verify"):
        modes.add_parser(name).add_argument("out", type=Path)
    check = modes.add_parser("check")
    check.add_argument("case", choices=[case.id for case in HELDOUT])
    check.add_argument("page", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.out)
    elif args.mode == "verify":
        result = verify(args.out)
    else:
        with tempfile.TemporaryDirectory(prefix="hinge-candidate-") as folder:
            with chrome_session(browser(), Path(folder)) as command:
                result = {str(w): inspect(command, CASES[args.case], args.page, (w, h)) for w, h in VIEWPORTS}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
