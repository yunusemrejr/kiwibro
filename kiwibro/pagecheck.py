"""Looking at a page the model wrote.

A small model says "here is your page" about a page that is blank, throws an error on its first line or does not
move. ``look`` opens the page in headless Chrome, takes a picture (and a second one two seconds later when the
page should move), lets a small probe script count what the page does, and returns plain findings the model can fix
and the user can read. A finding is only raised from something measured: when Chrome is missing or a render times
out the result says the page could not be looked at, which is not a problem with the page.

The probe is inserted into a temporary copy of the page that sits beside the original (so relative links keep
working) and is removed afterwards. It reports through one console line that is read from Chrome's stderr.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from . import connectivity
from .sandbox import SandboxError, Workspace

try:
    from PIL import Image, ImageChops
except ImportError:  # the picture measures are skipped without Pillow
    Image = ImageChops = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

DESKTOP = (1280, 800)
PHONE = (360, 740)
MIN_WINDOW = 500  # headless Chrome never opens a window narrower than this
BUDGET_MS = 2600  # virtual milliseconds the page runs before the first picture
LATER_MS = 2000  # the second picture is taken this much later
TIMEOUT_S = 40  # real seconds one Chrome run may take
MAX_HTML_BYTES = 2_000_000
CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")
PICTURE_DIR = "screenshots"
HTML_SUFFIXES = (".html", ".htm")
_LOCK = threading.Lock()  # one render at a time: each takes about 230 MB
_PLACEHOLDER = re.compile(r"\{\{\s*[A-Za-z_][\w.-]*\s*\}\}")
_MOTION = re.compile(
    r"\b(?:animat\w*|mov(?:e|es|ed|ing)|slid\w*|bounc\w*|spin\w*|rotat\w*|orbit\w*|puls\w*|blink\w*|drift\w*|swing\w*"
    r"|flies|flying|walks|walking|falls|falling|forever|3-?d|game|simulation)\b", re.IGNORECASE)
_PHONE = re.compile(r"\b(?:phones?|mobile|responsive|smartphone)\b", re.IGNORECASE)
_ANSWER = re.compile(r"^\W*(yes|no)\b[\W]*", re.IGNORECASE)

# Runs before any page script: records what goes wrong and what moves, then reports once near the end of the time budget.
_PROBE = """(function(){var P="@PREFIX@",E=[],F=[],C=[],raf=0,sent=false;
function s(v){return String(v==null?"":v).slice(0,300)}
addEventListener("error",function(e){var t=e.target;
if(t&&t!==window&&(t.src||t.href)){F.push({u:String(t.src||t.href),m:t.type==="module"})}
else{E.push({m:s(e.message),l:e.lineno||0})}},true);
addEventListener("unhandledrejection",function(e){var r=e.reason;E.push({m:s(r&&r.message||r),l:0})});
var ce=console.error;console.error=function(){try{C.push(s([].map.call(arguments,function(a){return a&&a.message||a}).join(" ")))}catch(x){}return ce.apply(console,arguments)};
var ra=window.requestAnimationFrame;window.requestAnimationFrame=function(f){return ra.call(window,function(t){raf++;return f(t)})};
function seen(n){var r=n.getBoundingClientRect();return r.width>0&&r.height>0}
function report(){if(sent)return;sent=true;var b=document.body,t=b?b.innerText:"",d=document.documentElement;
var cv=[].filter.call(document.querySelectorAll("canvas"),seen).length;
var md=[].filter.call(document.querySelectorAll("img,svg,video"),seen).length;
var el=b?[].filter.call(b.querySelectorAll("*"),function(n){return !/^(SCRIPT|STYLE|NOSCRIPT|TEMPLATE)$/.test(n.tagName)}).length:0;
var an=document.getAnimations?document.getAnimations().filter(function(a){return a.playState==="running"}).length:0;
console.log(P+encodeURIComponent(JSON.stringify({e:E.slice(0,4),f:F.slice(0,6),c:C.slice(0,3),raf:raf,an:an,text:t.slice(0,4000),tl:t.trim().length,cv:cv,md:md,el:el,w:innerWidth,sw:Math.max(d.scrollWidth,b?b.scrollWidth:0)})))}
setTimeout(report,@MS@);})();""".replace("\n", "")  # one line: the page's own line numbers stay what they were


@dataclass(frozen=True)
class Finding:
    """One thing wrong with a page: a sentence for the tool result, a clause for Kiwibro's own lines, and what is
    said when it is gone."""

    code: str
    text: str
    short: str
    fixed: str


@dataclass(frozen=True)
class Options:
    """What the request expects of the page."""

    moves: bool = False  # the page should move: "nothing moves" is only raised then
    phone: bool = False  # the phone look is asked for: it is also looked at 360 px wide
    not_text: tuple[str, ...] = ()  # texts the page must not show (a kit's notice that its values were not understood)
    goal: str = ""  # goal mode: the picture is also asked whether it shows this


@dataclass
class Render:
    """What one Chrome run gave back: the probe's report (None when it said nothing) or why nothing was rendered."""

    report: dict[str, Any] | None = None
    error: str = ""


@dataclass
class Look:
    page: str
    sizes: tuple[str, ...] = ()
    again: bool = False
    findings: list[Finding] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)
    picture: str = ""
    seen: str = ""
    could_not: str = ""  # why the page could not be looked at ("" when it was)

    def problems(self) -> str:
        return "; ".join(finding.short for finding in self.findings)

    def report(self) -> str:
        """The tool result: what was done, what was seen, then the problems or what was confirmed."""
        if self.could_not:
            done = f"Looked at {self.page} without a browser." if self.findings else f"Could not look at {self.page}: {self.could_not}."
        else:
            done = f"Looked at {self.page} at {self.sizes[0]}" + (", and again 2 seconds later" if self.again else "")
            if len(self.sizes) > 1:
                done += f", and at {self.sizes[1]}"
            done += "."
        lines = [done]
        if self.picture:
            lines.append(f"Picture: {self.picture}")
        if self.seen:
            lines.append(f"Seen: {self.seen}")
        if self.findings:
            lines += ["Problems:", *(f"- {finding.text}" for finding in self.findings), f"Fix each of these in {self.page}."]
        elif self.could_not:
            lines.append("This is not a problem with the page. Nothing more is to be done.")
        else:
            lines.append(f"No problems found: {join_words(self.confirmed)}. Nothing more is to be done."
                         if self.confirmed else "No problems found. Nothing more is to be done.")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page, "picture": self.picture, "seen": self.seen, "could_not": self.could_not,
            "confirmed": list(self.confirmed),
            "findings": [{"code": f.code, "text": f.text, "short": f.short, "fixed": f.fixed} for f in self.findings],
        }


def join_words(parts: list[str]) -> str:
    """"a, b and c"."""
    if len(parts) < 2:
        return "".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def wants_motion(message: str) -> bool:
    """The request names something that moves (the page is looked at again 2 seconds later)."""
    return bool(_MOTION.search(message or ""))


def wants_phone(message: str) -> bool:
    return bool(_PHONE.search(message or ""))


# --- findings -----------------------------------------------------------------------------------


def _quote(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def script_error(message: str, line: int) -> Finding:
    where = f", line {line}" if line else ""
    said = _quote(message)
    return Finding("script", f'A script error stops the page: "{said}"{where}.',
                  f'a script error stops the page ("{said}"{where})', "the script error is gone")


def logged_error(message: str) -> Finding:
    said = _quote(message)
    return Finding("logged", f'The page reports an error: "{said}".', f'the page reports an error ("{said}")',
                  "the reported error is gone")


def missing_file(name: str, *, local: bool = True, module: bool = False) -> Finding:
    if module:
        return Finding("module", f"The page loads {name} as a module script, which does not load from a local file. "
                      "Put its code inside the page.", f"{name} is a module script that does not load from a local file",
                      f"{name} is inside the page now")
    if local:
        return Finding("missing", f"The page loads {name} and that file is not there.",
                      f"it loads {name} and that file is not there", f"{name} is found")
    return Finding("missing", f"The page loads {name} and it did not load.", f"it loads {name} and that did not load",
                  f"{name} loads")


FLAT = Finding("flat", "The picture is one flat colour, so nothing is drawn.",
              "the picture is one flat colour, so nothing is drawn", "it draws now")
EMPTY = Finding("empty", "The page has no visible content.", "the page has no visible content", "it has content now")
STILL = Finding("still", "Nothing moves: two pictures two seconds apart are the same.", "nothing moves", "it moves now")
WIDE = Finding("wide", "At phone width (360 px) the page is wider than the screen.",
              "at phone width the page is wider than the screen", "it fits the phone screen now")


def unfilled(text: str) -> Finding:
    return Finding("unfilled", f'Unfilled text is showing: "{text}".', f'unfilled text is showing ("{text}")',
                  "the unfilled text is filled in")


def notice(line: str) -> Finding:
    said = _quote(line)
    return Finding("notice", f'The page shows a notice that its values were not understood: "{said}".',
                  f'the page shows "{said}"', "the notice is gone")


def unlike(sentence: str) -> Finding:
    said = _quote(sentence, 200)
    return Finding("unlike", f"It does not look like what was asked: {said}",
                  f"it does not look like what was asked: {said}", "it looks like what was asked now")


# --- checks that need no browser ------------------------------------------------------------------


class _Parts(HTMLParser):
    """The local files a page loads, the text it shows and how many elements it has."""

    LOADS = {"script": "src", "img": "src", "source": "src", "video": "src", "audio": "src", "link": "href"}
    BARE = {"html", "head", "meta", "title", "link", "script", "style", "base", "body", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.files: list[tuple[str, bool]] = []
        self.text: list[str] = []
        self.elements = 0
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in ("script", "style"):
            self._hidden += 1
        if tag not in self.BARE:
            self.elements += 1
        wanted = self.LOADS.get(tag)
        if wanted and values.get(wanted):
            if tag != "link" or (values.get("rel") or "").lower() in ("stylesheet", "icon", "modulepreload", "preload"):
                self.files.append((str(values[wanted]), (values.get("type") or "").lower() == "module"))

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._hidden:
            self._hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden:
            self.text.append(data)


def _local(link: str) -> str:
    """The relative path a link points at, or "" for anything that is not a local file."""
    parts = urlsplit(link.strip())
    if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
        return ""
    return unquote(parts.path)


def static_findings(page: Path, source: str) -> list[Finding]:
    """What the file alone shows: local files it loads that are not there, unfilled text, nothing in the body."""
    parts = _Parts()
    try:
        parts.feed(source)
        parts.close()
    except Exception:  # broken markup is the browser's to read
        log.info("pagecheck: could not parse %s", page, exc_info=True)
    found: list[Finding] = []
    for link, module in parts.files:
        name = _local(link)
        if name and not (page.parent / name).exists() and all(f.short != missing_file(name).short for f in found):
            found.append(missing_file(name))
    shown = "".join(parts.text)
    holder = _PLACEHOLDER.search(shown)
    if holder:
        found.append(unfilled(holder.group(0)))
    if not shown.strip() and not parts.elements:
        found.append(EMPTY)
    return found


# --- the picture ---------------------------------------------------------------------------------


def _flat(png: Path) -> bool | None:
    """Is the whole picture one colour? None without Pillow."""
    if Image is None:
        return None
    try:
        with Image.open(png) as image:
            return image.convert("RGB").getcolors(maxcolors=1) is not None
    except Exception:
        log.info("pagecheck: could not read %s", png, exc_info=True)
        return None


def _same(first: Path, second: Path) -> bool | None:
    """Are the two pictures the same, pixel for pixel? None without Pillow."""
    if Image is None:
        return None
    try:
        with Image.open(first) as one, Image.open(second) as two:
            return one.size == two.size and ImageChops.difference(one.convert("RGB"), two.convert("RGB")).getbbox() is None
    except Exception:
        log.info("pagecheck: could not compare the pictures", exc_info=True)
        return None


def _crop(png: Path, width: int) -> None:
    """A phone render sits in a window that is wider than the phone: keep the phone's part."""
    if Image is None:
        return
    try:
        with Image.open(png) as image:
            if image.width <= width:
                return
            kept = image.crop((0, 0, width, image.height))
        kept.save(png)
    except Exception:
        log.info("pagecheck: could not crop %s", png, exc_info=True)


# --- Chrome ---------------------------------------------------------------------------------------


def find_chrome() -> str | None:
    for name in CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _with_probe(source: str, probe: str) -> str:
    tag = f"<script>{probe}</script>"
    start = (re.search(r"<head(?:\s[^>]*)?>", source, re.IGNORECASE) or re.search(r"<html(?:\s[^>]*)?>", source, re.IGNORECASE)
             or re.match(r"\s*<!doctype[^>]*>", source, re.IGNORECASE))
    return source[: start.end()] + tag + source[start.end():] if start else tag + source


def _quit(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def chrome_render(page: Path, png: Path, width: int, height: int, budget_ms: int) -> Render:
    """Open ``page`` in a fresh headless Chrome, run it for ``budget_ms`` of page time and save the picture."""
    chrome = find_chrome()
    if chrome is None:
        return Render(error="Chrome is not installed")
    nonce = secrets.token_hex(4)
    prefix = f"KBLOOK{nonce}:"
    probe = _PROBE.replace("@PREFIX@", prefix).replace("@MS@", str(max(budget_ms - 200, 100)))
    try:
        source = page.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return Render(error=f"the file could not be read ({exc.strerror or exc})")
    copy = page.with_name(f".kblook-{nonce}.html")
    wrapper = page.with_name(f".kblook-{nonce}-phone.html")
    window = (max(width, MIN_WINDOW), height)
    profile = tempfile.mkdtemp(prefix="kiwibro-look-")
    try:
        copy.write_text(_with_probe(source, probe), encoding="utf-8-sig")  # a byte order mark: the probe pushes <meta charset> out of the first 1024 bytes
        target = copy
        if width < MIN_WINDOW:  # the page sits in a frame as wide as the phone, inside the smallest window
            wrapper.write_text(f'<!doctype html><body style="margin:0;background:#fff"><iframe src="{copy.name}" '
                               f'style="display:block;border:0;width:{width}px;height:{height}px"></iframe></body>',
                               encoding="utf-8")
            target = wrapper
        argv = [chrome, "--headless=new", "--no-first-run", "--no-default-browser-check", f"--user-data-dir={profile}",
                "--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--hide-scrollbars", "--mute-audio",
                f"--window-size={window[0]},{window[1]}", f"--virtual-time-budget={budget_ms}",
                "--enable-logging=stderr", f"--screenshot={png}", target.resolve().as_uri()]
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   text=True, errors="replace", start_new_session=True)
        try:
            _out, logged = process.communicate(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _quit(process)
            process.communicate()
            return Render(error=f"the page took longer than {TIMEOUT_S} seconds to load")
        finally:
            if process.poll() is None:
                _quit(process)
        if not png.is_file() or png.stat().st_size == 0:
            return Render(error="Chrome could not draw the page")
        if width < MIN_WINDOW:
            _crop(png, width)
        found = re.search(re.escape(prefix) + r"([%\w.~!*'()-]+)", logged or "")
        if not found:
            return Render()
        try:
            return Render(report=json.loads(unquote(found.group(1))))
        except ValueError:
            return Render()
    except OSError as exc:
        return Render(error=f"Chrome could not be started ({exc.strerror or exc})")
    finally:
        for stray in (copy, wrapper):
            stray.unlink(missing_ok=True)
        shutil.rmtree(profile, ignore_errors=True)


# --- the look -------------------------------------------------------------------------------------


def _from_report(report: dict[str, Any], page: Path, flat: bool | None) -> list[Finding]:
    """What one probe report shows is wrong with the page."""
    found: list[Finding] = []
    for error in report.get("e") or ():
        found.append(script_error(str(error.get("m") or "error"), int(error.get("l") or 0)))
    for message in report.get("c") or ():
        found.append(logged_error(str(message)))
    for failed in report.get("f") or ():
        link = str(failed.get("u") or "")
        if link.startswith("file:"):
            full = Path(unquote(urlsplit(link).path))
            name = full.relative_to(page.parent).as_posix() if full.is_relative_to(page.parent) else full.name
            found.append(missing_file(name, module=bool(failed.get("m")) and full.exists()))
        else:
            found.append(missing_file(link, local=False))
    content = bool(report.get("tl") or report.get("cv") or report.get("md"))
    if not content and (flat if flat is not None else not report.get("el")):
        found.append(EMPTY)
    elif flat and content:
        found.append(FLAT)
    holder = _PLACEHOLDER.search(str(report.get("text") or ""))
    if holder:
        found.append(unfilled(holder.group(0)))
    return found


def _notices(report: dict[str, Any], not_text: tuple[str, ...]) -> list[Finding]:
    lines = [line.strip() for line in str(report.get("text") or "").splitlines() if line.strip()]
    found = []
    for wanted in not_text:
        line = next((line for line in lines if wanted in line), "")
        if line:
            found.append(notice(line))
    return found


def _answer(text: str) -> tuple[bool | None, str]:
    """A yes or no said first, and the sentence after it."""
    said = " ".join(str(text).split())
    match = _ANSWER.match(said)
    sentence = said[match.end():] if match else said
    return (match.group(1).lower() == "yes" if match else None), sentence.strip()[:240]


def _slug(shown: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", shown.rsplit(".", 1)[0]).strip("-") or "page"


def look(
    workspace: Workspace,
    path: str,
    options: Options | None = None,
    *,
    render: Callable[[Path, Path, int, int, int], Render] | None = None,
    eyes: Callable[[str, str], str] | None = None,
) -> Look:
    """Look at one HTML file of the workspace. ``render`` and ``eyes`` stand in for Chrome and for a model that can see."""
    options = options or Options()
    render = render or chrome_render
    target = workspace.resolve(path, must_exist=True)
    if not target.is_file() or target.suffix.lower() not in HTML_SUFFIXES:
        raise SandboxError(f"{workspace.public(target)} is not an HTML file")
    if target.stat().st_size > MAX_HTML_BYTES:
        raise SandboxError(f"{workspace.public(target)} is too large to look at")
    shown = workspace.public(target)
    result = Look(page=shown, sizes=(f"{DESKTOP[0]}x{DESKTOP[1]}",))
    result.findings = static_findings(target, target.read_text(encoding="utf-8", errors="replace"))
    kept = workspace.resolve(f"{PICTURE_DIR}/look-{_slug(shown)}.png")
    with _LOCK, tempfile.TemporaryDirectory(prefix="kiwibro-look-") as scratch:
        first = Path(scratch) / "first.png"
        main = render(target, first, *DESKTOP, BUDGET_MS)
        if main.error:
            result.could_not = main.error
            return result
        picture, report = first, main.report
        flat = _flat(first)
        if options.moves:
            later = Path(scratch) / "later.png"
            again = render(target, later, *DESKTOP, BUDGET_MS + LATER_MS)
            if not again.error:
                result.again, picture = True, later
                report = again.report or report
        problems = _from_report(report, target, flat) if report else ([EMPTY] if flat else [])
        problems += _notices(report or {}, options.not_text)
        offline = next((f["u"] for f in (report or {}).get("f") or () if not f["u"].startswith("file:")), "")
        if offline and not connectivity.online():
            result.could_not = f"the page loads {urlsplit(offline).netloc or offline} from the internet and there is no connection"
            return result
        result.findings += [found for found in problems if found.short not in {f.short for f in result.findings}]
        same = _same(first, picture) if result.again else None
        if same and report and not report.get("raf") and not report.get("an"):
            result.findings.append(STILL)
        if options.phone:
            small = render(target, Path(scratch) / "phone.png", *PHONE, BUDGET_MS)
            result.sizes += (f"{PHONE[0]}x{PHONE[1]}",)
            narrow = small.report or {}
            if not small.error and narrow.get("w") == PHONE[0] and int(narrow.get("sw") or 0) > PHONE[0]:
                result.findings.append(WIDE)
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(picture, kept)
        result.picture = workspace.public(kept)
        if report:
            blocking = {"script", "logged", "missing", "module"}
            if not any(f.code in blocking for f in result.findings):
                result.confirmed.append("it loads without errors")
            if flat is False and not any(f.code in ("flat", "empty") for f in result.findings):
                result.confirmed.append("it draws")
            if same is False:
                result.confirmed.append("it moves")
            if options.phone and not any(f.code == "wide" for f in result.findings) and len(result.sizes) > 1:
                result.confirmed.append("it fits a phone screen")
    _see(result, options, eyes)
    return result


def _see(result: Look, options: Options, eyes: Callable[[str, str], str] | None) -> None:
    """Goal mode only: the picture is asked whether it shows the goal. A small vision model that is asked what a
    picture shows made up a person in a black shirt for a plain page, so a free description is never reported."""
    if eyes is None or not options.goal or not result.picture or any(f.code in ("empty", "flat") for f in result.findings):
        return
    question = f"The picture should show: {options.goal}. Does it? Answer yes or no first, then one sentence on what it shows."
    try:
        said = eyes(result.picture, question)
    except Exception as exc:  # a model that cannot see is no reason to fail the look
        log.info("pagecheck: nothing could look at the picture: %s", exc)
        return
    agrees, sentence = _answer(said)
    result.seen = sentence
    if agrees is False:
        result.findings.append(unlike(sentence))
    elif agrees:
        result.confirmed.append("it looks like what was asked")
