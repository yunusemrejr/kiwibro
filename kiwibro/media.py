"""Media handling for helper models and the no-model fallback.

* ``shrink_image`` / ``shrink_image_parts``: vision prompts cost tokens per
  pixel, so every image is downscaled before it reaches a model.
* ``video_frames``: evenly spaced keyframes (ffmpeg) so a vision model can
  "watch" a video.
* ``media_facts``: deterministic, code-only analysis (dimensions, palette,
  luminance, an ASCII thumbnail, SVG text, WAV loudness, ffprobe streams) for
  when no installed model has the needed capability.

Optional tools are feature-detected — Pillow, ImageMagick, ffmpeg/ffprobe —
and every function degrades to a smaller answer instead of failing.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

log = logging.getLogger(__name__)

MAX_IMAGE_SIDE = 1024  # keeps screenshots legible while bounding vision tokens
MAX_FRAMES = 6
TOOL_TIMEOUT_S = 30
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus", ".aac"}
_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.+)$", re.DOTALL)
_ASCII_RAMP = " .:-=+*#%@"


def _pillow():
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def _run(command: list[str], data: bytes | None = None) -> bytes | None:
    try:
        done = subprocess.run(command, input=data, capture_output=True, timeout=TOOL_TIMEOUT_S, check=True)
    except (OSError, subprocess.SubprocessError):
        log.debug("media command failed: %s", command[0], exc_info=True)
        return None
    return done.stdout or None


def media_kind(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".svg":
        return "svg"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return "other"


# --- images ------------------------------------------------------------------------


def shrink_image(data: bytes, max_side: int = MAX_IMAGE_SIDE) -> tuple[bytes, str] | None:
    """Downscale to ``max_side`` px as JPEG/PNG (Pillow, else ImageMagick); None if neither can."""
    Image = _pillow()  # noqa: N806
    if Image is not None:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if max(image.size) <= max_side and image.format in ("JPEG", "PNG"):
                    return data, f"image/{image.format.lower()}"
                image.thumbnail((max_side, max_side))
                out = io.BytesIO()
                if image.mode in ("RGBA", "LA", "P"):
                    image.save(out, "PNG", optimize=True)
                    return out.getvalue(), "image/png"
                image.convert("RGB").save(out, "JPEG", quality=85)
                return out.getvalue(), "image/jpeg"
        except Exception:  # corrupt/unsupported for Pillow: try ImageMagick
            log.debug("Pillow could not shrink image", exc_info=True)
    convert = shutil.which("magick") or shutil.which("convert")
    if convert:
        shrunk = _run([convert, "-", "-resize", f"{max_side}x{max_side}>", "-strip", "png:-"], data)
        if shrunk:
            return shrunk, "image/png"
    return None


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def image_data_url(data: bytes, mime: str = "image/png", max_side: int = MAX_IMAGE_SIDE) -> str:
    """A model-ready data URL, downscaled when a tool for it is available."""
    shrunk = shrink_image(data, max_side)
    return data_url(*shrunk) if shrunk else data_url(data, mime)


def shrink_image_parts(messages: list[dict[str, Any]], max_side: int = MAX_IMAGE_SIDE) -> list[dict[str, Any]]:
    """Copy of OpenAI-style ``messages`` with every base64 image part downscaled."""
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = [_shrunk_part(part, max_side) for part in content]
        out.append({**message, "content": parts})
    return out


def shrink_data_urls(urls: list[str], max_side: int = MAX_IMAGE_SIDE) -> list[str]:
    return [_shrunk_url(url, max_side) for url in urls]


def _shrunk_part(part: Any, max_side: int) -> Any:
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return part
    url = (part.get("image_url") or {}).get("url")
    if not isinstance(url, str):
        return part
    return {**part, "image_url": {**part["image_url"], "url": _shrunk_url(url, max_side)}}


def _shrunk_url(url: str, max_side: int) -> str:
    match = _DATA_URL_RE.match(url)
    if not match:
        return url
    try:
        raw = base64.b64decode(match.group(2), validate=False)
    except (ValueError, TypeError):
        return url
    shrunk = shrink_image(raw, max_side)
    return data_url(*shrunk) if shrunk else url


# --- video ---------------------------------------------------------------------------


def video_frames(path: str | Path, count: int = 4, max_side: int = MAX_IMAGE_SIDE) -> list[bytes]:
    """Evenly spaced PNG keyframes via ffmpeg ([] without ffmpeg or on failure)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    count = max(1, min(MAX_FRAMES, count))
    duration = (probe(path) or {}).get("duration_seconds") or 0
    frames: list[bytes] = []
    for index in range(count):
        at = duration * (index + 0.5) / count if duration else index
        frame = _run([
            ffmpeg, "-v", "error", "-ss", f"{at:.2f}", "-i", str(path), "-frames:v", "1",
            "-vf", f"scale='min({max_side},iw)':-2", "-f", "image2pipe", "-vcodec", "png", "-",
        ])
        if frame:
            frames.append(frame)
    return frames


# --- deterministic analysis -----------------------------------------------------------


def probe(path: str | Path) -> dict[str, Any] | None:
    """Container/stream facts from ffprobe, or None."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    raw = _run([ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    try:
        data = json.loads(raw or b"")
    except ValueError:
        return None
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    facts: dict[str, Any] = {
        "container": fmt.get("format_name"),
        "duration_seconds": number(fmt.get("duration")),
        "bit_rate": number(fmt.get("bit_rate")),
    }
    if video:
        facts.update(video_codec=video.get("codec_name"), width=video.get("width"), height=video.get("height"),
                     frame_rate=video.get("avg_frame_rate"))
    if audio:
        facts.update(audio_codec=audio.get("codec_name"), sample_rate=number(audio.get("sample_rate")),
                     channels=audio.get("channels"))
    return {key: value for key, value in facts.items() if value not in (None, "", 0)}


def _image_facts(path: Path) -> dict[str, Any]:
    Image = _pillow()  # noqa: N806
    if Image is None:
        return {}
    with Image.open(path) as image:
        facts: dict[str, Any] = {"format": image.format, "mode": image.mode,
                                 "width": image.width, "height": image.height}
        gray = image.convert("L")
        sample = list(gray.resize((min(128, image.width), min(128, image.height))).getdata())
        mean = sum(sample) / len(sample)
        facts["luminance_mean"] = round(mean, 1)
        facts["luminance_std"] = round(math.sqrt(sum((v - mean) ** 2 for v in sample) / len(sample)), 1)
        colors = image.convert("RGB").resize((min(64, image.width), min(64, image.height))).getcolors(4096) or []
        colors.sort(reverse=True)
        facts["dominant_colors"] = [f"#{r:02x}{g:02x}{b:02x}" for _count, (r, g, b) in colors[:5]]
        thumb = list(gray.resize((32, 12)).getdata())
        rows = [thumb[row * 32:(row + 1) * 32] for row in range(12)]
        facts["ascii_thumbnail"] = "\n".join(
            "".join(_ASCII_RAMP[min(len(_ASCII_RAMP) - 1, value * len(_ASCII_RAMP) // 256)] for value in row)
            for row in rows
        )
    return facts


def _svg_facts(path: Path) -> dict[str, Any]:
    root = ElementTree.parse(path).getroot()
    elements = list(root.iter())
    texts = [elem.text.strip() for elem in elements if elem.text and elem.text.strip()]
    tags = sorted({elem.tag.rsplit("}", 1)[-1] for elem in elements})
    return {"width": root.get("width"), "height": root.get("height"), "view_box": root.get("viewBox"),
            "elements": len(elements), "element_types": tags[:20], "text": texts[:20]}


def _wav_facts(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        rate, frames, width, channels = (handle.getframerate(), handle.getnframes(),
                                         handle.getsampwidth(), handle.getnchannels())
        facts: dict[str, Any] = {"sample_rate": rate, "channels": channels,
                                 "duration_seconds": round(frames / rate, 3) if rate else None}
        if width == 2 and rate:
            raw = handle.readframes(min(frames, rate * 30))  # first 30 s bounds the work
            samples = struct.unpack(f"<{len(raw) // 2}h", raw[: len(raw) // 2 * 2])
            if samples:
                rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                crossings = sum(1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0))
                facts.update(
                    rms_dbfs=round(20 * math.log10(rms / 32768), 1) if rms else None,
                    peak=max(abs(s) for s in samples),
                    zero_crossings_per_s=round(crossings / (len(samples) / channels / rate), 1),
                )
    return facts


def media_facts(path: str | Path) -> dict[str, Any]:
    """Deterministic facts about a media file (no model needed). Never raises for bad media."""
    target = Path(path)
    kind = media_kind(target)
    facts: dict[str, Any] = {"name": target.name, "kind": kind, "size_bytes": target.stat().st_size}
    analysers = {"image": _image_facts, "svg": _svg_facts}
    try:
        if kind in analysers:
            facts.update(analysers[kind](target))
        elif kind == "audio" and target.suffix.lower() == ".wav":
            facts.update(_wav_facts(target))
        if kind in ("audio", "video") or (kind == "image" and "width" not in facts):
            facts.update(probe(target) or {})
    except Exception as exc:  # malformed media must not break the agent
        facts["error"] = f"{exc.__class__.__name__}: {exc}"[:200]
    return {key: value for key, value in facts.items() if value not in (None, "", [], {})}


def facts_text(facts: dict[str, Any]) -> str:
    """Compact, model-friendly rendering of ``media_facts``."""
    thumbnail = facts.get("ascii_thumbnail")
    lines = [f"- {key}: {value}" for key, value in facts.items() if key != "ascii_thumbnail"]
    if thumbnail:
        lines.append("- ascii_thumbnail:\n" + thumbnail)
    return "\n".join(lines)
