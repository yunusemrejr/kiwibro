"""Computer-use helpers: system info, opening paths/URLs, clipboard,
notifications and screenshots.

Every capability degrades gracefully: when the desktop binary it needs is not
installed (headless box, different desktop), the function raises
``DesktopError`` with a plain message instead of failing obscurely. Nothing
here uses a shell — arguments are always passed as argv lists.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .sandbox import SandboxError, Workspace

log = logging.getLogger(__name__)

CLIPBOARD_READ_CAP = 20000
CLIPBOARD_WRITE_CAP = 100 * 1024
DESKTOP_TIMEOUT_S = 10
SCREENSHOT_TIMEOUT_S = 20


class DesktopError(Exception):
    """A desktop capability is unavailable or failed; message is user-facing."""


def _which(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _run(argv: list[str], *, input_text: str | None = None, timeout: int = DESKTOP_TIMEOUT_S) -> str:
    try:
        completed = subprocess.run(
            argv,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DesktopError(f"{argv[0]} timed out") from exc
    except OSError as exc:
        raise DesktopError(f"could not run {argv[0]}: {exc.strerror or exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise DesktopError(f"{Path(argv[0]).name} failed: {detail[:200] or 'unknown error'}")
    return completed.stdout or ""


def _human_bytes(size: float) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


# --- system_info ---------------------------------------------------------------


def _read_text(path: Path) -> str:
    """Contents of a pseudo-file, '' when it cannot be read."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _count_unit(count: int, unit: str) -> str:
    """``3 days`` / ``1 day``."""
    return f"{count} {unit}{'' if count == 1 else 's'}"


def _cpuinfo_fields(proc: Path) -> dict[str, str]:
    """Key/value pairs of the first block of /proc/cpuinfo."""
    fields: dict[str, str] = {}
    for line in _read_text(proc / "cpuinfo").splitlines():
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        if sep and key.strip() not in fields:
            fields[key.strip()] = value.strip()
    return fields


def _cpuinfo_core_pairs(proc: Path) -> set[tuple[str, str]]:
    """The distinct (physical id, core id) pairs of /proc/cpuinfo."""
    pairs: set[tuple[str, str]] = set()
    socket = ""
    slot = ""
    for line in _read_text(proc / "cpuinfo").splitlines() + [""]:
        key, sep, value = line.partition(":")
        if not line.strip():
            if slot:
                pairs.add((socket, slot))
            socket = slot = ""
            continue
        if not sep:
            continue
        name = key.strip()
        if name == "physical id":
            socket = value.strip()
        elif name == "core id":
            slot = value.strip()
    return pairs


def _cpu_model(proc: Path = Path("/proc")) -> str:
    """Brand of the processor; ARM boards report it under Hardware or Model."""
    fields = _cpuinfo_fields(proc)
    for name in ("model name", "Hardware", "Model"):
        if fields.get(name):
            return fields[name]
    return ""


def _cpu_cores(proc: Path = Path("/proc")) -> str:
    """Physical cores and threads; only threads when the kernel hides the topology."""
    threads = os.cpu_count() or 0
    cores = len(_cpuinfo_core_pairs(proc))
    if not cores:
        return f"{threads} threads" if threads else ""
    return f"{cores} cores, {threads} threads" if threads else f"{cores} cores"


def _meminfo_kbytes(proc: Path, name: str) -> int | None:
    """A /proc/meminfo line in bytes; the file states its values in kB."""
    for line in _read_text(proc / "meminfo").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == name:
            fields = value.split()
            try:
                return int(fields[0]) * 1024 if fields else None
            except ValueError:
                return None
    return None


def _memory_total(proc: Path = Path("/proc")) -> str:
    """Installed RAM."""
    value = _meminfo_kbytes(proc, "MemTotal")
    return _human_bytes(value) if value else ""


def _memory_available(proc: Path = Path("/proc")) -> str:
    """RAM the kernel can hand to a new process right now."""
    value = _meminfo_kbytes(proc, "MemAvailable")
    return _human_bytes(value) if value else ""


def _swap_total(proc: Path = Path("/proc")) -> str:
    """Swap space, '' when there is none."""
    value = _meminfo_kbytes(proc, "SwapTotal")
    return _human_bytes(value) if value and value > 0 else ""


def _uevent_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in _read_text(path).splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def _lspci_names(pci_id: str) -> str:
    """``Vendor Device`` from lspci, '' when it is missing or says nothing."""
    program = shutil.which("lspci")
    if not program or not pci_id:
        return ""
    try:
        completed = subprocess.run(
            [program, "-mm", "-d", pci_id],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""
    if completed.returncode != 0:
        return ""
    quoted = re.findall(r'"([^"]*)"', completed.stdout or "")
    if len(quoted) < 3:
        return ""
    vendor, device = quoted[1], quoted[2]
    short = re.search(r"\[([^\]]*)\]", vendor)
    if short:
        vendor = short.group(1).split("/")[0].strip() or vendor
    return f"{vendor} {device}".strip()


def _gpu(sysfs: Path = Path("/sys")) -> str:
    """One line per graphics adapter, '; '-joined; '' when none can be read."""
    drm = sysfs / "class" / "drm"
    try:
        cards = sorted(child.name for child in drm.iterdir() if re.fullmatch(r"card\d+", child.name))
    except OSError:
        return ""
    lines: list[str] = []
    for card in cards:
        fields = _uevent_fields(drm / card / "device" / "uevent")
        pci_id = fields.get("PCI_ID", "")
        if not pci_id:
            continue
        driver = fields.get("DRIVER", "")
        label = _lspci_names(pci_id) or f"PCI {pci_id}"
        line = f"{label} (driver {driver})" if driver else label
        if line not in lines:
            lines.append(line)
    return "; ".join(lines)


def _battery(sysfs: Path = Path("/sys")) -> str:
    """Charge and state of each battery, '' on a machine without one."""
    supplies = sysfs / "class" / "power_supply"
    try:
        names = sorted(child.name for child in supplies.iterdir() if child.name.startswith("BAT"))
    except OSError:
        return ""
    parts: list[str] = []
    for name in names:
        raw = _read_text(supplies / name / "capacity").strip()
        try:
            percent = int(raw)
        except ValueError:
            continue
        status = _read_text(supplies / name / "status").strip()
        parts.append(f"{percent}% ({status.lower()})" if status else f"{percent}%")
    return "; ".join(parts)


def _uptime(proc: Path = Path("/proc")) -> str:
    """How long the machine has been running."""
    fields = _read_text(proc / "uptime").split()
    try:
        seconds = int(float(fields[0]))
    except (IndexError, ValueError):
        return ""
    days, rest = divmod(max(0, seconds), 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes = rest // 60
    if days:
        return f"{_count_unit(days, 'day')} {_count_unit(hours, 'hour')}"
    if hours:
        return f"{_count_unit(hours, 'hour')} {_count_unit(minutes, 'minute')}"
    return _count_unit(minutes, "minute")


def _machine_facts(proc: Path = Path("/proc"), sysfs: Path = Path("/sys")) -> dict[str, str]:
    """Hardware read from procfs and sysfs; a fact that cannot be read is left out."""
    return {
        name: value
        for name, value in (
            ("cpu_model", _cpu_model(proc)),
            ("cpu_cores", _cpu_cores(proc)),
            ("memory_total", _memory_total(proc)),
            ("memory_available", _memory_available(proc)),
            ("swap_total", _swap_total(proc)),
            ("gpu", _gpu(sysfs)),
            ("battery", _battery(sysfs)),
            ("uptime", _uptime(proc)),
        )
        if value
    }


def system_info(workspace: Workspace) -> dict[str, Any]:
    """Read-only snapshot of the machine and the workspace."""
    root = workspace.root
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}".strip(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 0,
        **_machine_facts(),
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP") or "(unknown)",
        "session": (
            "wayland" if os.environ.get("WAYLAND_DISPLAY")
            else "x11" if os.environ.get("DISPLAY")
            else "none/headless"
        ),
        "hostname": platform.node(),
        "workspace": str(root),
    }
    try:
        load = os.getloadavg()
        info["load_average"] = ", ".join(f"{value:.2f}" for value in load)
    except OSError:  # pragma: no cover - non-unix fallback
        pass
    try:
        usage = shutil.disk_usage(root)
        info["disk_total"] = _human_bytes(usage.total)
        info["disk_free"] = _human_bytes(usage.free)
        home = Path.home()
        if home.is_dir() and home.stat().st_dev != root.stat().st_dev:  # the workspace sits on another disk
            usage = shutil.disk_usage(home)
            info["home_disk_total"] = _human_bytes(usage.total)
            info["home_disk_free"] = _human_bytes(usage.free)
    except OSError:  # pragma: no cover - defensive
        pass
    file_count = 0
    total_bytes = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            file_count += 1
            try:
                total_bytes += (Path(current) / name).stat().st_size
            except OSError:
                continue
            if file_count >= 5000:
                break
        if file_count >= 5000:
            break
    info["workspace_files"] = file_count
    info["workspace_size"] = _human_bytes(total_bytes)
    info["local_time"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    info["tools"] = {
        name: bool(shutil.which(binary))
        for name, binary in (
            ("git", "git"), ("xdg-open", "xdg-open"), ("notify", "notify-send"),
            ("clipboard", "wl-copy"), ("screenshot", "grim"),
        )
    }
    if not info["tools"]["clipboard"]:
        info["tools"]["clipboard"] = bool(_which("xclip", "xsel"))
    if not info["tools"]["screenshot"]:
        info["tools"]["screenshot"] = bool(_which("scrot", "gnome-screenshot", "spectacle", "import"))
    return info


# --- open_path -------------------------------------------------------------------


def open_path(workspace: Workspace, target: str) -> dict[str, Any]:
    """Open a workspace file/folder or an http(s) URL in the user's desktop."""
    target = (target or "").strip()
    if not target:
        raise DesktopError("nothing to open")
    opener = _which("xdg-open")
    if opener is None:
        raise DesktopError("xdg-open is not available on this machine")
    if target.startswith(("http://", "https://")):
        label = target
    elif target.startswith(("file://", "javascript:", "data:")):
        raise DesktopError("only http(s) URLs and workspace paths can be opened")
    else:
        resolved = workspace.resolve(target, must_exist=True)
        label = str(resolved)
        target = label
    try:
        subprocess.Popen(
            [opener, target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise DesktopError(f"could not open: {exc.strerror or exc}") from exc
    return {"opened": label}


# --- clipboard ---------------------------------------------------------------------


def _clipboard_backend() -> tuple[str, list[str], list[str]] | None:
    """(name, read argv, write argv) for the first available clipboard tool."""
    if _which("wl-paste") and _which("wl-copy"):
        return "wl-clipboard", ["wl-paste", "--no-newline"], ["wl-copy"]
    if _which("xclip"):
        return (
            "xclip",
            ["xclip", "-selection", "clipboard", "-o"],
            ["xclip", "-selection", "clipboard", "-i"],
        )
    if _which("xsel"):
        return "xsel", ["xsel", "--clipboard", "--output"], ["xsel", "--clipboard", "--input"]
    return None


def clipboard(action: str, text: str | None = None) -> dict[str, Any]:
    """Read or write the desktop clipboard (text only)."""
    backend = _clipboard_backend()
    if backend is None:
        raise DesktopError(
            "no clipboard tool found (install wl-clipboard, xclip or xsel)"
        )
    name, read_argv, write_argv = backend
    if action == "read":
        content = _run(read_argv, timeout=5)
        return {
            "backend": name,
            "text": content[:CLIPBOARD_READ_CAP],
            "truncated": len(content) > CLIPBOARD_READ_CAP,
            "chars": len(content),
        }
    if action == "write":
        payload = text or ""
        if len(payload.encode("utf-8")) > CLIPBOARD_WRITE_CAP:
            raise DesktopError("clipboard text is too large (limit 100 KiB)")
        _run(write_argv, input_text=payload, timeout=5)
        return {"backend": name, "written_chars": len(payload)}
    raise DesktopError(f"unknown clipboard action: {action}")


# --- notify --------------------------------------------------------------------------


def notify(title: str, body: str = "") -> dict[str, Any]:
    """Show a desktop notification."""
    sender = _which("notify-send")
    if sender is None:
        raise DesktopError("notify-send is not available on this machine")
    title = (title or "Kiwibro").strip()[:120] or "Kiwibro"
    body = (body or "").strip()[:500]
    _run([sender, "--app-name=Kiwibro", title, body], timeout=5)
    return {"title": title, "shown": True}


# --- screenshot -------------------------------------------------------------------


def screenshot(workspace: Workspace, filename: str | None = None) -> dict[str, Any]:
    """Capture the screen into ``screenshots/`` inside the workspace."""
    if filename:
        relative = str(filename).strip()
        if not relative.lower().endswith(".png"):
            relative += ".png"
    else:
        relative = f"screenshots/screenshot-{time.strftime('%Y%m%d-%H%M%S')}.png"
    target = workspace.resolve(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target_text = str(target)
    backends: list[tuple[str, list[str]]] = []
    if _which("grim"):
        backends.append(("grim", ["grim", target_text]))
    if _which("scrot"):
        backends.append(("scrot", ["scrot", "--overwrite", target_text]))
    if _which("gnome-screenshot"):
        backends.append(("gnome-screenshot", ["gnome-screenshot", f"--file={target_text}"]))
    if _which("spectacle"):
        backends.append(("spectacle", ["spectacle", "--background", "--nonotify", f"--output={target_text}"]))
    if _which("import"):
        backends.append(("imagemagick", ["import", "-window", "root", target_text]))
    if not backends:
        raise DesktopError(
            "no screenshot tool found (tried grim, scrot, gnome-screenshot, spectacle, import)"
        )
    errors: list[str] = []
    for name, argv in backends:
        try:
            _run(argv, timeout=SCREENSHOT_TIMEOUT_S)
        except DesktopError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if target.exists() and target.stat().st_size > 0:
            return {
                "path": workspace.public(target),
                "bytes": int(target.stat().st_size),
                "backend": name,
            }
        errors.append(f"{name}: produced no file")
    raise DesktopError("screenshot failed: " + "; ".join(errors)[:300])


__all__ = [
    "DesktopError",
    "SandboxError",
    "clipboard",
    "notify",
    "open_path",
    "screenshot",
    "system_info",
]

