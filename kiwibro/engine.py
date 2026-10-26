"""Direct engine: serve LM Studio's downloaded models without the LM Studio app.

LM Studio itself runs every GGUF model through a bundled ``llama-server``
(llama.cpp) behind its own API. The direct engine skips that layer: it finds
the models LM Studio downloaded (both the native and the Flatpak install),
picks a llama.cpp runtime (LM Studio's own bundled one, a custom build from
Settings or ``KIWIBRO_LLAMA_SERVER``, or ``llama-server`` on PATH) and runs it
as a child process on a private localhost port. It speaks the same
OpenAI-compatible protocol, so :class:`kiwibro.llm.LMStudioClient` uses it
unchanged once ``engine`` is set to ``direct``.

Why it is faster on a laptop CPU (measured, Ryzen 5 7530U, spark-x2.6-4b
IQ2_M, a fresh 3.9k-token prompt, same load on both): LM Studio launched the
model with 4 threads and flash attention off and prefilled at 27–34 tok/s;
the direct engine uses every physical core and prefilled at ~39 tok/s. It
also turns on what LM Studio leaves off: KV chunk reuse (``--cache-reuse``)
for prompts whose middle changed, a unified KV pool shared by the slots, and
host-RAM prompt caching sized to free memory, so switching between chats
restores a cached prefix instead of prefilling it again. The LM Studio app
does not have to run at all, which frees its memory too.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import logging
import os
import queue
import re
import secrets
import shutil
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import requests

from . import gguf

log = logging.getLogger(__name__)

LMSTUDIO_ROOTS = ("~/.lmstudio", "~/.var/app/ai.lmstudio.lm-studio/.lmstudio")  # native, Flatpak
EXTRA_MODEL_DIRS = ("~/.cache/lm-studio/models",)  # pre-0.3 LM Studio
DEFAULT_CONTEXT = 32768
DEFAULT_PARALLEL = 4  # one slot for the chat, others for background calls (one shared KV pool)
MAX_RESIDENT = 2  # chat model + one helper; a third evicts the least recently used idle one
START_TIMEOUT_S = 300.0
RUNTIME_TRIES = 3  # runtimes tried in order when one exits while loading
SCAN_TTL_S = 30.0
CACHE_REUSE_CHUNK = 256
SPEC_TYPE = "ngram-mod"
LMSTUDIO_ONLY_KEYS = frozenset({"ttl", "draft_model"})
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_QUANT = re.compile(r"(?:^|[-_.])((?:I?Q\d(?:_[A-Z0-9]+)*)|F16|BF16|F32|MXFP4)(?:$|[-_.])", re.IGNORECASE)
_PR_SET_PDEATHSIG = 1


class EngineError(Exception):
    """A calm, user-facing reason the direct engine cannot serve a request."""


class _Crashed(EngineError):
    """The runtime exited before serving: the next runtime may still work."""


# --- discovery: models ------------------------------------------------------------


@dataclass
class LocalModel:
    id: str
    path: str
    size_bytes: int
    publisher: str
    repo: str
    arch: str = ""
    name: str = ""
    params: str = ""
    quantization: str = ""
    context_length: int | None = None
    tools: bool = False
    thinking: bool = False
    thinking_toggle: bool = False
    embeddings: bool = False
    projector: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        if self.embeddings:
            return "embeddings"
        return "vlm" if self.projector else "llm"


def lmstudio_roots() -> list[Path]:
    return [Path(root).expanduser() for root in LMSTUDIO_ROOTS if Path(root).expanduser().is_dir()]


def _lmstudio_downloads(root: Path) -> Path | None:
    """LM Studio's configured models folder. The Flatpak build records paths as it sees
    them inside its sandbox (``~/.lmstudio/...``), which is ``root`` from outside."""
    try:
        folder = json.loads((root / "settings.json").read_text(encoding="utf-8")).get("downloadsFolder")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(folder, str) or not folder:
        return None
    path = Path(folder).expanduser()
    inner = Path.home() / ".lmstudio"
    if not path.is_dir() and path.is_relative_to(inner):
        path = root / path.relative_to(inner)
    return path


def model_dirs(custom: str | None = None) -> list[Path]:
    """Folders to scan, in priority order (first wins when two hold the same id)."""
    candidates: list[Path] = []
    for raw in (custom, *os.environ.get("KIWIBRO_MODELS_DIR", "").split(os.pathsep)):
        if raw and raw.strip():
            candidates.append(Path(raw.strip()).expanduser())
    for root in lmstudio_roots():
        downloads = _lmstudio_downloads(root)
        if downloads is not None:
            candidates.append(downloads)
        candidates.append(root / "models")
    candidates += [Path(extra).expanduser() for extra in EXTRA_MODEL_DIRS]
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.is_dir() and str(resolved) not in seen:
            seen.add(str(resolved))
            out.append(resolved)
    return out


def _strip_gguf(name: str) -> str:
    return re.sub(r"[-_.]gguf$", "", name, flags=re.IGNORECASE)


def _quant_from_name(filename: str) -> str:
    stem = _SHARD.sub("", filename)
    stem = re.sub(r"\.gguf$", "", stem, flags=re.IGNORECASE)
    matches = _QUANT.findall(stem)
    return matches[-1].upper() if matches else ""


def _gguf_files(directory: Path, depth: int = 3) -> Iterator[Path]:
    """GGUF files up to ``depth`` folders down (publisher/repo/file), hidden folders skipped."""
    base = len(directory.parts)
    for current, dirs, files in os.walk(directory, followlinks=True):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".")) if len(Path(current).parts) - base < depth else []
        for name in sorted(files):
            if name.lower().endswith(".gguf"):
                yield Path(current) / name


class ModelScanner:
    """Turns model folders into :class:`LocalModel` records; header reads are cached per file."""

    def __init__(self) -> None:
        self._meta: dict[tuple[str, int, int], dict[str, Any]] = {}

    def _summary(self, path: Path) -> dict[str, Any] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (str(path), stat.st_size, int(stat.st_mtime))
        if key not in self._meta:
            try:
                self._meta[key] = gguf.summarize(gguf.read_metadata(path))
            except (OSError, gguf.GGUFError) as exc:
                log.info("skipping %s: %s", path, exc)
                self._meta[key] = {}
        return self._meta[key] or None

    def scan(self, directories: list[Path]) -> list[LocalModel]:
        models: dict[str, LocalModel] = {}
        for directory in directories:
            groups: dict[Path, list[Path]] = {}
            for path in _gguf_files(directory):
                groups.setdefault(path.parent, []).append(path)
            for folder, files in groups.items():
                for model in self._folder(directory, folder, files):
                    if model.id in models:
                        continue  # an earlier folder already has this id
                    models[model.id] = model
        return list(models.values())

    def _folder(self, root: Path, folder: Path, files: list[Path]) -> list[LocalModel]:
        projectors: list[Path] = []
        weights: list[tuple[Path, int, dict[str, Any]]] = []
        for path in files:
            shard = _SHARD.search(path.name)
            if shard and shard.group(1) != "00001":
                continue  # later shards load with the first one
            summary = self._summary(path)
            if summary is None:
                continue
            if summary.get("projector") or "mmproj" in path.name.lower():
                projectors.append(path)
                continue
            size = path.stat().st_size
            if shard:
                pattern = _SHARD.sub("", path.name)
                size = sum(p.stat().st_size for p in files if _SHARD.search(p.name) and _SHARD.sub("", p.name) == pattern)
            weights.append((path, size, summary))
        parts = folder.relative_to(root).parts if folder != root else ()
        publisher = parts[0] if len(parts) >= 2 else ""
        out: list[LocalModel] = []
        for path, size, summary in weights:
            repo = parts[-1] if parts else re.sub(r"\.gguf$", "", _SHARD.sub("", path.name), flags=re.IGNORECASE)
            base = _strip_gguf(repo).lower()
            model_id = f"{publisher.lower()}/{base}" if publisher else base
            quant = summary.get("quantization") or _quant_from_name(path.name)
            if len(weights) > 1:  # several quantisations side by side: LM Studio's id@quant
                model_id = f"{model_id}@{(quant or path.stem).lower()}"
            stem = _SHARD.sub("", path.name)
            stem = re.sub(r"\.gguf$", "", stem, flags=re.IGNORECASE).lower()
            alias_set = {model_id, base, re.sub(r"-instruct$", "", base), stem,
                         f"{publisher.lower()}/{stem}" if publisher else stem}
            out.append(LocalModel(
                id=model_id, path=str(path), size_bytes=size, publisher=publisher, repo=repo,
                arch=summary.get("arch", ""), name=summary.get("name", ""),
                params=summary.get("size_label", ""), quantization=quant,
                context_length=summary.get("context_length"), tools=bool(summary.get("tools")),
                thinking=bool(summary.get("thinking")), thinking_toggle=bool(summary.get("thinking_toggle")),
                embeddings=bool(summary.get("embeddings")),
                projector=str(projectors[0]) if projectors and not summary.get("embeddings") else None,
                aliases=tuple(sorted(a for a in alias_set if a)),
            ))
        return out


# --- discovery: runtimes ------------------------------------------------------------


@dataclass(frozen=True)
class Runtime:
    path: str  # the llama-server executable
    version: str
    kind: str  # cpu | vulkan | cuda | rocm | unknown
    source: str  # custom | lmstudio | path

    @property
    def label(self) -> str:
        where = {"lmstudio": "LM Studio's llama.cpp", "custom": "custom llama.cpp", "path": "llama.cpp"}[self.source]
        return f"{where} {self.version} ({self.kind.upper() if self.kind != 'unknown' else 'build'})".strip()


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", version)[:4]) or (0,)


def _kind(name: str) -> str:
    lowered = name.lower()
    for kind in ("cuda", "vulkan", "rocm", "metal"):
        if kind in lowered:
            return kind
    return "cpu"


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def lmstudio_runtimes() -> tuple[list[Runtime], set[tuple[str, str]]]:
    """LM Studio's bundled llama.cpp engines, plus the (name, version) pairs it prefers."""
    found: list[Runtime] = []
    preferred: set[tuple[str, str]] = set()
    for root in lmstudio_roots():
        try:
            prefs = json.loads((root / ".internal" / "backend-preferences-v1.json").read_text(encoding="utf-8"))
            preferred |= {(str(p.get("name")), str(p.get("version"))) for p in prefs if isinstance(p, dict)}
        except (OSError, ValueError, TypeError):
            pass
        backends = root / "extensions" / "backends"
        if not backends.is_dir():
            continue
        for engine in sorted(backends.iterdir()):
            server = engine / "llama-server"
            if not engine.name.startswith("llama.cpp-") or not _executable(server):
                continue
            try:
                manifest = json.loads((engine / "backend-manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
            name = str(manifest.get("name") or engine.name.rsplit("-", 1)[0])
            version = str(manifest.get("version") or engine.name.rsplit("-", 1)[-1])
            found.append(Runtime(str(server), version, _kind(name), "lmstudio"))
            if (name, version) in preferred:
                preferred.add(("path", str(server)))
    return found, preferred


def _has_nvidia() -> bool:
    return Path("/dev/nvidia0").exists() or Path("/proc/driver/nvidia").exists()


def find_runtimes(custom: str | None = None) -> list[Runtime]:
    """Usable llama.cpp servers, best first: a custom build, the engine LM Studio is set to
    use, LM Studio's newest CPU engine, its other engines, then ``llama-server`` on PATH."""
    out: list[Runtime] = []
    for raw in (custom, os.environ.get("KIWIBRO_LLAMA_SERVER")):
        if raw and _executable(Path(raw).expanduser()):
            out.append(Runtime(str(Path(raw).expanduser()), "custom", "unknown", "custom"))
    bundled, preferred = lmstudio_runtimes()
    usable = [r for r in bundled if r.kind != "cuda" or _has_nvidia()]

    def rank(runtime: Runtime) -> tuple[int, tuple[int, ...], int]:
        chosen = ("path", runtime.path) in preferred
        return (0 if chosen else 1, tuple(-n for n in _version_key(runtime.version)), 0 if runtime.kind == "cpu" else 1)

    out += sorted(usable, key=rank)
    system = shutil.which("llama-server")
    if system:
        out.append(Runtime(system, "system", "unknown", "path"))
    seen: set[str] = set()
    return [r for r in out if not (r.path in seen or seen.add(r.path))]


# --- host facts ---------------------------------------------------------------------


def physical_cores() -> int:
    """Physical cores this process may use: SMT siblings slow llama.cpp's matrix work down."""
    try:
        allowed = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = os.cpu_count() or 4
    cores: set[tuple[str, str]] = set()
    physical = core = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "physical id":
                    physical = value
                elif key == "core id":
                    core = value
                elif not key and core:
                    cores.add((physical, core))
                    physical = core = ""
        if core:
            cores.add((physical, core))
    except OSError:
        pass
    count = len(cores) or max(1, allowed // 2)
    return max(1, min(count, allowed))


def available_memory_mb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# --- payload -------------------------------------------------------------------------


def adapt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """An LM Studio request body as llama-server wants it.

    ``reasoning_effort: "none"`` is understood natively (thinking off). Other
    effort words are template-specific (spark-x2.6 raises on "medium"), so they
    become a plain thinking-on switch. LM Studio's own knobs are dropped.
    """
    out = {k: v for k, v in payload.items() if k not in LMSTUDIO_ONLY_KEYS}
    out.setdefault("cache_prompt", True)
    effort = out.get("reasoning_effort")
    kwargs = dict(out.get("chat_template_kwargs") or {})
    if effort == "none":
        kwargs.setdefault("enable_thinking", False)
    elif effort is not None:
        out.pop("reasoning_effort")
        kwargs.setdefault("enable_thinking", True)
    if kwargs:
        out["chat_template_kwargs"] = kwargs
    return out


# --- processes -----------------------------------------------------------------------


class _Spawner:
    """Starts child processes from one long-lived thread.

    ``PR_SET_PDEATHSIG`` fires when the *thread* that forked the child exits, so
    a server started from a short-lived request thread would be killed with it.
    From here, servers die with Kiwibro (including its self-update ``execv``)."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any], queue.Queue[Any]]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def run(self, fn: Callable[[], Any]) -> Any:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="kiwibro-engine-spawner", daemon=True)
                self._thread.start()
        reply: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._jobs.put((fn, reply))
        ok, value = reply.get()
        if not ok:
            raise value
        return value

    def _loop(self) -> None:
        while True:
            fn, reply = self._jobs.get()
            try:
                reply.put((True, fn()))
            except BaseException as exc:  # handed back to the caller
                reply.put((False, exc))


_SPAWNER = _Spawner()


def _die_with_parent() -> None:  # pragma: no cover - runs in the forked child
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except (OSError, AttributeError):
        pass


@dataclass
class Server:
    model: LocalModel
    runtime: Runtime
    process: subprocess.Popen[bytes]
    port: int
    context_length: int
    parallel: int
    threads: int
    log_path: str
    started_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.monotonic)
    in_flight: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.process.poll() is None

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model.id, "pid": self.process.pid, "port": self.port,
            "context_length": self.context_length, "parallel": self.parallel, "threads": self.threads,
            "runtime": self.runtime.label, "started_at": int(self.started_at), "busy": self.in_flight > 0,
        }


def _flags_of(runtime: Runtime) -> frozenset[str]:
    """The long options a runtime's ``--help`` lists (older builds lack newer flags)."""
    try:
        result = subprocess.run(
            [runtime.path, "--help"], capture_output=True, timeout=20, env=_runtime_env(runtime),
            cwd=str(Path(runtime.path).parent), check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    text = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    return frozenset(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", text))


def _runtime_env(runtime: Runtime) -> dict[str, str]:
    env = dict(os.environ)
    folder = str(Path(runtime.path).parent)
    env["LD_LIBRARY_PATH"] = folder + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def server_args(
    runtime: Runtime, flags: frozenset[str], model: LocalModel, *, port: int, api_key: str,
    context_length: int, parallel: int, threads: int, cache_ram_mb: int | None, speculative: bool = True,
) -> list[str]:
    """The llama-server command line for one model (flags the runtime lacks are left out)."""
    def has(flag: str) -> bool:
        return not flags or flag in flags  # unknown help output: assume a current build

    args = [runtime.path, "--model", model.path, "--host", "127.0.0.1", "--port", str(port),
            "--alias", model.id, "--ctx-size", str(context_length), "--threads", str(threads)]
    if has("--threads-batch"):
        args += ["--threads-batch", str(threads)]
    if has("--api-key"):
        args += ["--api-key", api_key]
    if model.embeddings:
        return args + ["--embeddings", "--parallel", "1"]
    args += ["--parallel", str(parallel)]
    if has("--kv-unified") and parallel > 1:
        args.append("--kv-unified")  # one KV pool: a long chat may use most of the window
    if has("--jinja"):
        args.append("--jinja")  # the model's own template: tool calls and the thinking switch
    if has("--flash-attn"):
        args += ["--flash-attn", "auto"]
    if has("--reasoning-format"):
        args += ["--reasoning-format", "deepseek"]  # thinking arrives as reasoning_content
    if has("--cache-reuse"):
        args += ["--cache-reuse", str(CACHE_REUSE_CHUNK)]
    if cache_ram_mb is not None and has("--cache-ram"):
        args += ["--cache-ram", str(cache_ram_mb)]
    if has("--no-webui"):
        args.append("--no-webui")
    if has("--metrics"):
        args.append("--metrics")
    if model.projector and has("--mmproj"):
        args += ["--mmproj", model.projector]
    elif speculative and has("--spec-type"):
        # Drafts from n-grams already in the context: no draft model, idle on free text,
        # and ~45% faster decode when a reply repeats what it read (files, code edits).
        args += ["--spec-type", SPEC_TYPE]
    return args


# --- the engine -----------------------------------------------------------------------


class DirectEngine:
    """Runs llama-server children for LM Studio's models; thread-safe."""

    def __init__(self, settings_provider: Callable[[], dict[str, Any]], data_dir: str | Path | None = None) -> None:
        self._settings = settings_provider
        self._data_dir = Path(data_dir).expanduser() if data_dir else None
        self.api_key = secrets.token_urlsafe(24)
        self._scanner = ModelScanner()
        self._models: list[LocalModel] = []
        self._scanned_at = 0.0
        self._scan_key: tuple[Any, ...] = ()
        self._servers: dict[str, Server] = {}
        self._starting: dict[str, threading.Event] = {}
        self._flags: dict[str, frozenset[str]] = {}
        self._lock = threading.RLock()
        self._last_error: str | None = None
        self._reap_stale()
        atexit.register(self.shutdown)

    # --- settings ---

    def _custom_runtime(self) -> str | None:
        return str(self._settings().get("direct_runtime") or "").strip() or None

    def _custom_models_dir(self) -> str | None:
        return str(self._settings().get("direct_models_dir") or "").strip() or None

    # --- catalog ---

    def models(self, force: bool = False) -> list[LocalModel]:
        key = (self._custom_models_dir(), os.environ.get("KIWIBRO_MODELS_DIR"))
        with self._lock:
            fresh = time.monotonic() - self._scanned_at < SCAN_TTL_S and key == self._scan_key
            if fresh and not force:
                return list(self._models)
        found = self._scanner.scan(model_dirs(self._custom_models_dir()))
        with self._lock:
            self._models, self._scanned_at, self._scan_key = found, time.monotonic(), key
            return list(found)

    def find(self, model_id: str | None) -> LocalModel | None:
        if not model_id:
            return None
        wanted = str(model_id).strip().lower()
        models = self.models()
        for model in models:
            if model.id == wanted:
                return model
        for model in models:
            if wanted in model.aliases:
                return model
        # LM Studio names some models after their maker ("liquid/lfm2.5-1.2b" for
        # lmstudio-community's upload): the name part alone still identifies it.
        name = wanted.rsplit("/", 1)[-1]
        matches = [model for model in models if name in model.aliases]
        return matches[0] if len(matches) == 1 else None

    def canonical(self, model_id: str | None) -> str | None:
        """The direct id for an id from either engine (LM Studio's ids usually match)."""
        model = self.find(model_id)
        return model.id if model else None

    def runtime(self) -> Runtime | None:
        runtimes = find_runtimes(self._custom_runtime())
        return runtimes[0] if runtimes else None

    def raw_entries(self) -> list[dict[str, Any]]:
        """Models in LM Studio's ``/api/v0/models`` shape (see llm.normalize_model)."""
        with self._lock:
            running = {mid: s for mid, s in self._servers.items() if s.alive()}
        entries = []
        for model in self.models():
            server = running.get(model.id)
            entry: dict[str, Any] = {
                "id": model.id, "object": "model", "type": model.kind, "publisher": model.publisher,
                "arch": model.arch, "compatibility_type": "gguf", "quantization": model.quantization,
                "state": "loaded" if server else "not-loaded",
                "max_context_length": model.context_length or DEFAULT_CONTEXT,
                "capabilities": ["tool_use"] if model.tools else [],
            }
            if server:
                entry["loaded_context_length"] = server.context_length
            entries.append(entry)
        return entries

    def native_entries(self) -> list[dict[str, Any]]:
        """Models in LM Studio's ``/api/v1/models`` shape (see llm.merge_native_metadata)."""
        with self._lock:
            running = {mid: s for mid, s in self._servers.items() if s.alive()}
        entries = []
        for model in self.models():
            capabilities: dict[str, Any] = {"vision": bool(model.projector), "trained_for_tool_use": model.tools}
            if model.thinking:
                options = ["off", "on"] if model.thinking_toggle else ["on"]
                capabilities["reasoning"] = {"allowed_options": options, "default": "on"}
            entry: dict[str, Any] = {
                "key": model.id, "display_name": model.name or model.repo, "architecture": model.arch,
                "params_string": model.params, "size_bytes": model.size_bytes,
                "quantization": {"name": model.quantization}, "capabilities": capabilities,
                "path": model.path, "loaded_instances": [],
            }
            server = running.get(model.id)
            if server:
                entry["loaded_instances"] = [{"id": model.id, "config": {
                    "context_length": server.context_length, "parallel": server.parallel, "flash_attention": True,
                }}]
            entries.append(entry)
        return entries

    # --- serving ---

    def loaded(self, model_id: str) -> Server | None:
        model = self.find(model_id)
        with self._lock:
            server = self._servers.get(model.id) if model else None
            return server if server and server.alive() else None

    def acquire(self, model_id: str) -> Server:
        """A running server for ``model_id``, already marked busy (see :meth:`release`).

        Marking happens under the lock that eviction checks, so a server handed out
        here can never be stopped before its request is done."""
        while True:
            server = self.ensure(model_id)
            with self._lock:
                if server.alive() and self._servers.get(server.model.id) is server:
                    server.in_flight += 1
                    server.last_used = time.monotonic()
                    return server
            # stopped between ensure() and here (evicted or crashed): start it again

    def release(self, server: Server) -> None:
        with self._lock:
            server.in_flight = max(0, server.in_flight - 1)
            server.last_used = time.monotonic()

    @contextmanager
    def lease(self, model_id: str) -> Iterator[str]:
        """The URL serving ``model_id`` (started just in time, like LM Studio's JIT load), held busy."""
        server = self.acquire(model_id)
        try:
            yield server.url
        finally:
            self.release(server)

    def ensure(self, model_id: str, context_length: int | None = None, parallel: int | None = None) -> Server:
        model = self.find(model_id)
        if model is None:
            raise EngineError(f"{model_id} is not among the models LM Studio downloaded.")
        while True:
            with self._lock:
                server = self._servers.get(model.id)
                wanted_ctx = self._window(model, context_length or (server.context_length if server else None))
                wanted_par = 1 if model.embeddings else int(parallel or (server.parallel if server else DEFAULT_PARALLEL))
                if server and server.alive() and (context_length is None or server.context_length == wanted_ctx) \
                        and (parallel is None or server.parallel == wanted_par):
                    return server
                pending = self._starting.get(model.id)
                if pending is None:
                    pending = self._starting[model.id] = threading.Event()
                    break
            pending.wait(START_TIMEOUT_S + 5)  # another thread is starting it; then re-check
        try:
            if server is not None:
                self._stop(model.id)
            return self._start(model, wanted_ctx, wanted_par)
        finally:
            with self._lock:
                self._starting.pop(model.id, None)
            pending.set()

    def _window(self, model: LocalModel, requested: int | None) -> int:
        trained = model.context_length or DEFAULT_CONTEXT
        return max(512, min(int(requested or DEFAULT_CONTEXT), trained))

    def _cache_ram_mb(self) -> int | None:
        """Host-RAM prompt cache: a quarter of free memory, 1–8 GiB (llama.cpp's default is 8)."""
        free = available_memory_mb()
        return None if free is None else max(1024, min(8192, free // 4))

    def _make_room(self, model: LocalModel) -> None:
        """Stop idle servers until this model fits: least recently used first."""
        need_mb = model.size_bytes // (1024 * 1024) + 1024
        while True:
            with self._lock:
                idle = sorted((s for mid, s in self._servers.items() if mid != model.id and s.in_flight == 0),
                              key=lambda s: s.last_used)
                crowded = len(self._servers) >= MAX_RESIDENT
            free = available_memory_mb()
            if not idle or not (crowded or (free is not None and free < need_mb)):
                return
            log.info("stopping %s to make room for %s", idle[0].model.id, model.id)
            if not self._stop(idle[0].model.id, only_idle=True):
                return  # it just took a request: leave it, the new model loads beside it

    def _start(self, model: LocalModel, context_length: int, parallel: int) -> Server:
        runtimes = find_runtimes(self._custom_runtime())
        if not runtimes:
            raise EngineError(
                "No llama.cpp runtime was found. Install LM Studio's llama.cpp engine once, "
                "or set a llama-server path in Settings."
            )
        self._make_room(model)
        last: EngineError | None = None
        attempts = [(runtime, spec) for runtime in runtimes[:RUNTIME_TRIES] for spec in (True, False)]
        for runtime, speculative in attempts:
            try:
                server = self._launch(runtime, model, context_length, parallel, speculative)
            except _Crashed as exc:  # e.g. an older engine that lacks this architecture
                log.info("direct engine: %s could not run %s%s: %s", runtime.label, model.id,
                         "" if speculative else " (no speculative decoding)", exc)
                last = exc
                continue
            except EngineError as exc:
                last = exc
                break
            with self._lock:
                self._servers[model.id] = server
            self._last_error = None
            return server
        self._last_error = str(last)
        raise last or EngineError(f"Could not start {model.id}.")

    def _launch(
        self, runtime: Runtime, model: LocalModel, context_length: int, parallel: int, speculative: bool = True,
    ) -> Server:
        if runtime.path not in self._flags:
            self._flags[runtime.path] = _flags_of(runtime)
        threads = int(self._settings().get("direct_threads") or 0) or physical_cores()
        port = _free_port()
        args = server_args(
            runtime, self._flags[runtime.path], model, port=port, api_key=self.api_key,
            context_length=context_length, parallel=parallel, threads=threads, cache_ram_mb=self._cache_ram_mb(),
            speculative=speculative,
        )
        log_path = self._log_path(model)
        log.info("direct engine: starting %s (%s, ctx %d, %d slots, %d threads)",
                 model.id, runtime.label, context_length, parallel, threads)

        def spawn() -> subprocess.Popen[bytes]:
            handle = open(log_path, "ab")  # noqa: SIM115 - the child owns it from here
            try:
                return subprocess.Popen(
                    args, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    env=_runtime_env(runtime), cwd=str(Path(runtime.path).parent),
                    preexec_fn=_die_with_parent, close_fds=True,
                )
            finally:
                handle.close()

        try:
            process = _SPAWNER.run(spawn)
        except OSError as exc:
            raise _Crashed(f"Could not start llama.cpp: {exc}") from exc
        server = Server(model, runtime, process, port, context_length, parallel, threads, log_path)
        self._write_pid(server)
        try:
            self._wait_ready(server)
        except EngineError:
            self._kill(server)
            raise
        log.info("direct engine: %s ready on port %d in %.1fs", model.id, port, time.time() - server.started_at)
        return server

    def _wait_ready(self, server: Server) -> None:
        deadline = time.monotonic() + START_TIMEOUT_S
        headers = {"Authorization": f"Bearer {self.api_key}"}
        while time.monotonic() < deadline:
            if not server.alive():
                raise _Crashed(f"llama.cpp stopped while loading {server.model.id}: {self._log_tail(server)}")
            try:
                response = requests.get(f"{server.url}/health", timeout=2, headers=headers)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise EngineError(f"llama.cpp did not finish loading {server.model.id} in time.")

    @staticmethod
    def _log_tail(server: Server) -> str:
        try:
            lines = Path(server.log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "no log"
        errors = [line for line in lines[-80:] if re.search(r"\b(error|failed|unknown|invalid)\b", line, re.I)]
        return (errors[-1] if errors else (lines[-1] if lines else "no output")).strip()[:300]

    def load(self, model_id: str, context_length: int | None = None, parallel: int | None = None) -> bool:
        try:
            self.ensure(model_id, context_length=context_length or DEFAULT_CONTEXT, parallel=parallel or DEFAULT_PARALLEL)
            return True
        except EngineError as exc:
            log.info("direct engine could not load %s: %s", model_id, exc)
            self._last_error = str(exc)
            return False

    def unload(self, model_id: str) -> bool:
        """Stop a model's server; False when it is not running or still answering a request."""
        model = self.find(model_id)
        return self._stop(model.id if model else str(model_id), only_idle=True)

    def _stop(self, model_id: str, *, only_idle: bool = False) -> bool:
        with self._lock:
            server = self._servers.get(model_id)
            if server is None or (only_idle and server.in_flight > 0):
                return False
            del self._servers[model_id]
        if server is None:
            return False
        self._kill(server)
        return True

    def _kill(self, server: Server) -> None:
        if server.alive():
            server.process.terminate()
            try:
                server.process.wait(10)
            except subprocess.TimeoutExpired:
                server.process.kill()
                server.process.wait(5)
        self._remove_pid(server)

    def shutdown(self) -> None:
        with self._lock:
            ids = list(self._servers)
        for model_id in ids:
            self._stop(model_id)

    # --- bookkeeping ---

    def _run_dir(self) -> Path | None:
        if self._data_dir is None:
            return None
        path = self._data_dir / "engine"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return path

    def _log_path(self, model: LocalModel) -> str:
        folder = self._run_dir()
        name = re.sub(r"[^a-z0-9.@-]+", "_", model.id) + ".log"
        return str(folder / name) if folder else os.devnull

    def _write_pid(self, server: Server) -> None:
        folder = self._run_dir()
        if folder:
            try:
                (folder / f"{server.process.pid}.pid").write_text(server.runtime.path, encoding="utf-8")
            except OSError:
                pass

    def _remove_pid(self, server: Server) -> None:
        folder = self._run_dir()
        if folder:
            (folder / f"{server.process.pid}.pid").unlink(missing_ok=True)

    def _reap_stale(self) -> None:
        """Stop servers an earlier Kiwibro left behind (killed before it could clean up)."""
        folder = self._run_dir()
        if folder is None:
            return
        for pidfile in folder.glob("*.pid"):
            try:
                pid = int(pidfile.stem)
                exe = pidfile.read_text(encoding="utf-8").strip()
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                if cmdline and cmdline[0].decode(errors="replace") == exe:
                    os.kill(pid, signal.SIGTERM)
                    log.info("stopped a llama.cpp server (pid %d) left by an earlier run", pid)
            except (OSError, ValueError):
                pass
            pidfile.unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        runtimes = find_runtimes(self._custom_runtime())
        with self._lock:
            servers = [s.describe() for s in self._servers.values() if s.alive()]
        models = self.models()
        dirs = [str(p) for p in model_dirs(self._custom_models_dir())]
        used = [d for d in dirs if any(m.path.startswith(d + os.sep) for m in models)]
        return {
            "available": bool(runtimes) and bool(models),
            "runtime": ({"path": runtimes[0].path, "label": runtimes[0].label, "kind": runtimes[0].kind,
                         "source": runtimes[0].source} if runtimes else None),
            "runtimes": [{"path": r.path, "label": r.label} for r in runtimes],
            "model_dirs": used + [d for d in dirs if d not in used],  # folders holding models first
            "models_count": sum(1 for m in models if not m.embeddings),
            "threads": int(self._settings().get("direct_threads") or 0) or physical_cores(),
            "servers": servers,
            "error": self._last_error,
        }
