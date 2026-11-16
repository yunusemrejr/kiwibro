"""Kiwi Mind 3 on llama.cpp: the server that holds the pack and the client that speaks to it.

A 3.x pack is a GGUF trunk with small files around it: a route layer, one layer per skill and the vision
layers. The route layer and the skill layers are activated LoRA: each switches on at its own token of the
prompt (``<think>`` for the route, ``</think>`` for a skill) and leaves everything before it as the trunk read
it, so llama-server keeps what it has read when the skill changes. Every reply therefore takes two requests:
the route (a few tokens, held to the route's grammar), then the reply under the routed skill's layer.

No request is sent with every layer off: llama-server would drop what it has read.
"""

from __future__ import annotations

import json
import logging
import math
import secrets
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import requests

from . import render3
from .client import HostGone, MindClient, MindRequestError
from .host import STOP_WAIT_S, HostError, MindHost, _Running
from .pack import TRUNK3, Pack

log = logging.getLogger(__name__)

CHAT_SLOTS = 2  # conversations kept read at once; one more slot serves utility requests
UTIL_SLOT = "util"  # conductor.UTIL_SLOT
READY_TIMEOUT_S = 300.0
ROUTE = "route"
ROUTE_TOKENS = 16  # a route is three or four; the grammar ends it
ROUTE_PROBS = 12  # candidates returned for the light-or-heavy word
CTX_CHECKPOINTS = 8
DEFAULT_CTX = 8192
NO_RUNTIME = "Kiwi Mind needs llama.cpp, which was not found on this computer."


@dataclass(frozen=True)
class LlamaRuntime:
    """The llama-server build to run, as Kiwibro's engine chose it."""

    path: str
    flags: frozenset[str]  # the options its --help lists (empty: not known, taken as a current build)
    offload: bool | None  # True: on the GPU; False: a GPU build kept on the CPU; None: a CPU build
    env: dict[str, str]


def layers(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """(name, file) of the pack's layers in the order the server loads them: the route layer, then the skills."""
    route = manifest.get("route")
    file = str((route if isinstance(route, dict) else {}).get("file") or "route.gguf")
    return [(ROUTE, file)] + [(str(name), f"skills/{name}.gguf") for name in manifest.get("skills") or []]


def server_args(runtime: LlamaRuntime, pack: Pack, manifest: dict[str, Any], *, port: int, threads: int) -> list[str]:
    """The llama-server command line for a pack (options the build lacks are left out)."""
    def has(flag: str) -> bool:
        return not runtime.flags or flag in runtime.flags

    args = [runtime.path, "--model", str(pack.path(TRUNK3)), "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(int(manifest.get("ctx") or DEFAULT_CTX)), "--threads", str(threads),
            "--parallel", str(CHAT_SLOTS + 1),
            "--special"]  # tool-call tags are tokens of their own in this model and have to be printed
    if has("--threads-batch"):
        args += ["--threads-batch", str(threads)]
    if runtime.offload and has("--gpu-layers"):
        args += ["--gpu-layers", "auto" if has("--fit") else "999"]
    elif runtime.offload is False:
        args += ["--device", "none"] if has("--device") else ["--gpu-layers", "0"]
    for _name, file in layers(manifest):
        args += ["--lora", str(pack.path(file))]
    args.append("--lora-init-without-apply")  # every request says which layer is on
    for flag in ("--kv-unified", "--no-cache-idle-slots", "--no-webui"):
        if has(flag):
            args.append(flag)
    if has("--flash-attn"):
        args += ["--flash-attn", "auto"]
    if has("--ctx-checkpoints"):
        args += ["--ctx-checkpoints", str(CTX_CHECKPOINTS)]
    vision = manifest.get("vision")
    if isinstance(vision, dict) and vision.get("file") and has("--mmproj"):
        args += ["--mmproj", str(pack.path(str(vision["file"])))]
    return args


class LlamaHost(MindHost):
    """Starts, watches and stops the llama-server that holds a 3.x pack. It dies with Kiwibro."""

    def __init__(self, pack: Pack, runtime: Callable[[], LlamaRuntime | None], *, threads: int | None = None) -> None:
        super().__init__(pack, threads=threads)
        self._runtime = runtime
        self.adapters: dict[str, int] = {}  # layer name -> the number the running server gave it
        self.marker = render3.MEDIA  # what the running server calls the place of a picture

    def _start(self) -> _Running:
        from ..engine import _SPAWNER, _die_with_parent, _free_port

        state = self.pack.check()
        if not state.ok:
            raise HostError(state.reason)
        manifest = self.pack.manifest() or {}
        runtime = self._runtime()
        if runtime is None:
            raise HostError(NO_RUNTIME)
        port, key = _free_port(), secrets.token_urlsafe(24)
        args = server_args(runtime, self.pack, manifest, port=port, threads=self.threads)
        env = {**runtime.env, "LLAMA_API_KEY": key}
        log_file = self._open_log()

        def spawn() -> subprocess.Popen[bytes]:
            return subprocess.Popen(
                args, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, env=env,
                preexec_fn=_die_with_parent, close_fds=True,
            )

        began = time.monotonic()
        try:
            process = _SPAWNER.run(spawn)
        except OSError as exc:
            raise HostError(f"Kiwi Mind could not start: {exc}") from exc
        finally:
            log_file.close()
        try:
            self._wait_ready(process, port, key)
            self.adapters = self._numbered(manifest, port, key)
            self.marker = self._marker(port, key)
        except HostError:
            self._kill(process)
            raise
        elapsed = int((time.monotonic() - began) * 1000)
        log.info("Kiwi Mind started on llama.cpp (pid %d, port %d, %d threads%s) in %d ms",
                 process.pid, port, self.threads, ", GPU" if runtime.offload else "", elapsed)
        return _Running(process, port, key, process.pid, time.time(), elapsed)

    def _wait_ready(self, process: subprocess.Popen[bytes], port: int, key: str) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while process.poll() is None and time.monotonic() < deadline:
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", headers=_auth(key), timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise HostError(f"Kiwi Mind did not start. Its log is {self.log_path}.")

    def _numbered(self, manifest: dict[str, Any], port: int, key: str) -> dict[str, int]:
        try:
            loaded = requests.get(f"http://127.0.0.1:{port}/lora-adapters", headers=_auth(key), timeout=10).json()
            by_path = {str(item["path"]): int(item["id"]) for item in loaded}
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise HostError("This llama.cpp build cannot switch Kiwi Mind's layers.") from exc
        numbers = {name: by_path.get(str(self.pack.path(file))) for name, file in layers(manifest)}
        missing = [name for name, number in numbers.items() if number is None]
        if missing:
            raise HostError(f"llama.cpp did not load Kiwi Mind's {missing[0]} layer. Its log is {self.log_path}.")
        return {name: int(number) for name, number in numbers.items() if number is not None}

    @staticmethod
    def _marker(port: int, key: str) -> str:
        """Newer llama.cpp servers choose the picture marker themselves at every start and name it in /props."""
        try:
            props = requests.get(f"http://127.0.0.1:{port}/props", headers=_auth(key), timeout=10).json()
        except (requests.RequestException, ValueError):
            return render3.MEDIA
        marker = props.get("media_marker") if isinstance(props, dict) else None
        return marker if isinstance(marker, str) and marker else render3.MEDIA

    @staticmethod
    def _kill(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(STOP_WAIT_S)


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


class LlamaClient(MindClient):
    """The requests of a 3.x pack: count tokens, ask for a route, read ahead, write a reply."""

    host: LlamaHost

    def __init__(self, host: LlamaHost) -> None:
        super().__init__(host)
        self._slots: OrderedDict[str, int] = OrderedDict()  # conversation -> server slot, least recently used first
        self._slot_lock = threading.Lock()

    def _refusal(self, response: requests.Response) -> MindRequestError:
        try:
            error = response.json().get("error")
        except ValueError:
            error = None
        return _refused(error, response.status_code, response.text[:300])

    def _ready(self) -> None:
        try:
            self.host.ensure()  # the layers get their numbers when the server starts
        except HostError as err:
            raise HostGone(str(err)) from err

    def _slot(self, name: str) -> int:
        if name == UTIL_SLOT:
            return CHAT_SLOTS
        with self._slot_lock:
            if name in self._slots:
                self._slots.move_to_end(name)
                return self._slots[name]
            if len(self._slots) < CHAT_SLOTS:
                slot = next(i for i in range(CHAT_SLOTS) if i not in self._slots.values())
            else:
                _least_recent, slot = self._slots.popitem(last=False)
            self._slots[name] = slot
            return slot

    def _on(self, layer: str) -> list[dict[str, Any]]:
        """The layer switch of a request: ``layer`` on, the others off."""
        numbers = self.host.adapters
        chosen = layer if layer in numbers else "chat"
        return [{"id": number, "scale": 1.0 if name == chosen else 0.0} for name, number in numbers.items()]

    def _base(self, prompt: str, slot: str, images: Sequence[str]) -> dict[str, Any]:
        marker = getattr(self.host, "marker", render3.MEDIA)
        text: Any = {"prompt_string": prompt.replace(render3.MEDIA, marker), "multimodal_data": list(images)} if images else prompt
        return {"prompt": text, "id_slot": self._slot(slot), "cache_prompt": True}

    # --- endpoints ---------------------------------------------------------------

    def tokenize(self, text: str) -> int:
        """Tokens of a prompt; a picture counts as the tokens the vision layers turn it into."""
        count = len(self._json("/tokenize", {"content": text, "add_special": False}).get("tokens") or [])
        vision = (self.host.pack.manifest() or {}).get("vision")
        each = int(vision.get("tokens") or 0) if isinstance(vision, dict) else 0
        return count + text.count(render3.MEDIA) * each

    def route(self, prompt: str, slot: str, *, skill: str | None = None, weigh: bool = True,
              images: Sequence[str] = ()) -> dict[str, Any]:
        """The route the model writes for the reply to come. ``prompt`` ends in ``<think>``. With ``skill`` the
        skill is given and only the rest is asked; without ``weigh`` the request is not weighed again."""
        self._ready()
        answer = self._json("/completion", {
            **self._base(prompt, slot, images), "n_predict": ROUTE_TOKENS, "temperature": 0.0,
            "grammar": render3.route_grammar(skill, weigh=weigh), "n_probs": ROUTE_PROBS if weigh else 0,
            "stop": list(render3.STOP), "lora": self._on(ROUTE),
        })
        text = str(answer.get("content") or "")
        for mark in render3.STOP:
            text = text.split(mark)[0]
        found = render3.parse_route(text)
        if found is None:
            raise MindRequestError(f"no route in {text!r}")
        timings = answer.get("timings") or {}
        return {
            "skill": found[0], "heavy": found[1], "act": found[2],
            "heavy_share": share(answer.get("completion_probabilities"), "heavy", "light") if weigh else None,
            "ms": float(timings.get("prompt_ms") or 0) + float(timings.get("predicted_ms") or 0),
        }

    def warm(self, prompt: str, slot: str, skill: str = "none", images: Sequence[str] = ()) -> dict[str, Any]:
        """Read a prompt that ends in ``<think>`` into the slot's cache."""
        self._ready()
        return self._json("/completion", {
            **self._base(prompt, slot, images), "n_predict": 1, "temperature": 0.0,
            "grammar": render3.route_grammar(), "lora": self._on(ROUTE),
        })

    def generate(self, body: dict[str, Any], abort: threading.Event | None = None) -> Iterator[dict[str, Any]]:
        """The events of one reply: {"text": ...} pieces, then one {"done": true, ...}. ``body`` is the
        conductor's: its prompt ends in ``</think>`` and ``skill`` names the layer that writes."""
        self._ready()
        request = {
            **self._base(str(body["prompt"]), str(body.get("slot") or "default"), body.get("images") or ()),
            "stream": True, "n_predict": int(body.get("max_tokens") or 256),
            "temperature": float(body.get("temperature") or 0.0), "stop": list(body.get("stop") or render3.STOP),
            "lora": self._on(str(body.get("skill") or "chat")),
            **{name: body[name] for name in ("top_k", "top_p", "repeat_penalty", "grammar") if name in body},
        }
        with self.host.busy():
            response = self._call("/completion", request, stream=True, abort=abort)
            finished = getattr(response, "kiwibro_finished", None)
            try:
                for raw in response.iter_lines(decode_unicode=False):
                    if not raw.startswith((b"data:", b"error:")):
                        continue
                    try:
                        event = json.loads(raw.split(b":", 1)[1].strip().decode("utf-8"))
                    except ValueError:
                        continue
                    if raw.startswith(b"error:") or event.get("error"):
                        raise _refused(event.get("error") or event, 0)
                    if event.get("content"):
                        yield {"text": str(event["content"])}
                    if event.get("stop"):
                        yield _done(event)
                        return
            except requests.RequestException as exc:
                if abort is not None and abort.is_set():
                    return
                raise HostGone("The connection to Kiwi Mind was interrupted.") from exc
            finally:
                if finished is not None:
                    finished.set()
                response.close()


def _refused(error: Any, status: int, fallback: str = "") -> MindRequestError:
    kind = str(error.get("type") or "") if isinstance(error, dict) else ""
    message = str(error.get("message") or fallback) if isinstance(error, dict) else str(error or fallback)
    return MindRequestError(message, status, context="context" in kind or "context size" in message)


def _done(event: dict[str, Any]) -> dict[str, Any]:
    """llama-server's last event in the shape the conductor reads."""
    timings = event.get("timings") or {}
    cached, read = int(timings.get("cache_n") or 0), int(timings.get("prompt_n") or 0)
    return {
        "done": True, "finish": "length" if event.get("stop_type") == "limit" else "stop",
        "usage": {"prompt_tokens": cached + read, "cached_tokens": cached,
                  "completion_tokens": int(timings.get("predicted_n") or 0)},
        "timings": {"upper_ms": float(timings.get("prompt_ms") or 0),
                    "decode_ms": float(timings.get("predicted_ms") or 0),
                    "predicted_per_second": float(timings.get("predicted_per_second") or 0)},
    }


def share(written: Any, word: str, other: str) -> float | None:
    """How much of the choice between two words went to ``word``, from the candidates llama-server lists for
    the place where one of them was written (None when it lists none)."""
    for entry in written if isinstance(written, list) else []:
        if str(entry.get("token") or entry.get("content") or "").strip() not in (word, other):
            continue
        found = {word: 0.0, other: 0.0}
        for item in entry.get("top_logprobs") or entry.get("top_probs") or entry.get("probs") or []:
            name = str(item.get("token") or item.get("tok_str") or "").strip()
            if name in found:
                chance = item.get("prob")
                if chance is None and item.get("logprob") is not None:
                    chance = math.exp(float(item["logprob"]))
                found[name] += float(chance or 0.0)
        total = found[word] + found[other]
        return found[word] / total if total > 0 else None
    return None
