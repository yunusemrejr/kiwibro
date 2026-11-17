"""Kiwi Mind packs served through the model's native llama.cpp chat template."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

from .. import adapters, gguf
from ..config import MEMORY_KINDS
from ..engine import SOURCE_MIND, DirectEngine, LocalModel
from ..llm import ADAPTERS_OFF, ChatResult, Completion, LLMClient, LLMError, TurnAborted
from .conductor import Events, has_images
from .pack import TRUNK3, Pack, PackError

NATIVE_FORMAT = "native-gguf"
# Keep the public Kiwi Mind identity distinct from the internal engine alias.
NATIVE_MODEL = "kiwimind-native"
DEFAULT_CTX = 8192
_SHA256 = re.compile(r"[a-f0-9]{64}")
MEMORY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_operations",
        "schema": {
            "type": "object",
            "properties": {
                "ops": {
                    "type": "array", "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"enum": ["add", "update", "delete"]},
                            "kind": {"enum": list(MEMORY_KINDS)},
                            "content": {"type": "string"},
                            "id": {"type": "string"},
                        },
                        "required": ["op"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["ops"],
            "additionalProperties": False,
        },
    },
}


class NativeChatClient(LLMClient):
    """Use the same private local engine even if the app also has an attached server."""

    def __init__(self, engine: Callable[[], DirectEngine]) -> None:
        super().__init__(lambda: {})
        self._engine = engine
        self._used_engine: DirectEngine | None = None

    def direct_engine(self) -> DirectEngine:
        if self._used_engine is None:
            self._used_engine = self._engine()
        return self._used_engine

    def unload(self) -> None:
        if self._used_engine is not None:
            self._used_engine.unload(NATIVE_MODEL)

    def schema_tokens(self, tools: list[dict[str, Any]]) -> int | None:
        """Count tool-template overhead on the loaded server without starting it."""
        engine = self._used_engine
        server = engine.loaded(NATIVE_MODEL) if engine is not None else None
        if engine is None or server is None:
            return None
        encoded = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        plain = _template_tokens(server.url, engine.api_key, "[]")
        with_tools = _template_tokens(server.url, engine.api_key, encoded) if plain is not None else None
        if plain is None or with_tools is None or with_tools <= plain:
            return None
        return with_tools - plain


@lru_cache(maxsize=64)
def _template_tokens(url: str, key: str, encoded: str) -> int | None:
    body = {
        "messages": [{"role": "system", "content": "You are KiwiBro."},
                     {"role": "user", "content": "Continue."}],
        "tools": json.loads(encoded),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {key}"}
    try:
        with requests.post(f"{url}/apply-template", json=body, headers=headers, timeout=1) as response:
            response.raise_for_status()
            prompt = response.json()["prompt"]
        with requests.post(f"{url}/tokenize", json={"content": prompt, "add_special": False, "parse_special": True},
                           headers=headers, timeout=1) as response:
            response.raise_for_status()
            tokens = response.json()["tokens"]
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return None
    return len(tokens) if isinstance(tokens, list) else None


@dataclass(frozen=True)
class NativeModel:
    model: LocalModel
    reasoning_effort: bool
    expert_bank: dict[str, int] | None = None
    training_steps: int | None = None
    feed_forward: dict[str, int] | None = None


def _expert_bank(manifest: Mapping[str, Any]) -> dict[str, int] | None:
    """Expose only the bounded summary of the optional architecture report."""
    raw = manifest.get("expert_bank")
    keys = ("experts_before", "experts_after", "top_k_before", "top_k_after")
    if not isinstance(raw, dict) or any(type(raw.get(key)) is not int or raw[key] < 1 for key in keys):
        return None
    if raw["top_k_before"] > raw["experts_before"] or raw["top_k_after"] > raw["experts_after"]:
        return None
    summary = {key: raw[key] for key in keys}
    before, after = raw.get("width_before"), raw.get("width_after")
    if type(before) is int and type(after) is int and 1 <= after < before:
        summary.update(width_before=before, width_after=after)
    return summary


def _training_steps(manifest: Mapping[str, Any]) -> int | None:
    training = manifest.get("training")
    if not isinstance(training, dict):
        return None
    steps = training.get("steps")
    total = training.get("total_steps", steps)
    if type(steps) is int and type(total) is int and 1 <= steps <= total <= 2**53 - 1:
        return total
    return None


def _dense_feed_forward(pack: Pack, manifest: dict, trunk: str) -> dict[str, int] | None:
    raw = manifest.get("feed_forward")
    keys = ("layers", "width_before", "width_after")
    if not isinstance(raw, dict) or any(type(raw.get(key)) is not int or not 1 <= raw[key] <= 2**53 - 1 for key in keys):
        return None
    if raw["width_after"] >= raw["width_before"] or manifest.get("arch") != "gemma4":
        return None
    try:
        header = _checked_header(pack.path(trunk), manifest["files"][trunk]["sha256"])
    except (OSError, gguf.GGUFError):
        return None
    if (header.get("general.architecture") != "gemma4" or header.get("gemma4.expert_count", 0) != 0
            or header.get("gemma4.block_count") != raw["layers"]
            or header.get("gemma4.feed_forward_length") != raw["width_after"]):
        return None
    return {key: raw[key] for key in keys}


def registered_lora(
    path: Path, metadata: Mapping[str, Any], *, arch: str, base: str, tasks: Any, scale: Any,
) -> adapters.Adapter:
    """Build the task adapter from the registered metadata and GGUF header."""
    if (metadata.get("general.type") != "adapter" or metadata.get("adapter.type") != "lora"
            or not isinstance(arch, str) or not arch or metadata.get("general.architecture") != arch):
        raise PackError("The native LoRA must be a GGUF adapter for the model's architecture.")
    if (not isinstance(tasks, list) or not tasks
            or any(task not in (*adapters.TASKS, adapters.ALWAYS) for task in tasks)
            or len(set(tasks)) != len(tasks)):
        raise PackError("The native LoRA has invalid task labels.")
    if type(scale) not in (int, float) or not 0 <= scale <= 2:
        raise PackError("The native LoRA scale must be a number from zero to two.")
    return adapters.Adapter(str(path), str(metadata.get("general.name") or path.stem),
                            arch, base, tuple(tasks), float(scale))


@lru_cache(maxsize=8)
def _checked_header(path: Path, sha256: str) -> dict[str, Any]:
    # Called only after Pack.check. Key by the checked digest so a new artifact
    # at the same path gets a new header without rereading vocabulary each round.
    return gguf.read_metadata(path)


def _pack_adapters(pack: Pack, manifest: dict, trunk: str, vision: str | None) -> tuple[adapters.Adapter, ...]:
    if "lora" not in manifest:
        return ()
    entry = manifest["lora"]
    filename = entry.get("file") if isinstance(entry, dict) else None
    files = manifest["files"]
    if not isinstance(filename, str) or not filename.endswith(".gguf") or filename not in files:
        raise PackError("The native LoRA must name a checksummed GGUF file in this pack.")
    path = pack.path(filename)
    if path == pack.path(trunk) or (vision is not None and str(path) == vision):
        raise PackError("The native LoRA must be a separate GGUF file.")
    binding = entry.get("trunk_sha256")
    if (not isinstance(binding, str) or not _SHA256.fullmatch(binding)
            or binding != files[trunk]["sha256"]):
        raise PackError("The native LoRA must bind to this pack's exact trunk SHA-256.")
    checksum = files[filename]["sha256"]
    if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum) or checksum == binding:
        raise PackError("The native LoRA must have its own SHA-256 in this pack.")
    try:
        model_arch = _checked_header(pack.path(trunk), binding).get("general.architecture")
        metadata = _checked_header(path, checksum)
    except (OSError, gguf.GGUFError) as exc:
        raise PackError("The native LoRA or trunk GGUF header could not be read.") from exc
    if model_arch != manifest.get("arch"):
        raise PackError("The native LoRA manifest architecture disagrees with the trunk GGUF.")
    return (registered_lora(path, metadata, arch=model_arch, base=str(manifest.get("base") or ""),
                            tasks=entry.get("tasks"), scale=entry.get("scale")),)


def describe_pack(pack: Pack) -> NativeModel:
    """Build an engine model from an independently verified native pack."""
    manifest = pack.manifest()
    if not manifest or manifest.get("engine") != "llama" or manifest.get("format") != NATIVE_FORMAT:
        raise PackError("This is not a native Kiwi Mind GGUF pack.")
    state = pack.check()
    if not state.ok:
        raise PackError(state.reason)
    files = manifest["files"]
    trunk = manifest.get("model", TRUNK3)
    if not isinstance(trunk, str) or not trunk.endswith(".gguf") or trunk not in files:
        raise PackError("The native Kiwi Mind manifest is missing its GGUF model.")
    vision = manifest.get("vision")
    projector = None
    if vision is not None:
        filename = vision.get("file") if isinstance(vision, dict) else None
        if not isinstance(filename, str) or not filename.endswith(".gguf") or filename == trunk or filename not in files:
            raise PackError("The native Kiwi Mind manifest is missing its vision projector.")
        projector = str(pack.path(filename))
    ctx = manifest.get("ctx", DEFAULT_CTX)
    if type(ctx) is not int or ctx < 512:
        raise PackError("The native Kiwi Mind manifest has an invalid context limit.")
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, dict) or capabilities.get("tools") is not True:
        raise PackError("The native Kiwi Mind manifest must declare tool calling support.")
    thinking = capabilities.get("thinking") is True
    base = str(manifest.get("base") or "")
    return NativeModel(
        LocalModel(
            id=NATIVE_MODEL, name="Kiwi Mind", path=str(pack.path(trunk)), projector=projector,
            size_bytes=sum(int(info["bytes"]) for info in files.values()), arch=str(manifest.get("arch") or ""),
            publisher=base.partition("/")[0], repo=base, params=str(manifest.get("params") or ""),
            quantization=str(manifest.get("quant") or ""), context_length=ctx,
            tools=True, thinking=thinking, thinking_toggle=thinking, source=SOURCE_MIND,
            pack_adapters=_pack_adapters(pack, manifest, trunk, projector),
        ),
        thinking and capabilities.get("reasoning_effort") is True,
        _expert_bank(manifest),
        _training_steps(manifest),
        _dense_feed_forward(pack, manifest, trunk),
    )


class NativeConductor:
    """Use the shared Chat Completions transport for tools, images and utility work."""

    def __init__(self, client: LLMClient, candidate: NativeModel) -> None:
        self.client = client
        self.candidate = candidate

    def _check_media(self, messages: list[dict[str, Any]]) -> None:
        if has_images(messages) and not self.candidate.model.projector:
            raise LLMError("vision", "This Kiwi Mind pack does not include a vision projector.")

    def _options(self, extra: Mapping[str, Any] | None, *, disable_thinking: bool = False) -> dict[str, Any]:
        options = dict(extra or {})
        template = dict(options.get("chat_template_kwargs") or {})
        if (disable_thinking or not self.candidate.model.thinking or options.get("reasoning_effort") == "none"
                or template.get("enable_thinking") is False):
            options["reasoning_effort"] = "none"
            template["enable_thinking"] = False
            template.pop("reasoning_effort", None)
        elif effort := options.get("reasoning_effort"):
            template.setdefault("enable_thinking", True)
            if self.candidate.reasoning_effort:
                # The shared engine may reduce the top-level field to thinking on/off.
                # The trained Qwen template can still consume the requested effort level.
                template.setdefault("reasoning_effort", effort)
            else:
                options.pop("reasoning_effort")
        if template:
            options["chat_template_kwargs"] = template
        return options

    def stream(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
        temperature: float, max_tokens: int, on_delta: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None, abort: threading.Event | None = None,
        events: Events | None = None, open_call: bool = False, extra: Mapping[str, Any] | None = None,
        on_rewrite: Callable[[str, str], None] | None = None, skill: str | None = None,
    ) -> ChatResult:
        if abort is not None and abort.is_set():
            raise TurnAborted()
        self._check_media(messages)
        options = self._options(extra, disable_thinking=open_call or extra is None)
        if open_call and tools:
            options.update(tool_choice="required", parallel_tool_calls=False)
        if not tools:
            options.pop("tool_choice", None)
            options.pop("parallel_tool_calls", None)
        choice = options.get("tool_choice")
        name = (choice.get("function") or {}).get("name") if isinstance(choice, dict) else None
        require_call = choice == "required" or bool(name)
        if events is not None:
            events("brain", {"brain": "kiwimind", "skill": skill, "hard": False})
        result = self.client.stream_chat(
            model=self.candidate.model.id, messages=messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, on_delta=None if require_call else on_delta, on_reasoning=on_reasoning,
            abort=abort, extra=options, on_rewrite=None if require_call else on_rewrite,
        )
        if require_call and not result.tool_calls:
            raise LLMError("tools", "Kiwi Mind did not return the required tool call.")
        if name and any(call["function"]["name"] != name for call in result.tool_calls):
            raise LLMError("tools", f"Kiwi Mind returned a different tool call than {name}.")
        if choice == "none" and result.tool_calls:
            raise LLMError("tools", "Kiwi Mind returned a tool call for a text-only request.")
        if require_call and on_delta is not None and result.content:
            on_delta(result.content)
        return result

    def complete(
        self, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float,
        abort: threading.Event | None = None, response_format: dict[str, Any] | None = None,
    ) -> Completion:
        # Streaming preserves cancellation during prefill, including for schema-constrained utilities.
        extra: dict[str, Any] = {"reasoning_effort": "none", ADAPTERS_OFF: True}
        if response_format is not None:
            extra["response_format"] = response_format
        result = self.stream(messages=messages, tools=None, max_tokens=max_tokens, temperature=temperature,
                             abort=abort, extra=extra)
        return Completion(text=result.content, usage=result.usage, finish_reason=result.finish_reason)

    def warm(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *,
        extra: Mapping[str, Any] | None = None, abort: threading.Event | None = None,
    ) -> None:
        self._check_media(messages)
        self.client.prefill(model=self.candidate.model.id, messages=messages, tools=tools,
                            extra=self._options(extra, disable_thinking=extra is None), abort=abort)
