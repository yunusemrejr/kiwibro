"""Kiwi Mind packs served through the model's native llama.cpp chat template."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..config import MEMORY_KINDS
from ..engine import SOURCE_MIND, DirectEngine, LocalModel
from ..llm import ChatResult, Completion, LLMClient, LLMError, TurnAborted
from .conductor import Events, has_images
from .pack import TRUNK3, Pack, PackError

NATIVE_FORMAT = "native-gguf"
# Keep the public Kiwi Mind identity distinct from the internal engine alias.
NATIVE_MODEL = "kiwimind-native"
DEFAULT_CTX = 8192
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


@dataclass(frozen=True)
class NativeModel:
    model: LocalModel
    reasoning_effort: bool


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
        ),
        thinking and capabilities.get("reasoning_effort") is True,
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
        extra: dict[str, Any] = {"reasoning_effort": "none"}
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
