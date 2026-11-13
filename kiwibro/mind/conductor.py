"""One generation call for Kiwi Mind: which skill, who answers, and how the reply comes back.

The agent hands over its messages and tools. This module renders them the way the skills were trained
(:mod:`kiwibro.mind.render`), asks the control head which skill fits the newest user message, streams the reply
without letting a ``<tool_call>`` block reach the chat as answer text, and returns the same ``ChatResult`` the
llama.cpp path returns. A hard question may be handed to the helper model for its words; tool decisions stay here.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from ..llm import ChatResult, Completion, ContextLimitError, LLMError, TurnAborted
from . import render
from .client import HostGone, MindClient, MindRequestError

log = logging.getLogger(__name__)

HARD = "hard"
UTIL_SLOT = "util"
SWITCH_MARGIN = 0.15  # how much better a new skill must score before a running conversation changes skill
HANDOFF_PEEK = 8  # generated pieces to look at before a hard question goes to the helper
MAX_TEMPERATURE = 0.7
SAMPLING = {"top_k": 20, "top_p": 0.8, "repeat_penalty": 1.05}
CALL_OPEN = "<tool_call"
CALL_TAG = "<tool_call>"

Events = Callable[[str, dict[str, Any]], None]
HelperStream = Callable[..., ChatResult]


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class Holdback:
    """Text that may be shown: everything before the first character that could open a tool call."""

    def __init__(self) -> None:
        self.text = ""
        self.sent = 0
        self.blocked = False

    def feed(self, piece: str) -> str:
        self.text += piece
        if self.blocked:
            return ""
        at = self.text.find(CALL_OPEN)
        if at >= 0:
            self.blocked = True
            out, self.sent = self.text[self.sent:at], len(self.text)
            return out
        keep = next((k for k in range(min(len(CALL_TAG) - 1, len(self.text)), 0, -1)
                     if self.text.endswith(CALL_TAG[:k])), 0)
        end = len(self.text) - keep
        out, self.sent = self.text[self.sent:end], end
        return out

    def finish(self) -> str:
        if self.blocked:
            return ""
        out, self.sent = self.text[self.sent:], len(self.text)
        return out


def opens_call(text: str) -> bool:
    """Whether the reply so far is, or is about to become, a tool call."""
    start = text.lstrip()
    return CALL_OPEN in text or (bool(start) and CALL_TAG.startswith(start))


def manual_text(tools: list[dict[str, Any]] | None) -> str:
    """What the prompt spends on instructions and the tool manual before the conversation starts."""
    return render.render([], tools, add_generation_prompt=False)


def has_images(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(m.get("content"), list)
        and any(isinstance(part, dict) and part.get("type") in ("image_url", "image") for part in m["content"])
        for m in messages
    )


@dataclass
class _Conversation:
    skill: str = ""
    hard: bool = False
    users: int = -1  # user messages seen when the skill was last chosen


@dataclass
class Settings:
    """What the conductor needs to know from its owner."""

    ctx: Callable[[], int]
    helper: Callable[[], dict[str, Any] | None]
    helper_stream: HelperStream


class Conductor:
    def __init__(self, client: MindClient, settings: Settings) -> None:
        self.client = client
        self.cfg = settings
        self._conversations: dict[str, _Conversation] = {}
        self._calls = 0
        self._lock = threading.Lock()

    # --- the agent's generation ---------------------------------------------------

    def stream(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, temperature: float,
        max_tokens: int, on_delta: Callable[[str], None] | None, abort: threading.Event | None,
        conversation: str | None, events: Events | None,
    ) -> ChatResult:
        emit: Events = events or (lambda _name, _data: None)
        helper = self.cfg.helper()
        if helper is not None and has_images(messages) and helper.get("vision"):
            return self._hand_off(helper, messages, temperature, max_tokens, on_delta, abort, emit,
                                  skill="", hard=False, fallback=None)
        slot = conversation or "default"
        prompt = render.render(messages, tools)
        cap = self._room(prompt, max_tokens)
        try:
            skill, hard = self._choose(slot, messages, prompt)
            emit("brain", {"brain": "kiwimind", "skill": skill, "hard": hard})
            return self._generate(
                prompt, slot, skill, hard, temperature, cap, on_delta, abort, emit,
                helper=helper if hard else None, messages=messages,
            )
        except MindRequestError as exc:
            raise _mapped(exc) from exc
        except HostGone as exc:
            raise LLMError("connection", str(exc)) from exc

    def _room(self, prompt: str, max_tokens: int) -> int:
        try:
            used = self.client.tokenize(prompt)
        except MindRequestError as exc:
            raise _mapped(exc) from exc
        except HostGone as exc:
            raise LLMError("connection", str(exc)) from exc
        room = self.cfg.ctx() - used - 1
        if room <= 0:
            raise ContextLimitError()
        return max(1, min(int(max_tokens), room))

    def _choose(self, slot: str, messages: list[dict[str, Any]], prompt: str) -> tuple[str, bool]:
        """The skill for this generation. The head is asked once per user message; the tool steps that follow
        keep the answer. A running conversation changes skill only for a clear gain."""
        users = sum(1 for m in messages if m.get("role") == "user")
        with self._lock:
            state = self._conversations.setdefault(slot, _Conversation())
            if state.skill and state.users == users:
                return state.skill, state.hard
            current = state.skill
        try:
            control = self.client.control(prompt, slot)
        except MindRequestError as exc:
            if exc.context:
                raise
            log.warning("Kiwi Mind has no control head answer (%s); using the chat skill", exc.message)
            control = {"names": ["chat"], "logits": [0.0]}
        names, logits = list(control.get("names") or []), list(control.get("logits") or [])
        scored = [(name, float(logits[i])) for i, name in enumerate(names) if name != HARD and i < len(logits)]
        hard = HARD in names and names.index(HARD) < len(logits) and float(logits[names.index(HARD)]) > 0
        if not scored:
            chosen = current or "chat"
        else:
            probs = dict(zip((name for name, _ in scored), softmax([score for _, score in scored])))
            top = max(probs, key=lambda name: probs[name])
            chosen = current if current in probs and probs[top] - probs[current] < SWITCH_MARGIN else top
        with self._lock:
            state.skill, state.hard, state.users = chosen, hard, users
        return chosen, hard

    def _body(self, prompt: str, slot: str, skill: str, temperature: float, max_tokens: int) -> dict[str, Any]:
        return {
            "prompt": prompt, "slot": slot, "skill": skill, "max_tokens": max_tokens,
            "temperature": 0.0 if temperature <= 0 else min(float(temperature), MAX_TEMPERATURE),
            "stop": list(render.STOP), **SAMPLING,
        }

    def _events(self, events: Iterator[dict[str, Any]], abort: threading.Event | None) -> Iterator[dict[str, Any]]:
        try:
            for event in events:
                if abort is not None and abort.is_set():
                    raise TurnAborted()
                yield event
            if abort is not None and abort.is_set():
                raise TurnAborted()
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()

    def _generate(
        self, prompt: str, slot: str, skill: str, hard: bool, temperature: float, cap: int,
        on_delta: Callable[[str], None] | None, abort: threading.Event | None, emit: Events, *,
        helper: dict[str, Any] | None, messages: list[dict[str, Any]],
    ) -> ChatResult:
        hold = Holdback()
        pending = ""  # shown only once we know the helper is not taking over
        peeking = helper is not None
        seen = 0
        done: dict[str, Any] | None = None
        stream = self._events(self.client.generate(self._body(prompt, slot, skill, temperature, cap), abort), abort)
        try:
            for event in stream:
                if event.get("done"):
                    done = event
                    break
                piece = str(event.get("text") or "")
                seen += 1
                shown = hold.feed(piece)
                if peeking:
                    pending += shown
                    if seen < HANDOFF_PEEK:
                        continue
                    peeking = False
                    if not opens_call(hold.text):
                        stream.close()
                        return self._hand_off(helper or {}, messages, temperature, cap, on_delta, abort, emit,
                                              skill=skill, hard=True,
                                              fallback=lambda: self._generate(
                                                  prompt, slot, skill, hard, temperature, cap, on_delta, abort,
                                                  emit, helper=None, messages=messages),
                                              )
                    shown, pending = pending, ""
                if shown and on_delta:
                    on_delta(shown)
        except MindRequestError as exc:
            raise _mapped(exc) from exc
        except HostGone as exc:
            raise LLMError("connection", str(exc)) from exc
        finally:
            stream.close()
        if done is None:
            raise LLMError("stream", "The connection to Kiwi Mind was interrupted.")
        tail = pending + hold.finish() if peeking else hold.finish()
        if tail and on_delta:
            on_delta(tail)
        return self._result(hold.text, done)

    def _result(self, text: str, done: dict[str, Any]) -> ChatResult:
        words, calls = render.parse_reply(text)
        with self._lock:
            tool_calls = []
            for call in calls:
                self._calls += 1
                tool_calls.append({
                    "id": f"call_{self._calls}", "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)},
                })
        usage = done.get("usage") or {}
        prompt_tokens, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        cached = int(usage.get("cached_tokens") or 0)
        timings = done.get("timings") or {}
        prompt_ms = sum(float(timings.get(k) or 0) for k in ("lower_ms", "upper_ms", "switch_ms"))
        finish = str(done.get("finish") or "stop")
        return ChatResult(
            content=words,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else ("length" if finish == "length" else "stop"),
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                   "total_tokens": prompt_tokens + completion},
            timings={
                "cache_n": float(cached), "prompt_n": float(max(0, prompt_tokens - cached)), "prompt_ms": prompt_ms,
                "predicted_n": float(completion), "predicted_ms": float(timings.get("decode_ms") or 0),
                "predicted_per_second": float(timings.get("predicted_per_second") or 0),
            },
        )

    # --- the helper takes over the words ---------------------------------------------

    def _hand_off(
        self, helper: dict[str, Any], messages: list[dict[str, Any]], temperature: float, max_tokens: int,
        on_delta: Callable[[str], None] | None, abort: threading.Event | None, emit: Events, *, skill: str,
        hard: bool, fallback: Callable[[], ChatResult] | None,
    ) -> ChatResult:
        from ..inference import NO_THINKING
        from ..toolcall import to_text_protocol

        name = str(helper.get("name") or helper.get("id"))
        emit("brain", {"brain": "helper", "name": name, "skill": skill, "hard": hard})
        spoke = False

        def delta(text: str) -> None:
            nonlocal spoke
            spoke = True
            if on_delta:
                on_delta(text)

        try:
            return self.cfg.helper_stream(
                model=str(helper["id"]), messages=to_text_protocol(messages), tools=None, temperature=temperature,
                max_tokens=max_tokens, on_delta=delta, abort=abort, extra=NO_THINKING,
            )
        except LLMError as exc:
            if spoke or fallback is None:
                raise
            log.info("helper %s could not answer (%s); Kiwi Mind answers itself", name, exc.message)
            emit("status", {"phase": "thinking", "label": f"{name} could not answer. Kiwi Mind is answering."})
            emit("brain", {"brain": "kiwimind", "skill": skill, "hard": hard})
            return fallback()

    # --- utility calls -------------------------------------------------------------

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float,
                 abort: threading.Event | None = None) -> Completion:
        prompt = render.render_plain(messages)
        cap = self._room(prompt, max_tokens)
        text, done = "", None
        body = self._body(prompt, UTIL_SLOT, "none", temperature, cap)
        stream = self._events(self.client.generate(body, abort), abort)
        try:
            for event in stream:
                if event.get("done"):
                    done = event
                    break
                text += str(event.get("text") or "")
        except MindRequestError as exc:
            raise _mapped(exc) from exc
        except HostGone as exc:
            raise LLMError("connection", str(exc)) from exc
        finally:
            stream.close()
        if done is None:
            raise LLMError("stream", "The connection to Kiwi Mind was interrupted.")
        usage = done.get("usage") or {}
        prompt_tokens, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        return Completion(
            text=text.strip(),
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                   "total_tokens": prompt_tokens + completion},
            finish_reason="length" if done.get("finish") == "length" else "stop",
        )

    def warm(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, slot: str | None) -> None:
        """Read the conversation so far into the slot's cache (the first reply then starts at once)."""
        try:
            self.client.warm(render.render(messages, tools), slot or "default")
        except MindRequestError as exc:
            raise _mapped(exc) from exc
        except HostGone as exc:
            raise LLMError("connection", str(exc)) from exc

    def forget(self, slot: str) -> None:
        with self._lock:
            self._conversations.pop(slot, None)


def _mapped(exc: MindRequestError) -> LLMError:
    if exc.context:
        return ContextLimitError()
    return LLMError("http", exc.message or "Kiwi Mind could not handle that request.", exc.status or None)
