"""One generation call for a Kiwi Mind 3 pack.

The model writes its own route before every reply (:mod:`kiwibro.mind.render3`): which skill answers, whether
the request is heavy, and whether the reply says something or acts. This conductor asks for that route, writes
it into the prompt and lets the routed skill's layer write the reply. It remembers the route of every reply,
because a later prompt has to repeat it exactly for the server to keep what it has read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Callable

from ..llm import ChatResult, Completion, LLMError
from . import render3
from .client import HostGone, MindRequestError
from .conductor import CALL_TAG, Conductor, Events, Settings, _Conversation, _mapped, from_kiwibro, has_images

log = logging.getLogger(__name__)

SEE = "see"
ROUTES_KEPT = 4096


# JSON as the tool calls are written: one line, a space after each colon and comma.
_JSON_RULES = r'''
object ::= "{" (string ":" ws value ("," ws string ":" ws value)*)? "}"
array ::= "[" (value ("," ws value)*)? "]"
value ::= object | array | string | number | "true" | "false" | "null"
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws ::= " "?
'''


def call_grammar(tools: list[dict[str, Any]] | None) -> str:
    """What may follow the opening of a tool call, as a llama.cpp grammar: calls of the tools on offer, each
    with a JSON object of arguments. A model this small otherwise drops a brace now and then or names a tool
    that is not there. "" when no tool has a name."""
    names = [str((tool.get("function") or tool).get("name") or "") for tool in tools or []]
    offered = " | ".join(json.dumps(json.dumps(name)) for name in names if name)
    if not offered:
        return ""
    return (
        'root ::= call ("\\n<tool_call>\\n" call)*\n'
        'call ::= "{\\"name\\": " name ", \\"arguments\\": " object "}\\n</tool_call>"\n'
        f"name ::= {offered}" + _JSON_RULES
    )


def with_pictures(messages: list[dict[str, Any]], sees: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """The messages with only the pictures the model can be given, and those pictures as base64 in the order
    their markers stand in the prompt. A model without eyes gets the words alone."""
    out: list[dict[str, Any]] = []
    pictures: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                data = _base64(part) if sees and message.get("role") == "user" else ""
                if not data:
                    continue
                pictures.append(data)
            parts.append(part)
        out.append({**message, "content": parts})
    return out, pictures


def _base64(part: dict[str, Any]) -> str:
    source = part.get("image_url")
    url = source.get("url") if isinstance(source, dict) else source
    if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
        return url.split(";base64,", 1)[1]
    return ""


def _digest(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8")).digest()


class Conductor3(Conductor):
    stop = render3.STOP

    def __init__(self, client: Any, settings: Settings) -> None:
        super().__init__(client, settings)
        # The route each reply was written under, by the reply's text and by its place in the conversation.
        self._routes: OrderedDict[tuple[str, Any], str] = OrderedDict()

    # --- the agent's generation ---------------------------------------------------

    def stream(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, temperature: float,
        max_tokens: int, on_delta: Callable[[str], None] | None, abort: threading.Event | None,
        conversation: str | None, events: Events | None, open_call: bool = False,
    ) -> ChatResult:
        emit: Events = events or (lambda _name, _data: None)
        helper = self.cfg.helper()
        sees = self.cfg.sees()
        if helper is not None and not sees and has_images(messages) and helper.get("vision"):
            return self._hand_off(helper, messages, temperature, max_tokens, on_delta, abort, emit,
                                  skill="", hard=False, fallback=None)
        slot = conversation or "default"
        messages, pictures = with_pictures(messages, sees)
        seen = {"images": pictures} if pictures else {}
        prompt = render3.render(messages, tools, routes=self._known(slot, messages))
        try:
            cap = self._room(prompt, max_tokens)
            skill, heavy, act, hard = self._route(slot, messages, prompt, pictures, bool(tools), open_call)
            emit("brain", {"brain": "kiwimind", "skill": skill, "hard": hard})
            route = render3.route_text(skill, heavy, act)

            def write() -> ChatResult:
                # A reply that acts is opened as the tool call its route announced, and held to a real call.
                grammar = call_grammar(tools) if act else ""
                return self._generate(prompt + route + render3.THINK_END, slot, skill, hard, temperature, cap,
                                      on_delta, abort, emit, helper=None, messages=messages,
                                      opened=CALL_TAG + "\n" if act else "",
                                      extra={**seen, "grammar": grammar} if grammar else seen)

            # A heavy request that is answered in words goes to the helper; a step with a tool stays here.
            if hard and not act and helper is not None:
                result = self._hand_off(helper, messages, temperature, cap, on_delta, abort, emit,
                                        skill=skill, hard=True, fallback=write)
            else:
                result = write()
        except MindRequestError as exc:
            raise _mapped(exc) from exc
        except HostGone as exc:
            raise LLMError("connection", str(exc)) from exc
        self._remember(slot, messages, result, route)
        return result

    def _route(self, slot: str, messages: list[dict[str, Any]], prompt: str, pictures: list[str],
               tools: bool, open_call: bool) -> tuple[str, bool, bool, bool]:
        """(skill, the route says heavy, the reply acts, the turn is heavy). The first reply to a user message
        chooses the skill and weighs the request; the tool steps that follow keep both and only choose between
        saying and acting."""
        asks = [m for m in messages if m.get("role") == "user" and not from_kiwibro(m)]
        with self._lock:
            state = self._conversations.setdefault(slot, _Conversation())
            first = not (state.skill and state.users == len(asks))
            kept = "" if first else state.skill
            hard = state.hard
        if kept and (open_call or not tools):
            # The skill and weight are fixed. A forced call, or the absence of
            # tools, also fixes the last route word, so go straight to generation.
            return kept, False, tools and open_call, hard
        answer: dict[str, Any] = {}
        looking = bool(first and pictures and asks and render3.images(asks[-1]))
        if looking:
            kept = SEE  # a message that brings a picture is answered by the skill that looks, in words: it has no route to choose
        else:
            try:
                answer = self.client.route(prompt, slot, skill=kept or None, weigh=first, images=pictures)
            except MindRequestError as exc:
                if exc.context:
                    raise
                log.warning("Kiwi Mind wrote no route (%s); using the %s skill", exc.message, kept or "chat")
        skill = str(answer.get("skill") or kept or "chat")
        heavy = first and self._heavy(answer)
        if first:
            hard = heavy
        with self._lock:
            state.skill, state.hard, state.users, state.read = skill, hard, len(asks), True
        return skill, heavy, tools and not looking and (open_call or bool(answer.get("act"))), hard

    def _heavy(self, answer: dict[str, Any]) -> bool:
        """The pack names the share of "heavy" from which a request counts as heavy; without it the word the
        route layer wrote decides."""
        mark, share = self.cfg.heavy_at(), answer.get("heavy_share")
        if mark is not None and share is not None:
            return float(share) >= mark
        return bool(answer.get("heavy"))

    # --- the routes of earlier replies ----------------------------------------------

    def _known(self, slot: str, messages: list[dict[str, Any]]) -> list[str | None]:
        replies = [m for m in messages if m.get("role") == "assistant"]
        with self._lock:
            return [
                self._routes.get((slot, _digest(render3.assistant_text(reply)))) or self._routes.get((slot, place))
                for place, reply in enumerate(replies)
            ]

    def _remember(self, slot: str, messages: list[dict[str, Any]], result: ChatResult, route: str) -> None:
        said = render3.assistant_text({"content": result.content, "tool_calls": result.tool_calls})
        place = sum(1 for m in messages if m.get("role") == "assistant")
        with self._lock:
            for key in ((slot, _digest(said)), (slot, place)):
                self._routes[key] = route
                self._routes.move_to_end(key)
            while len(self._routes) > ROUTES_KEPT:
                self._routes.popitem(last=False)

    # --- utility calls -------------------------------------------------------------

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float,
                 abort: threading.Event | None = None) -> Completion:
        messages, pictures = with_pictures(messages, self.cfg.sees())
        if pictures:  # a question about a picture: the see skill answers it, as it learned to
            prompt = render3.render(messages, None) + render3.route_text(SEE) + render3.THINK_END
            return self._complete(prompt, SEE, max_tokens, temperature, abort, {"images": pictures})
        return self._complete(render3.render_plain(messages), "chat", max_tokens, temperature, abort)

    def warm(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, slot: str | None) -> None:
        slot = slot or "default"
        messages, pictures = with_pictures(messages, self.cfg.sees())
        try:
            self.client.warm(render3.render(messages, tools, routes=self._known(slot, messages)), slot,
                             images=pictures)
        except MindRequestError as exc:
            raise _mapped(exc) from exc
        except HostGone as exc:
            raise LLMError("connection", str(exc)) from exc
