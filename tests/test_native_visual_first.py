"""A bounded native decision can use parsed visual fields or defer to the normal agent."""

from kiwibro import pagecheck
from kiwibro.llm import ChatResult
from kiwibro.mind import KIWIMIND

from test_backend import chat_events, env, tool_call_result  # noqa: F401
from test_pagecheck import Fake


BRIEF = (
    "Save aperture.html, a single self-contained HTML access panel. Show a filter bay under a service lid. "
    "Use CSS perspective. The service lid opens toward the rear on its top-edge hinge while the base stays fixed. "
    "Give the scene, base and lid the IDs housing, frame and cover. Start closed. "
    "Use a single labeled native button with ID hatch to open and close the lid, keep keyboard focus, "
    "and set aria-expanded. Keep the scene within a 360 px phone viewport. Check the saved page using look_at_page."
)
INFO = {"id": KIWIMIND, "engine": "native-gguf", "arch": "gemma4", "tools": True, "context": 8192}


def _component_choice(action):
    result = tool_call_result("component", {"action": action, "name": "web/hinged-panel"})
    result.finish_reason = "tool_calls"
    return result


def test_accepted_visual_decision_uses_the_bound_fields(env, monkeypatch):
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    fake = Fake()

    def render(page, png, width, height, budget, *, expected_ids=()):
        result = fake(page, png, width, height, budget)
        result.report["ids"] = [f'id="{identity}"' in page.read_text() for identity in expected_ids]
        return result

    monkeypatch.setattr(pagecheck, "chrome_render", render)
    env.fake.stream_script = [_component_choice("use")]

    events, _raw = chat_events(env, BRIEF)

    first = env.fake.stream_calls[0]
    assert first["extra"]["tool_choice"] == "required"
    assert first["extra"]["parallel_tool_calls"] is False
    assert [item["function"]["name"] for item in first["tools"]] == ["component"]
    page = env.workspace / "aperture.html"
    assert page.is_file(), [(name, item) for name, item in events if name in {"done", "error", "tool_result"}]
    output = page.read_text()
    assert all(literal in output for literal in ('id="housing"', 'id="frame"', 'id="cover"', 'id="hatch"',
                                                 'data-edge="top"', 'data-direction="away"'))
    assert any(name == "tool_result" and item["name"] == "look_at_page" and item["ok"] for name, item in events)
    assert len(env.fake.stream_calls) == 1


def test_deferred_visual_decision_retains_general_tools(env, monkeypatch):
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    env.fake.stream_script = [
        _component_choice("show"),
        ChatResult(content="I need a custom design for this mechanism."),
    ]

    chat_events(env, BRIEF)

    assert "write_file" in [item["function"]["name"] for item in env.fake.stream_calls[1]["tools"]]
    assert not (env.workspace / "aperture.html").exists()
