"""Regression tests for kiwibro.toolcall: lenient JSON, tool-name and argument repair,
rescue of tool calls written as text, the text tool protocol and the streaming filter."""

from __future__ import annotations

import json

import pytest

from kiwibro import toolcall as tc
from kiwibro import tools

SPECS = tools.TOOL_SPECS
NAMES = list(SPECS)


# --- lenient JSON --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("{'query': 'kiwi', 'max_results': '3',}", {"query": "kiwi", "max_results": "3"}),
        ("{query: weather in Ankara, max_results: 20}", {"query": "weather in Ankara", "max_results": 20}),
        ('{"content": "say "hi" now", "path": "a.txt"}', {"content": 'say "hi" now', "path": "a.txt"}),
        ('{"a": 1 "b": 2}', {"a": 1, "b": 2}),
        ('{"query": "x"', {"query": "x"}),
        ('```json\n{"query": "x"}\n```', {"query": "x"}),
        ("{'text': 'don't stop', 'n': True, 'm': None}", {"text": "don't stop", "n": True, "m": None}),
    ],
)
def test_repair_json_recovers_common_small_model_mistakes(text, expected):
    assert tc.repair_json(text) == expected


def test_repair_json_keeps_backslashes_in_regex_text():
    assert tc.repair_json('{"re": "\\d+"}') == {"re": "\\d+"}


def test_repair_json_returns_none_without_brackets():
    assert tc.repair_json("just some words") is None
    assert tc.repair_json("") is None


def test_repair_json_survives_deep_nesting():
    tc.repair_json("[" * 5000)  # must return instead of raising


# --- parse_arguments -----------------------------------------------------------------------


def test_parse_arguments_passes_dicts_through():
    args, notes = tc.parse_arguments({"query": "x"})
    assert args == {"query": "x"}
    assert notes == []


def test_parse_arguments_decodes_double_encoded_json():
    args, notes = tc.parse_arguments('"{\\"query\\": \\"double\\"}"')
    assert args == {"query": "double"}
    assert notes


def test_parse_arguments_wraps_plain_text_as_positional():
    args, _notes = tc.parse_arguments("weather in Ankara")
    assert args == {tc.POSITIONAL: ["weather in Ankara"]}


@pytest.mark.parametrize("raw", [None, ""])
def test_parse_arguments_empty_input_gives_no_arguments(raw):
    args, _notes = tc.parse_arguments(raw)
    assert args == {}


# --- tool names ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("websearch", "web_search"),
        ("functions.web_search", "web_search"),
        ("bash", "run_command"),
        ("cat", "read_file"),
        ("Web-Search", "web_search"),
        ("totally_unknown_xyz", None),
    ],
)
def test_resolve_name_maps_near_miss_names(raw, expected):
    assert tc.resolve_name(raw, NAMES) == expected


# --- argument normalisation ----------------------------------------------------------------


def test_normalize_args_maps_synonyms_and_coerces_types():
    args, _notes = tc.normalize_args(
        {"file": "README.md", "start_line": "1"}, SPECS["read_file"].parameters
    )
    assert args == {"path": "README.md", "start": 1}


def test_normalize_args_clamps_out_of_range_numbers_with_a_note():
    args, notes = tc.normalize_args({"query": "x", "max_results": 20}, SPECS["web_search"].parameters)
    assert args["max_results"] == 8
    assert any("max_results" in note and "8" in note for note in notes)


def test_normalize_args_drops_unknown_arguments_with_a_note():
    args, notes = tc.normalize_args({"query": "x", "bogus": 1}, SPECS["web_search"].parameters)
    assert args == {"query": "x"}
    assert any("bogus" in note for note in notes)


def test_normalize_args_assigns_positional_values_to_schema_order():
    args, _notes = tc.normalize_args({tc.POSITIONAL: ["cats", 3]}, SPECS["web_search"].parameters)
    assert args == {"query": "cats", "max_results": 3}


def test_normalize_args_unwraps_arguments_envelope():
    args, _notes = tc.normalize_args({"arguments": {"query": "x"}}, SPECS["web_search"].parameters)
    assert args == {"query": "x"}


@pytest.mark.parametrize("word, expected", [("yes", True), ("no", False)])
def test_normalize_args_reads_yes_no_for_boolean_properties(word, expected):
    args, _notes = tc.normalize_args(
        {"pattern": "x", "case_sensitive": word}, SPECS["grep_files"].parameters
    )
    assert args["case_sensitive"] is expected


# --- prepare_call ----------------------------------------------------------------------------


def test_prepare_call_repairs_name_and_arguments_together():
    call = {"id": "c1", "function": {"name": "websearch", "arguments": "{'q': 'kiwibro'}"}}
    prepared = tc.prepare_call(call, SPECS)
    assert prepared.name == "web_search"
    assert prepared.args == {"query": "kiwibro"}
    assert prepared.error is None
    assert prepared.notes


def test_prepare_call_unknown_tool_returns_error_listing_tools():
    call = {"id": "c2", "function": {"name": "frobnicate", "arguments": "{}"}}
    prepared = tc.prepare_call(call, SPECS)
    assert prepared.error is not None
    assert "Unknown tool" in prepared.error
    assert "web_search" in prepared.error


# --- tool calls written as text ------------------------------------------------------------

_DEEPSEEK_BAR = "｜"  # fullwidth vertical bar used in DeepSeek markers
_DEEPSEEK_SEP = "▁"  # lower one eighth block used in DeepSeek markers

EXTRACT_CASES = [
    pytest.param(
        '<tool_call>\n{"name": "web_search", "arguments": {"query": "a"}}\n</tool_call>',
        "web_search",
        {"query": "a"},
        id="hermes",
    ),
    pytest.param(
        "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes/a.md\n</parameter>\n</function>\n</tool_call>",
        "read_file",
        {"path": "notes/a.md"},
        id="qwen3-coder",
    ),
    pytest.param(
        '<|python_tag|>{"name": "web_search", "parameters": {"query": "llama"}}<|eom_id|>',
        "web_search",
        {"query": "llama"},
        id="llama",
    ),
    pytest.param(
        '[TOOL_CALLS] [{"name": "fetch_page", "arguments": {"url": "https://x.org"}}]',
        "fetch_page",
        {"url": "https://x.org"},
        id="mistral-json",
    ),
    pytest.param(
        '[TOOL_CALLS]web_search[ARGS]{"query": "v11"}',
        "web_search",
        {"query": "v11"},
        id="mistral-args",
    ),
    pytest.param(
        '<|channel|>commentary to=functions.web_search <|constrain|>json<|message|>{"query":"h"}<|call|>',
        "web_search",
        {"query": "h"},
        id="harmony",
    ),
    pytest.param(
        f"<{_DEEPSEEK_BAR}tool{_DEEPSEEK_SEP}calls{_DEEPSEEK_SEP}begin{_DEEPSEEK_BAR}>"
        f"<{_DEEPSEEK_BAR}tool{_DEEPSEEK_SEP}call{_DEEPSEEK_SEP}begin{_DEEPSEEK_BAR}>function"
        f"<{_DEEPSEEK_BAR}tool{_DEEPSEEK_SEP}sep{_DEEPSEEK_BAR}>web_search\n```json\n"
        '{"query": "d"}\n```'
        f"<{_DEEPSEEK_BAR}tool{_DEEPSEEK_SEP}call{_DEEPSEEK_SEP}end{_DEEPSEEK_BAR}>"
        f"<{_DEEPSEEK_BAR}tool{_DEEPSEEK_SEP}calls{_DEEPSEEK_SEP}end{_DEEPSEEK_BAR}>",
        "web_search",
        {"query": "d"},
        id="deepseek",
    ),
    pytest.param(
        '```tool_code\nweb_search(query="g", max_results=2)\n```',
        "web_search",
        {"query": "g", "max_results": 2},
        id="gemma",
    ),
    pytest.param(
        'Thought: x\nAction: web_search\nAction Input: {"query": "r"}',
        "web_search",
        {"query": "r"},
        id="react",
    ),
    pytest.param(
        '{"name": "read_file", "arguments": {"path": "a"}}',
        "read_file",
        {"path": "a"},
        id="bare-json",
    ),
    pytest.param(
        '<invoke name="list_files"><parameter name="path">.</parameter></invoke>',
        "list_files",
        {"path": "."},
        id="invoke",
    ),
]


@pytest.mark.parametrize("content, name, expected_args", EXTRACT_CASES)
def test_extract_calls_recognises_each_text_format(content, name, expected_args):
    calls, _cleaned = tc.extract_calls(content, NAMES)
    assert len(calls) == 1
    call = calls[0]
    assert call["id"]
    assert call["type"] == "function"
    assert call["function"]["name"] == name
    assert json.loads(call["function"]["arguments"]) == expected_args


def test_extract_calls_keeps_prose_before_hermes_block():
    calls, cleaned = tc.extract_calls(
        'Let me check.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "a"}}\n</tool_call>', NAMES
    )
    assert len(calls) == 1
    assert cleaned == "Let me check."


def test_extract_calls_removes_harmony_markup_from_cleaned_content():
    _calls, cleaned = tc.extract_calls(
        '<|channel|>commentary to=functions.web_search <|constrain|>json<|message|>{"query":"h"}<|call|>', NAMES
    )
    assert "to=functions" not in cleaned


def test_extract_calls_pythonic_call_maps_positional_arguments_on_prepare():
    calls, _cleaned = tc.extract_calls('web_search("p", 3)', NAMES)
    assert calls[0]["function"]["name"] == "web_search"
    prepared = tc.prepare_call(calls[0], SPECS)
    assert prepared.args == {"query": "p", "max_results": 3}


@pytest.mark.parametrize(
    "content",
    [
        'For example, you could call {"name": "web_search", "arguments": {"query": "x"}} to search.',
        "```js\nconst user = {name: \"b\"};\n```",
        "",
    ],
    ids=["example-prose", "js-code-block", "empty"],
)
def test_extract_calls_ignores_examples_and_code(content):
    calls, cleaned = tc.extract_calls(content, NAMES)
    assert calls == []
    assert cleaned == content


def test_extract_calls_ignores_long_prose_mentioning_a_call_mid_sentence():
    prose = (
        "The history of the kiwi fruit is long and well documented across many sources. " * 8
        + 'In the middle of one sentence we could write web_search(query="x") as an example, and then '
        + "the answer continues with more explanation about the topic at hand. " * 8
    )
    assert len(prose) > 500
    calls, cleaned = tc.extract_calls(prose, NAMES)
    assert calls == []
    assert cleaned == prose


def test_extract_calls_ignores_long_prose_with_a_call_on_its_own_line():
    prose = (
        "The history of the kiwi fruit is long and well documented across many sources. " * 8
        + '\nweb_search(query="x")\n'
        + "The answer then continues with more explanation about the topic at hand. " * 8
    )
    calls, _cleaned = tc.extract_calls(prose, NAMES)
    assert calls == []


# --- StreamFilter ---------------------------------------------------------------------------


def test_stream_filter_holds_tool_markup_and_releases_prose():
    stream = tc.StreamFilter()
    chunks = ["Sure, let me look ", "that up. <tool", "_call>\n{...}", "\n</tool_call>"]
    visible = "".join(stream.feed(chunk) for chunk in chunks)
    assert visible == "Sure, let me look that up. "
    assert stream.held.startswith("<tool_call>")
    assert stream.flush() == ""


def test_stream_filter_flush_releases_text_that_only_looked_like_a_marker():
    stream = tc.StreamFilter()
    chunks = ["a < b and ", "c <", "= d"]
    visible = "".join(stream.feed(chunk) for chunk in chunks) + stream.flush()
    assert visible == "a < b and c <= d"
    assert stream.held == ""


# --- corrective errors and the text protocol -----------------------------------------------


def test_correction_shows_error_signature_and_example():
    spec = SPECS["web_search"]
    text = tc.correction(spec, "boom")
    assert "Error: boom" in text
    assert "web_search(query: string, max_results?: integer 1-8)" in text
    assert "Example arguments" in text


def test_protocol_block_is_deterministic_and_lists_each_tool():
    specs = [SPECS["web_search"], SPECS["read_file"]]
    first = tc.protocol_block(specs)
    assert first == tc.protocol_block(specs)
    assert "<tool_call>" in first
    assert "web_search(query: string, max_results?: integer 1-8)" in first
    assert "read_file(path: string" in first


def test_to_text_protocol_rewrites_tool_history_as_plain_text():
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "find kiwi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": "kiwi"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "kiwi results"},
        {"role": "user", "content": "thanks"},
    ]
    out = tc.to_text_protocol(messages)

    assert out[0] == {"role": "system", "content": "be brief"}
    assistant = out[2]
    assert assistant["role"] == "assistant"
    assert "<tool_call>" in assistant["content"]
    assert "web_search" in assistant["content"]
    assert "tool_calls" not in assistant

    result_turn = out[3]
    assert result_turn["role"] == "user"
    assert '<tool_result name="web_search">' in result_turn["content"]
    assert "kiwi results" in result_turn["content"]

    assert len(out) == 4  # the trailing user message is merged into the tool-result turn
    assert result_turn["content"].endswith("thanks")
    assert all(message["role"] != "tool" for message in out)


# --- empty replies ---------------------------------------------------------------------------


def test_is_empty_reply_ignores_think_blocks_and_whitespace():
    assert tc.is_empty_reply("<think>hmm</think>  ", None) is True


def test_is_empty_reply_false_for_visible_text():
    assert tc.is_empty_reply("Hi", None) is False


def test_is_empty_reply_false_when_there_is_a_tool_call():
    assert tc.is_empty_reply("", [{"id": "c1"}]) is False


# --- regression from live runs: small models invent sandbox roots -------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("/workspace", "."), ("/workspace/notes/a.md", "notes/a.md"), ("/mnt/data/report.csv", "report.csv")],
)
def test_phantom_sandbox_roots_become_workspace_relative(raw, expected):
    args, notes = tc.normalize_args({"path": raw}, tools.TOOL_SPECS["list_files"].parameters)
    assert args["path"] == expected
    assert any("relative to the workspace" in note for note in notes)


def test_real_paths_are_left_alone():
    for raw in ("notes/a.md", "/home/someone/Kiwibro/a.md", "/workspaces/a.md"):
        args, _notes = tc.normalize_args({"path": raw}, tools.TOOL_SPECS["read_file"].parameters)
        assert args["path"] == raw


def test_phantom_roots_in_path_lists():
    if "read_files" not in tools.TOOL_SPECS:
        pytest.skip("read_files not in this catalog")
    args, _notes = tc.normalize_args({"paths": ["/workspace/a.md", "b.md"]}, tools.TOOL_SPECS["read_files"].parameters)
    assert args["paths"] == ["a.md", "b.md"]
