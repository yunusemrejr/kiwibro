"""A tool call Kiwi Mind is still writing: the status line says what and how far, and a call that loops is cut."""

import json

import pytest

from kiwibro.inference import DegenerateOutput
from kiwibro.mind.conductor import CallWatch


def _watch():
    seen = []
    watch = CallWatch(lambda name, data: seen.append((name, data)))
    watch.EVERY = 0  # every piece is looked at
    return watch, seen


def _call(content: str, name: str = "write_file", path: str = "kiwitree.html") -> str:
    """The text of a call cut off while its content is still being written (no closing quote yet)."""
    whole = json.dumps({"name": name, "arguments": {"path": path, "content": content}})
    return "<tool_call>\n" + whole[: whole.rindex('"}}')]


def test_status_names_the_file_and_counts_lines():
    watch, seen = _watch()
    watch.see("<tool_call>\n{\"name\": \"write_file\"")
    page = "<html>\n<body>\n<canvas id=\"scene\"></canvas>\n<script>\nconst c = 1;\nconst d = 2;\n"
    watch.see(_call(page, path="site/kiwitree.html"))
    labels = [data["label"] for name, data in seen if name == "status"]
    assert labels[0] == "Writing a file"
    assert labels[-1] == "Writing kiwitree.html · 7 lines"
    assert seen[-1][1]["phase"] == "writing"


def test_same_label_is_not_sent_twice():
    watch, seen = _watch()
    call = _call("a\nb\n")
    watch.see(call)
    watch.see(call)
    assert len(seen) == 1


def test_edit_and_other_tools():
    watch, seen = _watch()
    watch.see('<tool_call>\n{"name": "replace_in_file", "arguments": {"path": "a/b.css", "old": "x"')
    watch.see('<tool_call>\n{"name": "web_search", "arguments": {"query": "kiwi"')
    labels = [data["label"] for _name, data in seen]
    assert labels[0] == "Editing b.css"
    assert labels[1] and not labels[1].startswith("Editing")


def test_a_call_that_differs_only_in_numbers_is_cut():
    """What Kiwi Mind wrote on 16 Nov 2026 for 4,000 tokens before the user stopped it."""
    watch, _seen = _watch()
    lines = ["<html><body><canvas id=\"scene\"></canvas><script>", "function drawKiwi(x, y) {"]
    for step in range(60):
        lines += ['  ctx.fillStyle = "green";', f"  ctx.fillRect(x + {10 + 10 * (step % 3)}, y + {20 * step}, 60, 10);"]
    with pytest.raises(DegenerateOutput):
        watch.see(_call("\n".join(lines) + "\n"))


def test_real_code_with_repeated_shapes_is_not_cut():
    watch, _seen = _watch()
    page = ["<html><head><style>body { margin: 0; background: #2a1b4d; }</style></head><body>",
            '<canvas id="scene" width="800" height="600"></canvas>', "<script>",
            "const ctx = document.getElementById('scene').getContext('2d');",
            "const points = ["]
    page += [f"  [{i * 13 % 97}, {i * 7 % 53}]," for i in range(80)]  # a data table: numbers only
    page += ["];", "function tree(x, y) {"]
    page += [f"  ctx.lineTo(x + {i * 3}, y - {i * i % 41});" for i in range(20)]  # a polyline
    page += ["  ctx.stroke();", "}", "let t = 0;", "function frame() {", "  t += 1;",
             "  ctx.clearRect(0, 0, 800, 600);", "  tree(600, 500);", "  requestAnimationFrame(frame);", "}",
             "frame();", "</script></body></html>"]
    watch.see(_call("\n".join(page) + "\n"))  # does not raise


def test_escaped_text_is_read_as_lines():
    watch, seen = _watch()
    watch.see(_call('line "one"\nline two\nthree\nfour\nfive\n'))
    assert seen[-1][1]["label"] == "Writing kiwitree.html · 6 lines"


def test_a_patch_whose_old_text_loops_is_cut():
    """Rounds 4 and 5 of the same chat: an apply_patch call whose "old" text ran on to the token cap, twice."""
    watch, seen = _watch()
    old = "".join(f'ctx.fillStyle = "green";\n    ctx.fillRect(x + {10 + 10 * (i % 3)}, y + {220 + 20 * i}, 40, 10);\n    '
                  for i in range(40))
    whole = json.dumps({"name": "apply_patch", "arguments": {"path": "kiwitree.html", "edits": [{"old": old}]}})
    with pytest.raises(DegenerateOutput):
        watch.see("<tool_call>\n" + whole[: whole.rindex('"}]}}')])
    assert seen == [] or seen[-1][1]["label"].startswith("Editing kiwitree.html")
