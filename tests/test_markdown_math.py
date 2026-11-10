"""LaTeX maths in chat replies: the browser renderer makes MathML, and everything else renders as before."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "markdown_math.mjs"
MARKDOWN = ROOT / "static" / "js" / "markdown.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the browser renderer")


def render(texts: list[str], tmp_path: Path) -> list[str]:
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(texts), encoding="utf-8")
    done = subprocess.run(
        ["node", str(RUNNER), str(MARKDOWN), str(cases)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_inline_maths_becomes_mathml(tmp_path):
    [html] = render([r"\( x^2 + y_1 \)"], tmp_path)
    assert "<msup>" in html
    assert "<msub>" in html


def test_dollar_maths_becomes_a_fraction(tmp_path):
    [html] = render([r"$\frac{a}{b}$"], tmp_path)
    assert "<mfrac>" in html


def test_bare_variables_in_dollars_are_maths(tmp_path):
    [html] = render(["Let $x$ and $n$ be integers."], tmp_path)
    assert html.count("<math>") == 2


def test_display_sum_inside_a_paragraph(tmp_path):
    [html] = render([r"$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$"], tmp_path)
    assert 'display="block"' in html
    assert "<munderover>" in html


def test_display_block_spans_lines(tmp_path):
    [html] = render(["Before\n\\[\na + b\n\\]\nAfter"], tmp_path)
    assert html.startswith('<p>Before</p><div class="math-block"><math display="block">')
    assert html.endswith("</math></div><p>After</p>")


def test_display_block_of_dollars_spans_lines(tmp_path):
    [html] = render(["$$\nx^2\n$$"], tmp_path)
    assert html.startswith('<div class="math-block"><math display="block">')
    assert html.endswith("</math></div>")


def test_streaming_display_block_is_text_until_closed(tmp_path):
    [html] = render(["$$\nx^2"], tmp_path)
    assert html == "<p>$$<br>x^2</p>"


def test_display_block_closed_by_a_blank_line_is_text(tmp_path):
    [html] = render(["$$\nx^2\n\nnext"], tmp_path)
    assert html == "<p>$$<br>x^2</p><p>next</p>"


def test_cases_environment_gives_a_table(tmp_path):
    [html] = render([r"$$\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}$$"], tmp_path)
    assert "<mtable" in html
    assert '<mo stretchy="true">{</mo>' in html


def test_cube_root_with_an_index(tmp_path):
    [html] = render([r"$\sqrt[3]{x}$"], tmp_path)
    assert "<mroot>" in html


def test_money_is_not_maths(tmp_path):
    [html] = render(["It costs $5 and $10 today"], tmp_path)
    assert "<math" not in html
    assert html == "<p>It costs $5 and $10 today</p>"


def test_maths_inside_code_stays_code(tmp_path):
    [html] = render(["Use `$x^2$` here"], tmp_path)
    assert "<math" not in html
    assert "<code>$x^2$</code>" in html


def test_maths_inside_a_fenced_block_stays_code(tmp_path):
    [html] = render(["```\n$x^2$\n```"], tmp_path)
    assert "<math" not in html
    assert "$x^2$" in html


def test_unknown_commands_show_the_source(tmp_path):
    [html] = render([r"\( \unknowncmd{x} \)"], tmp_path)
    assert "<math" not in html
    assert r"\unknowncmd" in html


def test_less_than_is_escaped_in_maths(tmp_path):
    [html] = render([r"\( a < b \)"], tmp_path)
    assert "&lt;" in html
    assert "<b" not in html


def test_text_in_maths_is_escaped(tmp_path):
    [html] = render([r"\( \text{<script>} \)"], tmp_path)
    assert "&lt;script&gt;" in html
    assert "<script" not in html


def test_output_is_escaped(tmp_path):
    [html] = render([r"$\text{<img src=x onerror=alert(1)>}$"], tmp_path)
    assert "<img" not in html
    assert "&lt;img" in html


def test_unclosed_paren_is_left_as_text(tmp_path):
    [html] = render([r"\( x^2"], tmp_path)
    assert html == r"<p>\( x^2</p>"


def test_plain_markdown_is_unchanged(tmp_path):
    [html] = render(["**bold** and a [link](https://example.org)"], tmp_path)
    assert html == (
        '<p><strong>bold</strong> and a <a href="https://example.org" target="_blank" '
        'rel="noopener noreferrer">link</a></p>'
    )


def test_underscores_inside_words_and_spaced_stars_are_not_emphasis(tmp_path):
    texts = ["my_var_name and file_name.py", "2 * 3 = 6 and 4 * 5 = 20, also 2*3*4",
             "_it_ and *it* and (_it_) and __init__", "x_1 + $y_2$ and a_b"]
    plain, product, marked, mixed = render(texts, tmp_path)
    assert plain == "<p>my_var_name and file_name.py</p>"
    assert product == "<p>2 * 3 = 6 and 4 * 5 = 20, also 2*3*4</p>"
    assert marked == "<p><em>it</em> and <em>it</em> and (<em>it</em>) and <strong>init</strong></p>"
    assert mixed == "<p>x_1 + <math><msub><mi>y</mi><mn>2</mn></msub></math> and a_b</p>"
