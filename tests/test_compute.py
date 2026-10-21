"""Offline tests for kiwibro.compute: exact math, unit and date evaluation, and table analysis.

No network and no writes outside pytest's tmp_path: tables live in a temporary workspace.
"""

from __future__ import annotations

import json
import types
from datetime import date

import pytest

from kiwibro.compute import TOOLS, evaluate
from kiwibro.sandbox import SandboxError, Workspace

TODAY = date(2026, 10, 22)
TOOL = {tool.name: tool for tool in TOOLS}
ANALYZE = TOOL["analyze_table"]
CALCULATE = TOOL["calculate"]


def result_of(expression: str) -> str:
    return evaluate(expression, today=TODAY)["result"]


@pytest.fixture
def ctx(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    return types.SimpleNamespace(workspace=workspace, memory=None, search=None, conversation_id="c1")


@pytest.fixture
def sales(ctx):
    """30 rows: country cycles TR/US/DE; amount is the integer 1..30 times 3 (unique, exact)."""
    countries = ["TR", "US", "DE"]
    rows = [(countries[i % 3], (i + 1) * 3, f"item {i + 1}") for i in range(30)]
    root = ctx.workspace.root
    lines = ["country,amount,item"] + [f"{c},{a},{item}" for c, a, item in rows]
    (root / "sales.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "sales.jsonl").write_text(
        "\n".join(json.dumps({"country": c, "amount": a, "item": item}) for c, a, item in rows) + "\n",
        encoding="utf-8",
    )
    return rows


def analyze(ctx, path: str = "sales.csv", **options):
    return ANALYZE.execute(ANALYZE.validate({"path": path, **options}), ctx)


# --- math --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+2", "4"),
        ("15% of 240", "36"),
        ("2^10", "1024"),
        ("sqrt(16)", "4"),
        ("0.1+0.2", "0.3"),
        ("10 // 3", "3"),
        ("2**100", "1267650600228229401496703205376"),
        ("1,234 + 1", "1235"),
        ("factorial(20)", "2432902008176640000"),
        ("what is 7 x 6?", "42"),
        ("7/7", "1"),
        ("10/4", "2.5"),
        ("5 % 3", "2"),
        ("100 + 15%", "100.15"),
    ],
)
def test_math_exact_results(expression, expected):
    assert result_of(expression) == expected


def test_math_one_third_is_twelve_significant_digits():
    assert result_of("1/3").startswith("0.333333")


@pytest.mark.parametrize("expression", ["1/0", "__import__('os')", "().__class__", "(lambda: 1)()", "a.b"])
def test_math_rejects_unsafe_or_invalid_input(expression):
    with pytest.raises(ValueError):
        evaluate(expression, today=TODAY)


def test_math_power_guard_refuses_huge_results():
    with pytest.raises(ValueError, match="too large"):
        evaluate("9**9**9", today=TODAY)


def test_math_domain_error_is_reported():
    with pytest.raises(ValueError, match="math domain error"):
        evaluate("sqrt(-4)", today=TODAY)


# --- units -------------------------------------------------------------------


def numeric_part(expression: str) -> float:
    return float(result_of(expression).split()[0])


def test_units_length_and_temperature_are_approximately_right():
    assert numeric_part("5 miles to km") == pytest.approx(8.04672)
    assert numeric_part("72 F to C") == pytest.approx(22.2222, rel=1e-5)
    assert numeric_part("100 km/h to mph") == pytest.approx(62.1371, rel=1e-5)
    assert numeric_part("1 acre in m2") == pytest.approx(4046.856, rel=1e-6)


def test_units_binary_and_bit_conversions_are_exact():
    assert result_of("1 GiB to MB") == "1073.741824 MB"
    assert numeric_part("8 bits to bytes") == 1


def test_units_reject_different_categories():
    with pytest.raises(ValueError, match="cannot convert mass to length"):
        evaluate("5 kg to km", today=TODAY)


def test_units_unknown_unit_lists_examples():
    with pytest.raises(ValueError, match="unknown unit 'parsec'"):
        evaluate("2 parsec to km", today=TODAY)


# --- dates -------------------------------------------------------------------


def test_date_days_between():
    assert result_of("days between 2026-01-01 and 2026-10-22") == "294 days"


def test_date_month_addition_clamps_to_month_end():
    assert "2026-02-28" in result_of("2026-01-31 + 1 month")


def test_date_weekday_of():
    assert "Thursday" in result_of("weekday of 2026-10-22")


def test_date_plus_days_reports_weekday():
    assert result_of("2026-10-22 + 90 days") == "2027-01-20 (Wednesday)"


def test_date_dotted_minus_year():
    assert result_of("22.10.2026 - 1 year").startswith("2025-10-22")


def test_date_relative_keywords_use_today():
    assert result_of("3 days ago").startswith("2026-10-19")
    assert result_of("in 2 weeks").startswith("2026-11-05")
    assert result_of("5 days from today").startswith("2026-10-27")


def test_evaluate_reports_kind_and_input():
    outcome = evaluate("  2+2  ", today=TODAY)
    assert outcome["kind"] == "math"
    assert outcome["input"] == "2+2"
    assert evaluate("72 F to C", today=TODAY)["kind"] == "units"
    assert evaluate("2026-10-22 + 1 day", today=TODAY)["kind"] == "date"


# --- calculate tool ----------------------------------------------------------


def test_calculate_tool_content_and_meta(ctx):
    args = CALCULATE.validate({"expression": "  15% of 240 "})
    outcome = CALCULATE.execute(args, ctx)
    assert outcome.ok is True
    assert outcome.content == "15% of 240 = 36"
    assert outcome.meta == {"kind": "math", "result": "36"}
    assert hasattr(outcome, "preview")


# --- analyze_table -----------------------------------------------------------


def test_profile_lists_columns_and_row_count(ctx, sales):
    outcome = analyze(ctx)
    assert outcome.ok is True
    assert "Rows: 30; columns: 3" in outcome.content
    for column in ("country", "amount", "item"):
        assert f"- {column} (" in outcome.content
    assert outcome.meta == {"rows": 30, "matched": 30}


def test_group_by_sum_matches_expected(ctx, sales):
    expected: dict[str, int] = {}
    for country, amount, _ in sales:
        expected[country] = expected.get(country, 0) + amount
    outcome = analyze(ctx, group_by="country", aggregate="sum:amount")
    for country, total in expected.items():
        assert f"| {country} | {total} |" in outcome.content
    assert "groups: 3" in outcome.content


def test_where_filters_rows(ctx, sales):
    expected = [row for row in sales if row[1] > 50 and row[0] == "TR"]
    outcome = analyze(ctx, where='amount > 50 and country == "TR"', columns="item,amount")
    assert f"{len(expected)} rows matched" in outcome.content
    assert outcome.meta["matched"] == len(expected)
    for _, _, item in expected:
        assert f"| {item} |" in outcome.content


def test_sort_desc_with_limit_returns_largest(ctx, sales):
    outcome = analyze(ctx, sort_by="amount desc", limit=3)
    top = [str(a) for _, a, _ in sorted(sales, key=lambda row: row[1], reverse=True)[:3]]
    positions = [outcome.content.index(f"| {value} |") for value in top]
    assert positions == sorted(positions)
    assert "| 90 |" in outcome.content and "| 3 |" not in outcome.content


def test_unknown_column_raises_value_error(ctx, sales):
    with pytest.raises(ValueError, match="unknown column 'nope'; columns are: country, amount, item"):
        analyze(ctx, where="nope > 1")


def test_path_outside_workspace_raises_sandbox_error(ctx):
    with pytest.raises(SandboxError):
        analyze(ctx, path="../outside.csv")


def test_jsonl_input_works(ctx, sales):
    outcome = analyze(ctx, path="sales.jsonl", group_by="country", aggregate="count,mean:amount")
    assert "groups: 3" in outcome.content
    assert "| TR | 10 |" in outcome.content


def test_where_supports_backticks_and_contains(ctx, sales):
    outcome = analyze(ctx, where='`item` contains "ITEM 1" and amount >= 3', columns="item")
    assert "rows matched" in outcome.content
    assert "| item 1 |" in outcome.content
    assert "| item 10 |" in outcome.content
    assert "| item 2 |" not in outcome.content


def test_validate_limit_bounds_and_coercion():
    with pytest.raises(ValueError):
        ANALYZE.validate({"path": "x.csv", "limit": 0})
    assert ANALYZE.validate({"path": "x.csv", "limit": "7"})["limit"] == 7


def test_slash_and_text_numbers_are_not_numeric(ctx):
    (ctx.workspace.root / "mixed.csv").write_text("period,note\n2026/10,a\n2026/11,b\n", encoding="utf-8")
    outcome = analyze(ctx, path="mixed.csv")
    assert "- period (text): 2 non-empty" in outcome.content


def test_bad_aggregate_is_rejected(ctx, sales):
    with pytest.raises(ValueError, match="bad aggregate"):
        analyze(ctx, aggregate="total:amount")


# --- every tool --------------------------------------------------------------


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
def test_validate_rejects_missing_required_argument(tool):
    with pytest.raises(ValueError):
        tool.validate({})


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
def test_execute_returns_result_object(tool, ctx, sales):
    args = {"expression": "2+2"} if tool.name == "calculate" else {"path": "sales.csv"}
    outcome = tool.execute(tool.validate(args), ctx)
    for field in ("ok", "content", "preview", "meta"):
        assert hasattr(outcome, field)
