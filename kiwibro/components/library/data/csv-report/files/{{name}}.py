#!/usr/bin/env python3
"""Report on a CSV file: group rows by one column and total a numeric column."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

log = logging.getLogger("{{name}}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="{{name}}", description=__doc__)
    # KIWIBRO: the arguments are the example's; give the script the ones it needs
    parser.add_argument("input", type=Path, metavar="CSV", help="CSV file to read")
    parser.add_argument("--group", required=True, metavar="COLUMN", help="column to group by")
    parser.add_argument("--sum", required=True, metavar="COLUMN", help="numeric column to total")
    parser.add_argument("--out", type=Path, help="write the Markdown report here")
    return parser


def require_columns(found: list[str], wanted: str) -> None:
    if wanted not in found:
        raise SystemExit(f"column {wanted!r} not found; the file has: {', '.join(found)}")


# KIWIBRO: this function is the work; group, total or count what the script is for
def group_totals(rows: list[dict[str, str]], group: str, total: str) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = {}
    for row in rows:
        try:
            value = Decimal(row[total])
        except InvalidOperation:
            raise SystemExit(f"{total!r} must hold numbers; got {row[total]!r}") from None
        totals[row[group]] = totals.get(row[group], Decimal("0")) + value
    return totals


def render_report(source: Path, totals: dict[str, Decimal], total_column: str) -> str:
    lines = [f"# Report: {source.name}", ""]
    lines.append(f"| Group | Total {total_column} |")
    lines.append("| --- | ---: |")
    for group in sorted(totals):
        lines.append(f"| {group} | {totals[group]:.2f} |")
    lines.append(f"| **Total** | **{sum(totals.values()):.2f}** |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if not args.input.is_file():
        print(f"{{name}}: {args.input} is not a file", file=sys.stderr)
        return 1
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        found = reader.fieldnames or []
        require_columns(found, args.group)
        require_columns(found, args.sum)
        rows = list(reader)
    totals = group_totals(rows, args.group, args.sum)
    report = render_report(args.input, totals, args.sum)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        log.info("wrote %s", args.out)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
