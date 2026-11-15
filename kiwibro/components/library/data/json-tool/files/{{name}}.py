#!/usr/bin/env python3
"""Reshape a JSON file: pick and rename fields, filter records, write JSON or CSV."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("{{name}}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="{{name}}", description=__doc__)
    # KIWIBRO: the arguments are the example's; give the script the ones it needs
    parser.add_argument("input", type=Path, metavar="JSON", help="JSON file to read")
    parser.add_argument("--pick", metavar="a,b,c", help="comma-separated fields to keep")
    parser.add_argument("--rename", metavar="old=new", help="rename one field (repeatable)",
                        action="append", default=[])
    parser.add_argument("--where", metavar="field=value", help="keep records where a field equals a value")
    parser.add_argument("--format", choices=("json", "csv"), default="json", help="output format")
    parser.add_argument("--out", type=Path, help="write the result here instead of printing it")
    return parser


def parse_pairs(values: list[str], sep: str, what: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values:
        if sep not in value:
            raise SystemExit(f"{what} needs '{sep}' in it; got {value!r}")
        key, _, item = value.partition(sep)
        pairs[key] = item
    return pairs


# KIWIBRO: this function is the work; pick, rename or filter what the script is for
def reshape(records: list[dict], picks: list[str] | None, renames: dict[str, str],
            where: dict[str, str]) -> list[dict]:
    out = []
    for record in records:
        if any(str(record.get(field)) != value for field, value in where.items()):
            continue
        kept = {field: record[field] for field in (picks or record) if field in record}
        out.append({renames.get(field, field): value for field, value in kept.items()})
    return out


def to_csv_text(records: list[dict]) -> str:
    if not records:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if not args.input.is_file():
        print(f"{{name}}: {args.input} is not a file", file=sys.stderr)
        return 1
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        print(f"{{name}}: {args.input} must hold a list of objects", file=sys.stderr)
        return 1
    if data:
        found = sorted({key for item in data for key in item})
        for field in ([item for item in args.pick.split(",") if item] if args.pick else []):
            if field not in found:
                raise SystemExit(f"field {field!r} not found; the file has: {', '.join(found)}")
    picks = [item for item in args.pick.split(",") if item] if args.pick else None
    renames = parse_pairs(args.rename, "=", "--rename")
    where = parse_pairs([args.where] if args.where else [], "=", "--where")
    result = reshape(data, picks, renames, where)
    if args.format == "csv":
        text = to_csv_text(result)
    else:
        text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        log.info("wrote %s", args.out)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
