#!/usr/bin/env python3
"""{{purpose}}"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("{{name}}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="{{name}}", description=__doc__)
    # KIWIBRO: the arguments are the example's; give the script the ones it needs
    parser.add_argument("paths", nargs="+", type=Path, metavar="FILE", help="file to read")
    parser.add_argument("-o", "--output", type=Path, help="write the counts as JSON here")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every file")
    return parser


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


# KIWIBRO: this function is the work; count, parse, convert or fetch what the script is for
def summarise(paths: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"{path}: not a file")
        counts[str(path)] = count_lines(path)
        log.info("%s: %d lines", path, counts[str(path)])
    return counts


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    counts = summarise(args.paths)
    width = max((len(str(value)) for value in counts.values()), default=1)
    for name, value in counts.items():
        print(f"{value:>{width}}  {name}")
    print(f"{sum(counts.values()):>{width}}  total")
    if args.output:
        args.output.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        log.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
