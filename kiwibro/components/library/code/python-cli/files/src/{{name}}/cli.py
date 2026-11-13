"""Command-line entry point for {{name}}."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FILE_ERROR = 1


def count_text(text: str) -> tuple[int, int, int]:
    """Return (lines, words, characters) for text."""
    lines = len(text.splitlines())
    words = len(text.split())
    return lines, words, len(text)


def longest_line(text: str) -> str:
    """Return the longest line of text, or an empty string."""
    return max(text.splitlines(), key=len, default="")


# KIWIBRO: replace these subcommands and options with your tool's own
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="{{name}}", description="Report on text files."
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="log each file as it is read"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="print lines, words and characters per file")
    stats.add_argument("paths", nargs="+", type=Path, metavar="FILE")

    longest = sub.add_parser("longest", help="print the longest line of each file")
    longest.add_argument("paths", nargs="+", type=Path, metavar="FILE")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tool; returns the process exit code."""
    args = build_parser().parse_args(argv)
    # force=True replaces handlers an embedding process already installed, so
    # -v works the same under pytest and from a plain shell.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    totals = [0, 0, 0]
    for path in args.paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(
                f"{{name}}: cannot read {path}: {exc.strerror or exc}",
                file=sys.stderr,
            )
            return EXIT_FILE_ERROR
        log.debug("read %s (%d characters)", path, len(text))

        if args.command == "stats":
            counts = count_text(text)
            totals = [total + count for total, count in zip(totals, counts)]
            print(f"{counts[0]:>7} {counts[1]:>7} {counts[2]:>7}  {path}")
        else:
            print(f"{path}: {longest_line(text)}")

    if args.command == "stats" and len(args.paths) > 1:
        print(f"{totals[0]:>7} {totals[1]:>7} {totals[2]:>7}  total")
    return EXIT_OK
