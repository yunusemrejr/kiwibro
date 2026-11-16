"""{{title}}: browse {{items}} in a terminal. Rules here are plain functions, checked by
test_app.py without a terminal; screen.py draws them with curses."""

import json
import sys
import textwrap
import unicodedata
from collections import Counter
from pathlib import Path

TITLE, NOUN, ACCENT = "{{title}}", "{{items}}", "{{accent}}"
DATA = Path(__file__).with_name("items.json")
MIN_WIDTH, MIN_HEIGHT, WIDE = 40, 12, 70
SORTS, STATUSES = ("date", "title", "status"), ("doing", "todo", "blocked", "done")


class ItemError(Exception):
    """items.json cannot be used; the text says what to fix."""


def load_items(path):
    """The records in path, each with title, status, tags, date and body."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ItemError(f"{path.name} is missing.") from None
    except (OSError, ValueError) as exc:
        raise ItemError(f"{path.name} could not be read: {exc}") from None
    records = raw.get("items") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ItemError(f'{path.name} needs a list of records with a "title".')
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict) or not str(record.get("title") or "").strip():
            raise ItemError(f"record {number} in {path.name} has no title.")
    return [{"title": str(r["title"]).strip(), "status": str(r.get("status") or "todo").strip().lower(),
             "tags": [str(t) for t in r.get("tags") or []], "date": str(r.get("date") or "").strip(),
             "body": str(r.get("body") or "").strip()} for r in records]


def width_of(text):
    """Columns text takes: two for a wide character, none for a combining mark."""
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1
               for c in text)


def fit(text, width):
    """text as exactly width columns: cut with an ellipsis when long, padded when short."""
    if width <= 0:
        return ""
    if width_of(text) <= width:
        return text + " " * (width - width_of(text))
    end = 0
    while end < len(text) and width_of(text[:end + 1]) <= width - 1:
        end += 1
    return text[:end] + "…" + " " * (width - 1 - width_of(text[:end]))


def filter_items(items, query):
    """Items whose text holds every word of query, in any case; no query keeps them all."""
    words = query.casefold().split()
    return [i for i in items if all(w in " ".join(
        [i["title"], i["status"], i["date"], *i["tags"], i["body"]]).casefold() for w in words)]


def sort_items(items, mode):
    """date: newest first; title: alphabetical; status: doing, todo, blocked, done."""
    if mode == "title":
        return sorted(items, key=lambda i: i["title"].casefold())
    if mode == "status":
        return sorted(items, key=lambda i: (STATUSES.index(i["status"]) if i["status"] in STATUSES
                                            else 9, i["title"].casefold()))
    return sorted(items, key=lambda i: i["date"], reverse=True)


def clamp(selected, count):
    return min(max(selected, 0), max(count - 1, 0))


def scroll_for(selected, start, count, height):
    """First row of a window height rows tall that shows selected, moved as little as it can."""
    height = min(max(height, 1), max(count, 1))
    start = clamp(start, max(count - height, 0) + 1)
    if selected < start:
        return selected
    return selected - height + 1 if selected >= start + height else start


def layout(width, height):
    """(listing, detail, side_by_side, message), a box being (y, x, height, width): header on
    row 0, status bar on the last row, panes side by side from WIDE and stacked below."""
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return (0, 0, 0, 0), (0, 0, 0, 0), False, (
            f"Make the window larger: {MIN_WIDTH}x{MIN_HEIGHT} needed, this is {width}x{height}.")
    body = height - 2
    if width >= WIDE:
        left = max(24, width * 2 // 5)
        return (1, 0, body, left), (1, left + 2, body, width - left - 2), True, ""
    rows = max(3, (body - 1) * 3 // 5)
    return (1, 0, rows, width), (rows + 2, 0, body - rows - 1, width), False, ""


def header(items, total, query, mode):
    """Title and counts for the left end of the header line, filter and sort for the right."""
    counts = Counter(i["status"] for i in items)
    left = f"{TITLE}   {len(items)} {NOUN}" + (f" of {total}" if len(items) != total else "")
    order = [s for s in STATUSES if s in counts] + sorted(s for s in counts if s not in STATUSES)
    if order:
        left += "   " + " · ".join(f"{counts[s]} {s}" for s in order)
    return left, (f"filter: {query.strip()}   " if query.strip() else "") + f"sort: {mode}"


def detail_lines(item, width):
    """The detail rows as (text, role), the body wrapped to width columns."""
    meta = [p for p in (item["date"], ", ".join(item["tags"])) if p]
    rows = [(item["title"], "heading"), (item["status"], "status"), ("   ".join(meta), "muted"), ("", "text")]
    return rows + [(line, "text") for line in textwrap.wrap(item["body"], max(width, 8)) or [""]]


def main():
    try:
        items = load_items(DATA)
    except ItemError as exc:
        print(f"{TITLE}: {exc}", file=sys.stderr)
        return 1
    import screen  # imported here so the rules load without a terminal

    return screen.main(items)


if __name__ == "__main__":
    sys.exit(main())
