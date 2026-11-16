"""{{title}}: the curses screen. app.py holds the rules it draws."""

import curses
import sys

from app import (ACCENT, NOUN, SORTS, TITLE, clamp, detail_lines, filter_items,
                 fit, header, layout, scroll_for, sort_items, width_of)

HINTS = "j/k move  PgUp/PgDn  g/G  / filter  s sort  Tab pane  ? help  q quit"
HELP = ("j k  Up Down   move, or scroll the detail", "PgUp PgDn     a page at a time",
        "g  G          first and last item", "Enter  Tab    the other pane",
        "/             filter; Esc clears it", "s             sort: date, title, status",
        "?  this help   q  quit")
STATUS = {"doing": curses.COLOR_YELLOW, "blocked": curses.COLOR_RED, "done": curses.COLOR_GREEN}
MOVES = {curses.KEY_UP: -1, ord("k"): -1, curses.KEY_DOWN: 1, ord("j"): 1}


def begin_colours(accent):
    """A role -> attribute map; bold and reverse stand in when the terminal has no colour."""
    roles = {"accent": curses.A_BOLD, "selected": curses.A_REVERSE | curses.A_BOLD,
             "heading": curses.A_BOLD, "muted": curses.A_DIM, **{s: curses.A_BOLD for s in STATUS}}
    colours = {"accent": getattr(curses, "COLOR_" + accent.strip().upper(), curses.COLOR_CYAN),
               "muted": -1, **STATUS}
    if curses.has_colors() and curses.COLORS >= 8:
        try:
            curses.start_color()
            curses.use_default_colors()
            for n, name in enumerate(colours, 1):
                curses.init_pair(n, colours[name], -1)
            roles.update({name: curses.color_pair(n) for n, name in enumerate(colours, 1)})
            roles["selected"] = roles["heading"] = roles["accent"] | curses.A_BOLD
        except curses.error:
            pass
    return roles


class App:
    """Screen state and keys; every rule it uses is a function in app.py."""

    def __init__(self, win, items):
        self.win, self.items = win, items
        self.query, self.mode, self.focus = "", SORTS[0], "list"
        self.selected = self.list_top = self.detail_top = 0
        self.filtering = self.help = False
        self.visible, self.r = [], begin_colours(ACCENT)
        self.refresh(False)

    def refresh(self, keep=True):
        chosen = self.visible[self.selected]["title"] if keep and self.visible else None
        self.visible = sort_items(filter_items(self.items, self.query), self.mode)
        titles = [i["title"] for i in self.visible]
        self.selected = titles.index(chosen) if chosen in titles else clamp(self.selected, len(titles))
        self.detail_top = 0

    def put(self, y, x, text, width, role="text"):
        try:
            self.win.addnstr(y, x, fit(text, width), width, self.r.get(role, 0))
        except curses.error:
            pass  # the last cell of a full screen refuses the write

    def draw(self):
        self.win.erase()
        height, width = self.win.getmaxyx()
        listing, detail, side_by_side, message = layout(width, height)
        if message:
            self.put(height // 2, 1, message, width - 2, "heading")
            self.put(height - 1, 1, "q quits", width - 2, "muted")
        else:
            left, right = header(self.visible, len(self.items), self.query, self.mode)
            if width_of(left) + width_of(right) + 3 > width:  # narrow: keep the title and the count
                left = "   ".join(left.split("   ")[:2])
            room = max(width - width_of(right) - 2, 1)
            self.put(0, 1, left, room)
            self.put(0, room + 1, right, width - room - 2, "muted")
            self.draw_items(listing)
            self.draw_detail(detail, side_by_side)
            hint = "filter: " + (self.query or "type, Esc clears") if self.filtering else HINTS
            self.put(height - 1, 1, hint, width - 12, "muted")
            self.put(height - 1, width - 9, self.focus, 8, "accent")
            if self.help:
                self.draw_help(width, height)
        self.win.refresh()

    def draw_items(self, listing):
        top, left, height, width = listing
        self.list_top = scroll_for(self.selected, self.list_top, len(self.visible), height)
        if not self.visible:
            self.put(top + height // 2, left + 1, f"No {NOUN} " + (
                "match that filter. Esc clears it." if self.query else "in items.json."), width - 2, "muted")
            return
        for row, item in enumerate(self.visible[self.list_top:self.list_top + height]):
            chosen = self.list_top + row == self.selected
            status = item["status"] if item["status"] in STATUS else "muted"
            self.put(top + row, left, " " + item["title"], width - 8, "selected" if chosen else "text")
            self.put(top + row, left + width - 8, item["status"], 8, "selected" if chosen else status)

    def draw_detail(self, detail, side_by_side):
        top, left, height, pane = detail
        if not self.visible:
            return
        item = self.visible[self.selected]
        rows = detail_lines(item, max(pane - 2, 8))
        self.detail_top = min(self.detail_top, max(len(rows) - height, 0))
        for row, (text, role) in enumerate(rows[self.detail_top:self.detail_top + height]):
            self.put(top + row, left + 1, text, max(pane - 2, 1),
                     item["status"] if role == "status" and item["status"] in STATUS else role)
        for row in range(height if side_by_side else 0):
            self.put(top + row, left - 1, "│", 1, "muted")
        if not side_by_side:
            self.put(top - 1, left, "─" * pane, pane, "muted")

    def draw_help(self, width, height):
        rows, panel = [TITLE] + list(HELP), min(max(width - 4, 20), 60)
        top, left = max((height - len(rows) - 2) // 2, 0), max((width - panel) // 2, 0)
        self.put(top, left, "┌" + "─" * (panel - 2) + "┐", panel, "muted")
        self.put(top + len(rows) + 1, left, "└" + "─" * (panel - 2) + "┘", panel, "muted")
        for row, line in enumerate(rows, 1):
            self.put(top + row, left, "│" + fit(line, panel - 4) + "│", panel,
                     "heading" if row == 1 else "text")

    def move(self, step):
        if self.focus == "detail" and self.visible:
            self.detail_top = max(self.detail_top + step, 0)
        else:
            self.selected = clamp(self.selected + step, len(self.visible))
            self.detail_top = 0

    def key(self, key):
        """Handle one key; False means quit. Resize and unknown keys just redraw."""
        if self.help:
            self.help = False
        elif self.filtering:
            if key == 27:
                self.query, self.filtering = "", False
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                self.query = self.query[:-1]
            elif key in (10, 13):
                self.filtering = False
            elif 32 <= key < 127:
                self.query += chr(key)
            self.refresh(False)
        elif key == ord("q"):
            return False
        elif key in MOVES:
            self.move(MOVES[key])
        elif key in (curses.KEY_PPAGE, curses.KEY_NPAGE):
            page = max(layout(*self.win.getmaxyx())[0][2] - 1, 1)
            self.move(-page if key == curses.KEY_PPAGE else page)
        elif key in (ord("g"), ord("G")):
            self.move(-len(self.visible) if key == ord("g") else len(self.visible))
        elif key in (10, 13, 9):
            self.focus = "detail" if self.focus == "list" else "list"
        elif key == ord("/"):
            self.filtering = True
        elif key == ord("?"):
            self.help = True
        elif key == ord("s"):
            self.mode = SORTS[(SORTS.index(self.mode) + 1) % len(SORTS)]
            self.refresh()
        return True

    def run(self):
        try:
            curses.curs_set(0)
            self.win.keypad(True)
        except curses.error:
            pass
        self.draw()
        try:
            while self.key(self.win.getch()):
                self.draw()
        except KeyboardInterrupt:
            return


def main(items):
    try:
        curses.wrapper(lambda win: App(win, items).run())
    except curses.error as exc:
        print(f"{TITLE}: this terminal cannot run the app: {exc}", file=sys.stderr)
        return 1
    return 0
