#!/usr/bin/env python3
"""{{app_name}}: a GTK 4 / libadwaita application.

KIWIBRO: the window builds a list of notes as the example screen; keep the
window, header bar and action wiring and put your own widget in their place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

APP_ID = "{{app_id}}"
APP_NAME = "{{app_name}}"
STORE = Path(GLib.get_user_data_dir()) / APP_ID / "notes.json"


def load_notes() -> list[str]:
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        print(f"{APP_NAME}: cannot read {STORE}: {exc}", file=sys.stderr)
        return []
    return [str(item) for item in data] if isinstance(data, list) else []


def save_notes(notes: list[str]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")


class Window(Adw.ApplicationWindow):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.set_title(APP_NAME)
        self.set_default_size(520, 640)
        self.notes = load_notes()

        add_button = Gtk.Button(icon_name="list-add-symbolic")
        add_button.set_tooltip_text("Add Note")
        add_button.add_css_class("suggested-action")
        add_button.set_action_name("win.add")

        header = Adw.HeaderBar()
        header.pack_end(add_button)

        self.rows = Gtk.ListBox()
        self.rows.add_css_class("boxed-list")
        self.rows.set_selection_mode(Gtk.SelectionMode.NONE)
        self.rows.set_margin_top(12)
        self.rows.set_margin_bottom(12)
        self.rows.set_margin_start(12)
        self.rows.set_margin_end(12)

        scroller = Gtk.ScrolledWindow()
        scroller.set_child(self.rows)

        empty = Adw.StatusPage(
            icon_name="view-list-symbolic",
            title="No notes yet",
            description="Notes you add are kept on this computer.",
        )
        empty_button = Gtk.Button(label="Add Note")
        empty_button.add_css_class("suggested-action")
        empty_button.set_action_name("win.add")
        empty.set_child(empty_button)

        self.stack = Gtk.Stack()
        self.stack.add_named(empty, "empty")
        self.stack.add_named(scroller, "list")

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(header)
        toolbar.set_content(self.stack)
        self.set_content(toolbar)

        action = Gio.SimpleAction.new("add", None)
        action.connect("activate", self.on_add)
        self.add_action(action)

        for note in self.notes:
            self.rows.append(self.make_row(note))
        self.update_view()

    def make_row(self, note: str) -> Adw.ActionRow:
        # KIWIBRO: one row per item; swap ActionRow for the widget your data needs
        return Adw.ActionRow(title=note)

    def update_view(self) -> None:
        self.stack.set_visible_child_name("list" if self.notes else "empty")

    def on_add(self, *_args: object) -> None:
        entry = Gtk.Entry(placeholder_text="Note")
        entry.set_activates_default(True)
        dialog = Adw.AlertDialog(heading="New Note")
        dialog.set_extra_child(entry)
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("add", "Add")
        dialog.set_default_response("add")
        dialog.set_close_response("cancel")
        dialog.set_response_appearance("add", Adw.ResponseAppearance.SUGGESTED)
        dialog.connect("response", self.on_add_response, entry)
        dialog.present(self)

    def on_add_response(self, _dialog: Adw.AlertDialog, response: str, entry: Gtk.Entry) -> None:
        text = entry.get_text().strip()
        if response != "add" or not text:
            return
        # KIWIBRO: the primary action lands here; save, send or open your own thing
        self.notes.append(text)
        self.rows.append(self.make_row(text))
        save_notes(self.notes)
        self.update_view()


class App(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_args: self.quit())
        self.add_action(quit_action)
        self.set_accels_for_action("app.quit", ["<primary>q"])
        self.set_accels_for_action("win.add", ["<primary>n"])

    def do_activate(self) -> None:
        window = self.props.active_window
        if window is None:
            window = Window(application=self)
        window.present()


def main() -> int:
    return App().run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
