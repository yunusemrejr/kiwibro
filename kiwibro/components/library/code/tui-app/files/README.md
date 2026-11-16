# {{title}}

A terminal browser for {{items}}, in Python with curses and nothing else.

    python3 app.py       python3 -m pytest -q

`items.json` holds records with `title`, `status` (`doing`, `todo`, `blocked`,
`done`), `tags`, `date` and `body`. Replace it with your own list of
`{"title": ...}` records; for other fields change `load_items`, `detail_lines`
and `sort_items` in `app.py`.

`j`/`k` or arrows move, `PgUp`/`PgDn` page, `g`/`G` jump to the ends,
`Enter`/`Tab` switch panes, `/` filters as you type (`Esc` clears), `s` sorts,
`?` shows help, `q` quits. Below 40x12 the app asks for more room.
