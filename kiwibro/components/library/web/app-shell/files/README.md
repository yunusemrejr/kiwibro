# Single-page app shell

A framework-free app with hash-routed views, one state store, a fetch helper and local storage.

Run: `python3 -m http.server 8000` in this folder, then open `http://localhost:8000/`.

Test: `node --test` (runs `tests/store.test.mjs`).

Edit the view text marked with `KIWIBRO:` comments and point `api()` in `app.mjs` at the project's API. Palette and fonts are the custom properties at the top of `styles.css`.
