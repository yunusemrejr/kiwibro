# Settings screen

Three groups of options (display, notifications, profile) with switches, selects and a text field. Every change is saved at once in the browser's local storage and restored when the page loads.

Run: open `index.html` in a browser, or serve the folder with `python3 -m http.server 8000`.

Add an option by adding a control with a `name` in `index.html` and a default for that name in `settings.js` (both marked `KIWIBRO:`); read the saved values anywhere with `JSON.parse(localStorage.getItem("settings"))`. Text size and compact lists already change the page; the other options only store a value. Palette and spacing are the custom properties at the top of `styles.css`, the same tokens as the other kits.

There is no build step, no framework and no network request.
