# Sortable data table

A table that sorts when a column heading is pressed, filters as you type and shows five rows per page.

Run: open `index.html` in a browser, or serve the folder with `python3 -m http.server 8000`.

Put the records in `data.js` and describe the columns in `table.js` (both marked `KIWIBRO:`); `numeric: true` sorts a column as numbers. `PAGE_SIZE` sets the rows per page. Palette and spacing are the custom properties at the top of `styles.css`, the same tokens as the other kits.

There is no build step, no framework and no network request.
