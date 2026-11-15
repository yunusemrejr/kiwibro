# Dashboard page

Side navigation, three stat tiles, an SVG bar chart and a table, all built from `data.js`.

Run: open `index.html` in a browser, or serve the folder with `python3 -m http.server 8000`.

Edit the numbers, columns and rows in `data.js` (marked `KIWIBRO:`); the tiles are calculated from them. Palette, type and spacing are the custom properties at the top of `styles.css`, the same tokens as the other kits.

The navigation is a strip on phones and a column from 900 px. There is no build step, no framework and no network request.
