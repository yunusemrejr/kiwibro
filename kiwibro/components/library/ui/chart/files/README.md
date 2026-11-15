# Bar and line chart

One SVG chart with axes, tick labels and a switch between bars and a line. The same values sit in a table under the chart for screen readers and for anyone who wants the numbers.

Run: open `index.html` in a browser, or serve the folder with `python3 -m http.server 8000`.

Put the values and their unit in `data.js` (marked `KIWIBRO:`); the axis and the table follow. The chart is redrawn at the width of its box, so it stays sharp on every screen. Palette and spacing are the custom properties at the top of `styles.css`, the same tokens as the other kits.

There is no build step, no library and no network request.
