# Image gallery

A grid of thumbnails that opens each picture in a viewer. Arrow keys, the Previous and Next buttons and a swipe on the picture all move between pictures; Escape or Close returns to the grid.

Run: open `index.html` in a browser, or serve the folder with `python3 -m http.server 8000`.

The six sample pictures are drawings inside the page (marked `KIWIBRO:`). Replace each one with an `<img>` that has an `alt` text and keep the button around it; the viewer copies whatever sits inside the button. Palette and spacing are the custom properties at the top of `styles.css`, the same tokens as the other kits.

The grid fits two columns at 360 px and grows with the width; the viewer fills the screen on phones. There is no build step, no framework and no network request.
