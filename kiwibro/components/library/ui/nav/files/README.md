# Responsive site header

A sticky header with a wordmark, four links, a skip link and a menu button that appears on phones.

Run: open `index.html` in a browser, or serve the folder with `python3 -m http.server 8000`.

Change the links in the `nav` list and copy the header block into every page of the site (marked `KIWIBRO:` in `index.html`). `nav.js` marks the link whose address matches the page with `aria-current="page"`, closes the menu on Escape or an outside click, and leaves the links visible if scripts are off. Palette and spacing are the custom properties at the top of `styles.css`, the same tokens as the other kits.

The menu collapses below 48 rem and opens as a list; above it the links sit in a row. There is no build step, no framework and no network request.
