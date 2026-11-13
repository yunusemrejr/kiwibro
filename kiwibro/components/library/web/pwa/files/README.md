# {{app_name}} — installable offline app

A browser app with a web manifest, a service worker and an offline page. Notes are saved on the device.

Run: `python3 -m http.server 8000`, then open `http://localhost:8000/`. Service workers need localhost or HTTPS.

Test: load the page once, then set the browser to offline and reload. The app should open and keep the saved notes; an uncached address shows `offline.html`.

Change the text marked with `KIWIBRO:` comments and the palette at the top of `styles.css`. When you add or rename a shell file, update `FILES` in `sw.js` and raise `CACHE` so browsers fetch the new version.
