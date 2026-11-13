# Accessible form

A three-field form validated in the browser, with inline errors and a status message.

Run: open `index.html` in a browser, or serve the folder with `python3 -m http.server 8000`.

Each field has a real label; errors are linked with `aria-describedby`, marked with `aria-invalid` and announced through the first invalid field receiving focus. The success message uses `role="status"`.

Replace the sample text marked with `KIWIBRO:` comments and connect the valid branch in `form.js` to the project's endpoint. Palette and fonts are the custom properties at the top of `styles.css`.
