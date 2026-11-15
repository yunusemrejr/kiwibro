# Modal dialog and toasts

A confirmation dialog built on the native `dialog` element and short messages that disappear after four seconds.

Run: open `index.html` in a browser, or serve the folder with `python3 -m http.server 8000`.

The browser keeps focus inside the open dialog, closes it on Escape and returns focus to the button that opened it. Add a dialog by copying the `dialog` block and giving a button `data-open="its-id"`. Call `toast("Saved.")` or `toast("Could not save.", "error")` from any script; a button with `data-toast="text"` shows one without code. Edit the text marked `KIWIBRO:` in `index.html`.

On phones the dialog is a sheet at the bottom of the screen; from 640 px it is a centred box. There is no build step, no framework and no network request.
