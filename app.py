"""Kiwibro entry point: start the local Flask server."""

from __future__ import annotations

import os

from kiwibro import create_app

app = create_app()


def main() -> None:
    host = os.environ.get("KIWIBRO_HOST", "127.0.0.1")
    port = int(os.environ.get("KIWIBRO_PORT", "8777"))
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
