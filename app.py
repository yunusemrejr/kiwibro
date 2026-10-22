"""Kiwibro entry point: start the local Flask server."""

from __future__ import annotations

import os
from pathlib import Path

from kiwibro import create_app, selfupdate

app = create_app()


def _busy() -> bool:
    agent = app.extensions["kiwibro"].agent
    return agent.active_turns() > 0 or bool(getattr(agent, "_extracting", None))


def main() -> None:
    selfupdate.set_process_name("kiwibro")
    host = os.environ.get("KIWIBRO_HOST", "127.0.0.1")
    port = int(os.environ.get("KIWIBRO_PORT", "8777"))
    updater = selfupdate.Updater(Path(__file__).resolve().parent, _busy)
    app.extensions["kiwibro_updater"] = updater
    if os.environ.get("KIWIBRO_AUTO_UPDATE", "1") != "0":
        updater.start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
