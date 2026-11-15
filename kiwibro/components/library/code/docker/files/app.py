"""{{name}} — a small web app that runs in the container."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME = "{{name}}"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self.reply(200, {"status": "ok", "name": NAME})
        elif path in ("/", "/index.html"):
            self.reply(200, {"message": f"{NAME} is running", "try": "/health"})
        else:
            self.reply(404, {"error": f"There is no route {path}."})

    def reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, pattern: str, *args: object) -> None:
        print(f"{self.address_string()} {pattern % args}", flush=True)


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"{NAME} listening on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
