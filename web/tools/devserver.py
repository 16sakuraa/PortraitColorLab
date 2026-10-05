"""
Local development server for the web app.

  python web/tools/devserver.py [port]

Serves web/ like GitHub Pages would, plus one test-only endpoint:
POST /__save/<name> stores the request body in web/test/out/<name>, so
browser results can be compared with the desktop app in Python. Binds to
127.0.0.1 only.
"""

from __future__ import annotations
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

WEB = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(WEB, "test", "out")


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if not self.path.startswith("/__save/"):
            self.send_error(404)
            return
        name = os.path.basename(self.path[len("/__save/"):])
        os.makedirs(OUT, exist_ok=True)
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(data)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"serving {WEB} on http://127.0.0.1:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=WEB)).serve_forever()
