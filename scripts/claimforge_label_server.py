#!/usr/bin/env python3
"""Local static server with a tiny annotation-save endpoint.

This serves the repository like `python3 -m http.server`, and additionally
accepts POST /api/save-labels to write browser labeler JSON back into the repo.
"""
import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_PATH = Path("annotations/claimforge-good-mouse-source-stain-275-slots.json")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(REPO), **kwargs)

    def _send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/save-labels":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            rel_path = Path(request.get("path") or DEFAULT_SAVE_PATH)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise ValueError("save path must be a safe repository-relative path")
            if rel_path.suffix.lower() != ".json":
                raise ValueError("save path must end in .json")

            payload = request.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")

            out_path = (REPO / rel_path).resolve()
            if not str(out_path).startswith(str(REPO.resolve()) + "/"):
                raise ValueError("save path escapes repository")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            out_path.write_text(text, encoding="utf-8")
            self._send_json(HTTPStatus.OK, {
                "ok": True,
                "path": str(rel_path),
                "bytes": len(text.encode("utf-8")),
            })
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {REPO} on http://localhost:{args.port}")
    print("POST /api/save-labels enabled")
    server.serve_forever()


if __name__ == "__main__":
    main()
