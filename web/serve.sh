#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: AGPL-3.0-only

# Static server for the app (WebGPU needs a secure context: localhost is fine).
# Sends Cache-Control: no-store so a rebuilt app.js / worker.js / pkg/*.wasm is
# picked up by a plain reload (Chrome otherwise caches module scripts and wasm
# heuristically and keeps running the old app).
PORT=${1:-8765}
if curl -fs "http://127.0.0.1:$PORT/index.html" >/dev/null 2>&1; then
    echo "already serving on http://localhost:$PORT/"; exit 0
fi
cd "$(dirname "$0")" && exec python3 - "$PORT" <<'PY'
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

NoCache.extensions_map.update({".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript"})
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), NoCache).serve_forever()
PY
