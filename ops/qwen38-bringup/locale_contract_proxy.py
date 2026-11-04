#!/usr/bin/env python3
"""Small OpenAI-compatible reverse proxy that injects the frozen locale contract."""
from __future__ import annotations

import argparse
import http.server
import json
import pathlib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (ROOT / "runs" / "requalification" /
                    "QWEN38-HAUHAUCS-LOCALE-CONTROL-2026-08-23" / "contract_v2.txt")


def inject_contract(payload: bytes, contract: str) -> bytes:
    body = json.loads(payload.decode("utf-8"))
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("chat payload must contain a messages list")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("every chat message must be an object")
    if messages and messages[0].get("role") == "system":
        existing = messages[0].get("content", "")
        if contract not in existing:
            messages[0] = {**messages[0], "content": contract + "\n\n" + existing}
    else:
        messages.insert(0, {"role": "system", "content": contract})
    body["messages"] = messages
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def make_handler(upstream: str, contract: str):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self) -> None:  # noqa: N802
            self._forward(None)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            if self.path.split("?", 1)[0] == "/v1/chat/completions":
                try:
                    payload = inject_contract(payload, contract)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    self.send_error(400, str(exc))
                    return
            self._forward(payload)

        def _forward(self, payload: bytes | None) -> None:
            target = upstream.rstrip("/") + self.path
            headers = {key: value for key, value in self.headers.items()
                       if key.casefold() not in {"host", "content-length", "connection"}}
            request = urllib.request.Request(target, data=payload, headers=headers,
                                             method=self.command)
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    self.send_response(response.status)
                    for key, value in response.headers.items():
                        if key.casefold() not in {"content-length", "transfer-encoding", "connection"}:
                            self.send_header(key, value)
                    self.send_header("X-Locale-Contract", "qwen38-ptbr-v2")
                    self.end_headers()
                    while chunk := response.read(64 * 1024):
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except urllib.error.HTTPError as exc:
                self.send_response(exc.code)
                self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
                self.send_header("X-Locale-Contract", "qwen38-ptbr-v2")
                self.end_headers()
                self.wfile.write(exc.read())
            except (OSError, urllib.error.URLError) as exc:
                self.send_error(502, f"upstream unavailable: {exc}")

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.client_address[0]} {fmt % args}", flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--contract", type=pathlib.Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    contract = args.contract.read_text(encoding="utf-8").strip()
    server = http.server.ThreadingHTTPServer(
        (args.listen, args.port), make_handler(args.upstream, contract))
    server.daemon_threads = True
    print(f"locale proxy http://{args.listen}:{args.port} -> {args.upstream}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
