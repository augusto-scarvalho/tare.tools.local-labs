#!/usr/bin/env python3
"""OpenAI-compatible on-demand gateway for the qualified single-GPU fleet.

The public gateway owns port 8080. Exactly one llama-server child owns a private
loopback port. The JSON ``model`` field selects a frozen qualified card; switching
stops the old child before starting the new one, so two large models never overlap
in VRAM. Different cards may use different qualified llama-server builds.
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model_lifecycle.qualified_fleet import (  # noqa: E402
    DEFAULT_REGISTRY,
    build_backend_command,
    load_registry,
    public_card,
    resolve_model,
)

LOG = logging.getLogger("qualified-model-gateway")
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
ALLOWED_POST = {
    "/v1/chat/completions", "/v1/completions", "/completion", "/infill",
}


class FleetRuntime:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        backend_host: str,
        backend_port: int,
        state_dir: Path,
        load_timeout: float,
        stop_timeout: float,
    ) -> None:
        self.config = config
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.state_dir = state_dir
        self.load_timeout = load_timeout
        self.stop_timeout = stop_timeout
        self.request_lock = threading.RLock()
        self.process: subprocess.Popen[bytes] | None = None
        self.model_id: str | None = None
        self.requested_name: str | None = None
        self.last_switch_seconds: float | None = None
        self.last_error: str | None = None

    def backend_url(self, path: str) -> str:
        return f"http://{self.backend_host}:{self.backend_port}{path}"

    def backend_health(self, timeout: float = 2.0) -> bool:
        process = self.process
        if process is None or process.poll() is not None:
            return False
        try:
            with urllib_request.urlopen(self.backend_url("/health"), timeout=timeout) as response:
                body = json.load(response)
            return response.status == 200 and body.get("status") == "ok"
        except Exception:
            return False

    def _verify_identity(self, expected_path: str) -> bool:
        try:
            with urllib_request.urlopen(self.backend_url("/props"), timeout=3) as response:
                props = json.load(response)
            return props.get("model_path") == expected_path
        except Exception:
            return False

    def stop_backend(self) -> None:
        process = self.process
        self.process = None
        self.model_id = None
        self.requested_name = None
        if process is None or process.poll() is not None:
            return
        LOG.info("stopping backend pid=%s", process.pid)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            with contextlib.suppress(Exception):
                process.terminate()
        deadline = time.monotonic() + self.stop_timeout
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            LOG.warning("backend pid=%s exceeded graceful timeout; killing", process.pid)
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            with contextlib.suppress(Exception):
                process.wait(timeout=5)

    def ensure_model(self, requested: str | None) -> tuple[str, float]:
        model_id, card = resolve_model(self.config, requested)
        requested_name = requested or model_id
        if self.model_id == model_id and self.backend_health():
            self.requested_name = requested_name
            return model_id, 0.0

        self.stop_backend()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.state_dir / f"{model_id}.log"
        command = build_backend_command(
            model_id, card, host=self.backend_host, port=self.backend_port
        )
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in card["runtime"]["environment"].items()})
        environment.pop("GGML_CUDA_REGISTER_HOST", None)
        environment.pop("GGML_SCHED_PREFETCH_EXPERTS", None)
        LOG.info("loading %s: %s", model_id, " ".join(command))
        started = time.monotonic()
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        log_handle.close()
        self.process = process
        self.model_id = model_id
        self.requested_name = requested_name

        deadline = started + self.load_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.process = None
                self.model_id = None
                tail = ""
                with contextlib.suppress(Exception):
                    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
                raise RuntimeError(f"{model_id} exited while loading ({process.returncode})\n{tail}")
            if self.backend_health(timeout=2) and self._verify_identity(card["artifact"]["path"]):
                elapsed = time.monotonic() - started
                self.last_switch_seconds = elapsed
                self.last_error = None
                LOG.info("model %s ready in %.2fs", model_id, elapsed)
                return model_id, elapsed
            time.sleep(0.5)

        self.stop_backend()
        raise TimeoutError(f"{model_id} did not become healthy in {self.load_timeout:.0f}s")

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "role": "qualified-model-gateway",
            "current_model": self.model_id,
            "requested_name": self.requested_name,
            "backend_healthy": self.backend_health(),
            "backend_pid": self.process.pid if self.process and self.process.poll() is None else None,
            "backend_port": self.backend_port,
            "last_switch_seconds": self.last_switch_seconds,
            "last_error": self.last_error,
            "max_resident_models": 1,
            "available_models": sorted(self.config["models"]),
        }


RUNTIME: FleetRuntime


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def proxy_request(handler: BaseHTTPRequestHandler, body: bytes) -> None:
    connection = http.client.HTTPConnection(
        RUNTIME.backend_host, RUNTIME.backend_port, timeout=3600
    )
    headers = {
        key: value for key, value in handler.headers.items()
        if key.lower() not in HOP_HEADERS and key.lower() not in {"host", "content-length"}
    }
    headers["Content-Type"] = handler.headers.get("Content-Type", "application/json")
    headers["Content-Length"] = str(len(body))
    connection.request(handler.command, handler.path, body=body, headers=headers)
    response = connection.getresponse()
    handler.send_response(response.status)
    for key, value in response.getheaders():
        if key.lower() not in HOP_HEADERS and key.lower() != "content-length":
            handler.send_header(key, value)
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)
            handler.wfile.flush()
    finally:
        connection.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in {"/health", "/v1/health", "/fleet/status", "/v1/fleet/status"}:
            send_json(self, 200, RUNTIME.status())
            return
        if path in {"/models", "/v1/models"}:
            payload = []
            for model_id, card in sorted(RUNTIME.config["models"].items()):
                item = public_card(model_id, card)
                item.update({"object": "model", "owned_by": "tare.tools.local-labs"})
                payload.append(item)
            send_json(self, 200, {"object": "list", "data": payload})
            return
        if path == "/props":
            status = RUNTIME.status()
            if RUNTIME.model_id:
                _, card = resolve_model(RUNTIME.config, RUNTIME.model_id)
                status["model"] = public_card(RUNTIME.model_id, card)
            send_json(self, 200, status)
            return
        send_json(self, 404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path not in ALLOWED_POST:
            send_json(self, 404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 128 * 1024 * 1024:
            send_json(self, 400, {"error": {"message": "invalid body length"}})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            requested = payload.get("model") or RUNTIME.config["fleet"]["default_model"]
            model_id, _ = resolve_model(RUNTIME.config, requested)
        except (json.JSONDecodeError, AttributeError):
            send_json(self, 400, {"error": {"message": "body must be a JSON object"}})
            return
        except KeyError:
            send_json(self, 404, {"error": {
                "message": f"unknown or unqualified model: {requested}",
                "available_models": sorted(RUNTIME.config["models"]),
            }})
            return

        payload["model"] = model_id
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with RUNTIME.request_lock:
            try:
                RUNTIME.ensure_model(str(requested))
                proxy_request(self, body)
            except Exception as exc:
                RUNTIME.last_error = str(exc)
                LOG.exception("request failed for model=%s", requested)
                send_json(self, 503, {"error": {
                    "message": str(exc),
                    "type": "qualified_model_gateway_error",
                    "model": requested,
                }})


def port_is_free(host: str, port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex((host, port)) != 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=18080)
    parser.add_argument("--state-dir", default="/home/augus/.local/state/tare-qualified-models")
    parser.add_argument("--preload", default=None, help="model or alias to load before listening")
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--stop-timeout", type=float, default=90)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_registry(args.config)
    if not port_is_free(args.backend_host, args.backend_port):
        raise SystemExit(f"backend port {args.backend_port} is already occupied")

    global RUNTIME
    RUNTIME = FleetRuntime(
        config,
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        state_dir=Path(args.state_dir),
        load_timeout=args.load_timeout,
        stop_timeout=args.stop_timeout,
    )
    if args.preload:
        with RUNTIME.request_lock:
            RUNTIME.ensure_model(args.preload)

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    def shutdown(signum: int, _frame: Any) -> None:
        LOG.info("received signal %s", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    LOG.info("gateway listening on %s:%s; backend=%s:%s", args.host, args.port,
             args.backend_host, args.backend_port)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        with RUNTIME.request_lock:
            RUNTIME.stop_backend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
