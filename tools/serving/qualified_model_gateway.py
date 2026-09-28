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
import ipaddress
import json
import logging
import os
from pathlib import Path
import secrets
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
    backend_kind,
    build_backend_command,
    host_mode,
    load_registry,
    ninfer_chat_request,
    public_card,
    resolve_model,
)
from model_lifecycle.fleet_count import (  # noqa: E402
    BINDING_FIELD, BindingMismatch, check_binding, count_request, effective_profile,
)
from model_lifecycle.gpu_lease import SharedGpuLease  # noqa: E402
from model_lifecycle.comfy_routes import (  # noqa: E402
    ComfyBackendError,
    ComfyRouteError,
    CoordinationError,
    ImageValidationError,
    OutputNotFoundError,
    UncertainSubmissionError,
    cancel_image_job,
    get_image_job_status,
    get_image_model_catalog,
    get_image_public_cards,
    get_job_output_image,
    load_image_routes,
    submit_image_job,
    validate_comfy_url,
)

DEFAULT_IMAGE_ROUTES = REPO_ROOT / "config" / "comfy_image_routes.json"

LOG = logging.getLogger("qualified-model-gateway")
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
ALLOWED_POST = {
    "/v1/chat/completions", "/v1/completions", "/completion", "/infill",
}
FLEET_POST = {'/v1/fleet/count', '/v1/fleet/profile'}
INTERNAL_POST = {'/internal/gpu/yield'}


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
        gpu_lock: Path | str | None = None,
        route_timeout: float = 60.0,
        gpu_lease: SharedGpuLease | None = None,
        comfy_url: str | None = None,
        image_routes_path: Path | str | None = None,
    ) -> None:
        self.config = config
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.state_dir = state_dir
        self.load_timeout = load_timeout
        self.stop_timeout = stop_timeout
        self.route_timeout = route_timeout
        if gpu_lease is not None:
            self.gpu_lease = gpu_lease
        elif gpu_lock is not None:
            self.gpu_lease = SharedGpuLease(Path(gpu_lock), timeout=route_timeout)
        else:
            self.gpu_lease = SharedGpuLease(None, timeout=route_timeout)
        self.request_lock = threading.RLock()
        self.process: subprocess.Popen[bytes] | None = None
        self.model_id: str | None = None
        self.requested_name: str | None = None
        self.last_switch_seconds: float | None = None
        self.last_error: str | None = None
        self.last_used = time.monotonic()
        if comfy_url is not None:
            self.comfy_url: str | None = validate_comfy_url(comfy_url)
            routes_file = Path(image_routes_path) if image_routes_path else DEFAULT_IMAGE_ROUTES
            self.image_routes: dict[str, Any] | None = load_image_routes(routes_file)
        else:
            self.comfy_url = None
            self.image_routes = None

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

    def _verify_identity(self, model_id: str, card: dict[str, Any]) -> bool:
        try:
            if backend_kind(card) in ("ninfer", "strata"):
                # No /props: the backend serves what the gateway started; NInfer under --model-id,
                # Strata under the model_name its config declares.
                expected = model_id if backend_kind(card) == "ninfer" else card["runtime"]["model_name"]
                with urllib_request.urlopen(self.backend_url("/v1/models"), timeout=3) as response:
                    return [m.get("id") for m in json.load(response).get("data", [])] == [expected]
            with urllib_request.urlopen(self.backend_url("/props"), timeout=3) as response:
                props = json.load(response)
            return props.get("model_path") == card["artifact"]["path"]
        except Exception:
            return False

    def stop_backend(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            self.process = None
            self.model_id = None
            self.requested_name = None
            return
        LOG.info("stopping backend pid=%s", process.pid)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            with contextlib.suppress(Exception):
                process.terminate()
        deadline = time.monotonic() + self.stop_timeout
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            LOG.warning("backend pid=%s exceeded graceful timeout; killing", process.pid)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                with contextlib.suppress(Exception):
                    process.kill()
            kill_deadline = time.monotonic() + 5.0
            while process.poll() is None and time.monotonic() < kill_deadline:
                time.sleep(0.1)
        if process.poll() is None:
            self.last_error = f"failed to stop backend pid={process.pid}"
            raise RuntimeError(f"backend process {process.pid} could not be stopped")
        self.process = None
        self.model_id = None
        self.requested_name = None

    def unload_if_idle(self) -> bool:
        """Stop a resident backend whose card sets idle_unload_seconds once it has been idle that long."""
        card = self.config["models"].get(self.model_id) if self.model_id else None
        limit = card and card["runtime"].get("idle_unload_seconds")
        if not limit or self.process is None or time.monotonic() - self.last_used < limit:
            return False
        if not self.request_lock.acquire(blocking=False):   # a request is running: not idle
            return False
        try:
            idle = time.monotonic() - self.last_used
            if self.model_id is None or idle < limit:
                return False
            LOG.info("unloading %s after %.0fs idle (limit %ss)", self.model_id, idle, limit)
            self.stop_backend()
            return True
        finally:
            self.request_lock.release()

    def ensure_model(self, requested: str | None) -> tuple[str, float]:
        model_id, card = resolve_model(self.config, requested)
        requested_name = requested or model_id
        if self.model_id == model_id and self.backend_health():
            self.requested_name = requested_name
            return model_id, 0.0

        self.stop_backend()
        wait_until_bindable(self.backend_host, self.backend_port)
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
        self.last_used = time.monotonic()
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
            if self.backend_health(timeout=2) and self._verify_identity(model_id, card):
                elapsed = time.monotonic() - started
                self.last_switch_seconds = elapsed
                self.last_error = None
                LOG.info("model %s ready in %.2fs", model_id, elapsed)
                return model_id, elapsed
            time.sleep(0.5)

        self.stop_backend()
        raise TimeoutError(f"{model_id} did not become healthy in {self.load_timeout:.0f}s")

    def status(self) -> dict[str, Any]:
        lease = getattr(self, "gpu_lease", None)
        coord = lease.status() if lease is not None else {"enabled": False}
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
            "gpu_coordination": coord,
            "resident_readout": "pid-bound-v1",
        }


RUNTIME: FleetRuntime


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def proxy_request(handler: BaseHTTPRequestHandler, body: bytes, *, fleet_observation=None, timeout=3600) -> None:
    connection = http.client.HTTPConnection(
        RUNTIME.backend_host, RUNTIME.backend_port, timeout=timeout
    )
    headers = {
        key: value for key, value in handler.headers.items()
        if key.lower() not in HOP_HEADERS and key.lower() not in {"host", "content-length"}
    }
    headers["Content-Type"] = handler.headers.get("Content-Type", "application/json")
    headers["Content-Length"] = str(len(body))
    connection.request(handler.command, handler.path, body=body, headers=headers)
    response = connection.getresponse()
    streaming = response.getheader('Content-Type','').split(';',1)[0].strip() == 'text/event-stream'
    if fleet_observation is not None and not streaming:
        try:
            raw = response.read(8*1024*1024+1)
            if len(raw) > 8*1024*1024:
                raise ValueError('fleet_generation_response_over_budget')
            payload = json.loads(raw)
            payload['tare_fleet_observation'] = fleet_observation
            send_json(handler, response.status, payload)
            return
        finally:
            connection.close()
    handler.send_response(response.status)
    for key, value in response.getheaders():
        if key.lower() not in HOP_HEADERS and key.lower() != "content-length":
            handler.send_header(key, value)
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        if fleet_observation is not None:
            # The same request lock covers binding verification and forwarding.
            # This is route evidence, not completion or token usage.
            handler.wfile.write(b'data: '+json.dumps({'choices':[],
                'tare_fleet_observation':fleet_observation}).encode()+b'\n\n')
            handler.wfile.flush()
        while True:
            chunk = response.read1(4096)
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
        if path == "/v1/fleet/recommend":
            if "host_modes" not in RUNTIME.config["fleet"]:
                send_json(self, 404, {"error": {"message": "host modes not configured", "type": "not_found"}})
                return
            try:
                status = json.loads(Path(RUNTIME.config["fleet"]["host_modes"]["status_path"]).read_text())
            except (OSError, ValueError):
                status = None
            send_json(self, 200, host_mode(RUNTIME.config, status, time.time()))
            return
        if path in {"/models", "/v1/models"}:
            payload = []
            for model_id, card in sorted(RUNTIME.config["models"].items()):
                item = public_card(model_id, card)
                item.update({"object": "model", "owned_by": "tare.tools.local-labs"})
                payload.append(item)
            image_routes = getattr(RUNTIME, "image_routes", None)
            if image_routes is not None:
                payload.extend(get_image_public_cards(image_routes))
            send_json(self, 200, {"object": "list", "data": payload})
            return
        if path == "/props":
            status = RUNTIME.status()
            if RUNTIME.model_id:
                _, card = resolve_model(RUNTIME.config, RUNTIME.model_id)
                status["model"] = public_card(RUNTIME.model_id, card)
            send_json(self, 200, status)
            return
        if path in {"/images/models", "/v1/images/models"}:
            comfy_url = getattr(RUNTIME, "comfy_url", None)
            image_routes = getattr(RUNTIME, "image_routes", None)
            if not comfy_url or not image_routes:
                send_json(self, 404, {"error": {"message": "image routing not configured", "type": "not_found"}})
                return
            catalog = get_image_model_catalog(comfy_url, image_routes)
            send_json(self, 200, catalog)
            return
        if path.startswith("/v1/images/jobs/") or path.startswith("/images/jobs/"):
            comfy_url = getattr(RUNTIME, "comfy_url", None)
            if not comfy_url:
                send_json(self, 404, {"error": {"message": "image routing not configured", "type": "not_found"}})
                return
            rel = path.split("/images/jobs/", 1)[1].strip("/")
            if "/outputs/" in rel:
                parts = rel.split("/outputs/", 1)
                job_id, index_str = parts[0], parts[1]
                try:
                    index = int(index_str)
                    content, mime_type = get_job_output_image(comfy_url, job_id, index)
                except OutputNotFoundError as exc:
                    send_json(self, 404, {"error": {"message": str(exc), "type": "output_not_found"}})
                    return
                except (ImageValidationError, ValueError) as exc:
                    send_json(self, 400, {"error": {"message": str(exc), "type": "invalid_request"}})
                    return
                except (CoordinationError, ComfyBackendError, TimeoutError, ConnectionError, OSError):
                    LOG.exception("failed to get output for job %s index %s", job_id, index_str)
                    send_json(self, 503, {"error": {"message": "Image retrieval from backend failed", "type": "backend_error"}})
                    return
                except Exception:
                    LOG.exception("unexpected error retrieving output for job %s index %s", job_id, index_str)
                    send_json(self, 503, {"error": {"message": "Unexpected error retrieving output", "type": "internal_error"}})
                    return

                try:
                    self.send_response(200)
                    self.send_header("Content-Type", mime_type)
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(content)
                except (ConnectionResetError, BrokenPipeError, OSError):
                    pass
                return
            else:
                job_id = rel
                try:
                    status_info = get_image_job_status(comfy_url, job_id)
                    send_json(self, 200, status_info)
                except (ImageValidationError, ValueError) as exc:
                    send_json(self, 400, {"error": {"message": str(exc), "type": "invalid_request"}})
                except (CoordinationError, ComfyBackendError, TimeoutError, ConnectionError, OSError):
                    LOG.exception("failed to get status for job %s", job_id)
                    send_json(self, 503, {"error": {"message": "Job status check failed on backend", "type": "backend_error"}})
                except Exception:
                    LOG.exception("unexpected error getting status for job %s", job_id)
                    send_json(self, 503, {"error": {"message": "Unexpected error checking job status", "type": "internal_error"}})
                return
        send_json(self, 404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/internal/gpu/yield":
            client_ip = self.client_address[0]
            is_loopback = False
            try:
                is_loopback = ipaddress.ip_address(client_ip).is_loopback
            except ValueError:
                is_loopback = client_ip in {"127.0.0.1", "::1", "localhost"}
            if not is_loopback:
                send_json(self, 403, {"error": {"message": "forbidden: loopback client required", "type": "forbidden"}})
                return

            lease = getattr(RUNTIME, "gpu_lease", None)
            if lease is None or not lease.is_enabled:
                send_json(self, 400, {"error": {"message": "GPU coordination is not enabled", "type": "gpu_coordination_disabled"}})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 4096:
                send_json(self, 400, {"error": {"message": "invalid body length"}})
                return

            try:
                payload = json.loads(self.rfile.read(length))
            except Exception:
                send_json(self, 400, {"error": {"message": "body must be a JSON object"}})
                return

            if (not isinstance(payload, dict) or set(payload.keys()) != {"nonce"}
                    or not isinstance(payload.get("nonce"), str) or not payload["nonce"]):
                send_json(self, 400, {"error": {"message": "yield request must contain strictly {'nonce': '<string>'}", "type": "invalid_request"}})
                return

            if not lease.validate_image_lease(payload["nonce"]):
                send_json(self, 403, {"error": {"message": "invalid or unheld image lease proof", "type": "invalid_lease_proof"}})
                return

            with RUNTIME.request_lock:
                try:
                    # The holder may have exited while this handler waited.
                    if not lease.validate_image_lease(payload["nonce"]):
                        send_json(self, 403, {"error": {"message": "image lease is no longer held"}})
                        return
                    RUNTIME.stop_backend()
                    send_json(self, 200, {"status": "released", "backend_pid": None})
                except Exception as exc:
                    RUNTIME.last_error = str(exc)
                    send_json(self, 503, {"error": {"message": str(exc), "type": "backend_stop_failed"}})
            return

        if path in {"/v1/images/jobs", "/images/jobs"}:
            comfy_url = getattr(RUNTIME, "comfy_url", None)
            image_routes = getattr(RUNTIME, "image_routes", None)
            if not comfy_url or not image_routes:
                send_json(self, 404, {"error": {"message": "image routing not configured", "type": "not_found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 4 * 1024 * 1024:
                send_json(self, 400, {"error": {"message": "invalid body length", "type": "invalid_request"}})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("body must be a JSON object")
            except Exception as exc:
                send_json(self, 400, {"error": {"message": str(exc), "type": "invalid_json"}})
                return

            lease = getattr(RUNTIME, "gpu_lease", None)
            try:
                job_id, status = submit_image_job(comfy_url, lease, image_routes, payload)
                send_json(self, 200, {
                    "id": job_id,
                    "status": status,
                    "model": payload.get("model"),
                    "backend": "comfyui",
                })
            except UncertainSubmissionError as exc:
                LOG.warning("uncertain image job submission: %s", exc)
                send_json(self, 503, {
                    "error": {
                        "message": str(exc),
                        "type": "uncertain_submission",
                        "job_id": exc.job_id,
                    }
                })
            except (ImageValidationError, ValueError) as exc:
                send_json(self, 400, {"error": {"message": str(exc), "type": "invalid_image_request"}})
            except CoordinationError as exc:
                LOG.warning("GPU coordinator check failed: %s", exc)
                send_json(self, 503, {"error": {"message": str(exc), "type": "gpu_coordination_error"}})
            except (ComfyBackendError, TimeoutError, ConnectionError, OSError):
                LOG.warning("ComfyUI backend submission error")
                send_json(self, 503, {"error": {"message": "Image generation backend unavailable", "type": "backend_error"}})
            except Exception:
                LOG.exception("unexpected error submitting image job")
                send_json(self, 503, {"error": {"message": "Unexpected error submitting image job", "type": "internal_error"}})
            return

        if (path.startswith("/v1/images/jobs/") or path.startswith("/images/jobs/")) and path.endswith("/cancel"):
            comfy_url = getattr(RUNTIME, "comfy_url", None)
            if not comfy_url:
                send_json(self, 404, {"error": {"message": "image routing not configured", "type": "not_found"}})
                return
            rel = path.split("/images/jobs/", 1)[1]
            job_id = rel[: -len("/cancel")].strip("/")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 0:
                    self.rfile.read(min(length, 4096))
            except Exception:
                pass
            try:
                res = cancel_image_job(comfy_url, job_id)
                send_json(self, 200, res)
            except (ImageValidationError, ValueError) as exc:
                send_json(self, 400, {"error": {"message": str(exc), "type": "invalid_request"}})
            except (CoordinationError, ComfyBackendError, TimeoutError, ConnectionError, OSError):
                send_json(self, 503, {"error": {"message": "Cancellation failed on backend", "type": "cancel_failed"}})
            except Exception:
                send_json(self, 503, {"error": {"message": "Unexpected error cancelling job", "type": "internal_error"}})
            return

        if path not in ALLOWED_POST | FLEET_POST:
            send_json(self, 404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > (1024*1024 if path in FLEET_POST else 128*1024*1024):
            send_json(self, 400, {"error": {"message": "invalid body length"}})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            requested = payload.get("model") or RUNTIME.config["fleet"]["default_model"]
            if path in FLEET_POST and (not isinstance(payload.get('model'), str) or not payload['model']):
                raise ValueError('explicit_fleet_model_required')
            if not isinstance(requested, str):
                raise ValueError('model_must_be_string')
            model_id, card = resolve_model(RUNTIME.config, requested)
            if backend_kind(card) != 'llama' and (path in FLEET_POST or BINDING_FIELD in payload):
                # Fleet counting/binding read llama.cpp /props, /apply-template and /tokenize.
                raise ValueError(f'fleet_count_unavailable_for_{backend_kind(card)}_backend')
            if backend_kind(card) == 'ninfer' and path == '/v1/chat/completions':
                payload = ninfer_chat_request(payload)
        except (ValueError, AttributeError) as exc:
            send_json(self, 400, {"error": {"message": str(exc) if str(exc) else "body must be a JSON object"}})
            return
        except KeyError:
            send_json(self, 404, {"error": {
                "message": f"unknown or unqualified model: {requested}",
                "available_models": sorted(RUNTIME.config["models"]),
            }})
            return

        binding = payload.pop(BINDING_FIELD, None)
        resident_pid = payload.pop('_tare_resident_backend_pid', None)
        if resident_pid is not None and (path != '/completion' or binding is not None
                or type(resident_pid) is not int or resident_pid <= 0
                or payload.get('n_predict') != 1 or payload.get('stream', False) is not False):
            send_json(self, 400, {'error': {'type': 'invalid_resident_readout', 'message': 'invalid resident-only request'}})
            return
        lease = getattr(RUNTIME, "gpu_lease", None) or SharedGpuLease(None)
        route_timeout = .15 if resident_pid is not None else getattr(RUNTIME, "route_timeout", 60.0)
        request_id = str(secrets.token_hex(8))

        try:
            with lease.hold("text", request_id, timeout=route_timeout):
                with RUNTIME.request_lock:
                    try:
                        if resident_pid is not None and (not lease.is_enabled
                                or RUNTIME.model_id != model_id or RUNTIME.process is None
                                or RUNTIME.process.pid != resident_pid or RUNTIME.process.poll() is not None
                                or not RUNTIME.backend_health()):
                            send_json(self, 409, {'error': {'type': 'resident_backend_unavailable',
                                'message': 'resident backend changed; no model was loaded'}})
                            return
                        operations = []
                        if path == '/v1/fleet/count':
                            if set(payload) != {'model', 'request'} or binding is not None:
                                raise ValueError('fleet_count_envelope_invalid')
                            value = count_request(RUNTIME, requested, payload['request'], operations)
                            send_json(self, 200, value)
                            return
                        if path == '/v1/fleet/profile':
                            if set(payload) != {'model'} or binding is not None:
                                raise ValueError('fleet_profile_envelope_invalid')
                            RUNTIME.ensure_model(requested)
                            send_json(self, 200, {'profile': effective_profile(RUNTIME, model_id, operations),
                                                  'backend_operations': operations})
                            return
                        if binding is not None:
                            if path != '/v1/chat/completions' or type(payload.get('stream',False)) is not bool:
                                raise ValueError('fleet_binding_requires_chat_and_boolean_stream')
                            profile = check_binding(RUNTIME, requested, payload, binding, operations)
                            observed = {'binding': binding, 'profile': profile, 'backend_operations': operations,
                                'backend_operation_coverage': 'binding_checks_only_excludes_health_loading_and_generation'}
                        elif resident_pid is None:
                            RUNTIME.ensure_model(str(requested))
                            observed = None
                        else:
                            observed = None
                        payload['model'] = model_id
                        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                        try:
                            if resident_pid is not None:
                                proxy_request(self, body, timeout=15)
                            else:
                                proxy_request(self, body, fleet_observation=observed)
                            RUNTIME.last_used = time.monotonic()
                        except (ConnectionError, http.client.HTTPException, socket.error, OSError) as exc:
                            if lease.is_enabled:
                                LOG.warning("transport error during proxy; stopping backend under GPU lease: %s", exc)
                                RUNTIME.stop_backend()
                            raise
                    except BindingMismatch as exc:
                        send_json(self, 409, {'error': {'message': str(exc), 'type': 'fleet_binding_mismatch'},
                                             'backend_operations': operations})
                    except ValueError as exc:
                        send_json(self, 400, {'error': {'message': str(exc), 'type': 'invalid_fleet_request'},
                                             'backend_operations': operations})
                    except Exception as exc:
                        RUNTIME.last_error = str(exc)
                        LOG.exception("request failed for model=%s", requested)
                        send_json(self, 503, {"error": {
                            "message": str(exc),
                            "type": "qualified_model_gateway_error",
                            "model": requested,
                        }})
        except TimeoutError as exc:
            LOG.warning("GPU lease timeout for model=%s: %s", requested, exc)
            send_json(self, 503, {
                "error": {
                    "message": f"timed out waiting for GPU lease ({route_timeout:.0f}s)",
                    "type": "gpu_lease_timeout",
                    "model": requested,
                }
            })
        except InterruptedError as exc:
            LOG.warning("GPU lease cancelled for model=%s: %s", requested, exc)
            send_json(self, 503, {
                "error": {
                    "message": "GPU lease acquisition cancelled",
                    "type": "gpu_lease_cancelled",
                    "model": requested,
                }
            })


def port_is_free(host: str, port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex((host, port)) != 0


def port_is_bindable(host: str, port: int) -> bool:
    """Bind like llama-server does (no SO_REUSEADDR): server-side TIME_WAIT blocks it too."""
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def wait_until_bindable(host: str, port: int, timeout: float = 90.0) -> None:
    # NInfer closes connections itself, leaving the port in TIME_WAIT for ~60 s after it exits.
    deadline = time.monotonic() + timeout
    while not port_is_bindable(host, port):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"backend port {port} still unavailable after {timeout:.0f}s")
        time.sleep(0.5)


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
    parser.add_argument("--gpu-lock", default=None, help="path to cross-process shared GPU lock file")
    parser.add_argument("--route-timeout", type=float, default=60.0, help="seconds to wait for GPU lease")
    parser.add_argument("--comfy-url", default=None, help="loopback HTTP URL for managed ComfyUI instance")
    parser.add_argument("--image-routes", default=None, help="path to image routes JSON configuration")
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
        gpu_lock=args.gpu_lock,
        route_timeout=args.route_timeout,
        comfy_url=args.comfy_url,
        image_routes_path=args.image_routes,
    )
    if args.preload:
        with RUNTIME.gpu_lease.hold("text", "preload", timeout=args.load_timeout):
            with RUNTIME.request_lock:
                RUNTIME.ensure_model(args.preload)

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    def idle_watch() -> None:
        while True:
            time.sleep(30)
            with contextlib.suppress(Exception):
                RUNTIME.unload_if_idle()

    threading.Thread(target=idle_watch, name="idle-unload", daemon=True).start()

    def shutdown(signum: int, _frame: Any) -> None:
        LOG.info("received signal %s", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    LOG.info("gateway listening on %s:%s; backend=%s:%s; gpu_lock=%s; comfy_url=%s", args.host, args.port,
             args.backend_host, args.backend_port, args.gpu_lock, args.comfy_url)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        with RUNTIME.request_lock:
            RUNTIME.stop_backend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
