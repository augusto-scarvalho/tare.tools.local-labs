"""llama.cpp server adapter — start, health, stop.

Drives `llama-server` inside WSL from the Windows side. `llama-bench` is NOT used
here: it cannot measure load time, TTFT, request latency, pass rate or swap time,
which is most of what the platform needs. It stays a cheap pre-filter elsewhere.
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


DEFAULT_SERVER_BIN = os.environ.get(
    "SLOP_CPP_SERVER_BIN",
    "/home/augus/src/slop.cpp/build/bin/llama-server",
)


@dataclass(frozen=True)
class ServerProfile:
    model_path: str                  # path INSIDE the distro
    port: int = 8080
    n_cpu_moe: int | None = None     # the primary tuning axis on a MoE model
    ctx_size: int | None = None
    batch: int | None = None
    ubatch: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    n_gpu_layers: int | None = None
    flash_attn: str | None = None
    no_mmap: bool = False            # owner's benchmark: ~2x prefill, real RAM cost
    # Escape hatch for flags this dataclass does not model. Lives on the PROFILE,
    # not on start(), because it is part of what defines a configuration -- two
    # runs with different extra_args are different configs and must not share an id.
    extra_args: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    profile: ServerProfile
    load_seconds: float | None = None
    stderr_tail: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.profile.port}"


class LlamaCppAdapter:
    def __init__(self, *, distro: str = "Ubuntu-24.04",
                 server_bin: str = DEFAULT_SERVER_BIN,
                 env: dict[str, str] | None = None):
        # NOT the default distro: the default is `Ubuntu` and it is Stopped. A bare
        # `wsl` command lands in the wrong place and reports nothing useful.
        self.distro = distro
        self.server_bin = server_bin
        # Environment for the SERVER, inside the distro. Needed because some
        # llama.cpp features are opt-in by env var and invisible on the command
        # line -- GGML_SCHED_PREFETCH_EXPERTS is the case that forced this: a
        # whole A/B compared a fork against upstream with the fork's headline
        # feature switched off, because nothing on the CLI could reveal it.
        self.env = dict(env or {})

    def argv(self, p: ServerProfile) -> list[str]:
        # --host 0.0.0.0 is mandatory: without it the server only accepts connections
        # from inside WSL and nothing outside the distro can reach it.
        a = ["wsl", "-d", self.distro, "--"]
        if self.env:
            # `env` inside the distro: WSL does not forward the Windows environment, so
            # exporting on this side would silently do nothing.
            a += ["env"] + [f"{k}={v}" for k, v in sorted(self.env.items())]
        a += [self.server_bin,
              "--host", "0.0.0.0", "--port", str(p.port), "-m", p.model_path]
        if p.n_cpu_moe is not None:
            a += ["--n-cpu-moe", str(p.n_cpu_moe)]
        if p.ctx_size is not None:
            a += ["-c", str(p.ctx_size)]
        if p.batch is not None:
            a += ["-b", str(p.batch)]
        if p.ubatch is not None:
            a += ["-ub", str(p.ubatch)]
        if p.cache_type_k:
            a += ["-ctk", p.cache_type_k]
        if p.cache_type_v:
            a += ["-ctv", p.cache_type_v]
        if p.n_gpu_layers is not None:
            a += ["-ngl", str(p.n_gpu_layers)]
        if p.flash_attn:
            a += ["-fa", p.flash_attn]
        if p.no_mmap:
            a += ["--no-mmap"]
        a += list(p.extra_args)
        return a + list(p.extra)

    def start(self, p: ServerProfile) -> ServerHandle:
        proc = subprocess.Popen(self.argv(p), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        return ServerHandle(proc=proc, profile=p)

    def wait_until_healthy(self, h: ServerHandle, timeout_s: float = 600.0) -> bool:
        """Poll /health until the server answers. Load time is measured to the first
        healthy reply, not to process spawn: a spawned process that is still mapping
        25 GB is not a server."""
        t0 = time.monotonic()
        url = f"{h.base_url}/health"
        while time.monotonic() - t0 < timeout_s:
            if h.proc.poll() is not None:      # died while loading
                h.stderr_tail = self._drain(h)
                return False
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    if r.status == 200:
                        h.load_seconds = time.monotonic() - t0
                        return True
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(1.0)
        h.stderr_tail = self._drain(h)
        return False

    def _drain(self, h: ServerHandle, limit: int = 20) -> list[str]:
        try:
            out = h.proc.stdout.read() if h.proc.stdout else ""
        except Exception:
            return []
        return (out or "").strip().splitlines()[-limit:]

    def stop(self, h: ServerHandle, grace_s: float = 15.0) -> None:
        """Terminate, then verify. A dead parent does NOT kill its children: `wsl.exe`
        is a thin front for a process living in the distro, so the tree must be taken
        down explicitly or the server keeps serving after the run 'ended'. Two servers
        survived three days elsewhere for exactly this reason."""
        if h.proc.poll() is None:
            h.proc.terminate()
            try:
                h.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                h.proc.kill()
        self.force_stop(h)

    def force_stop(self, h: ServerHandle) -> None:
        """Belt and braces: kill anything still holding the model inside the distro."""
        try:
            subprocess.run(["wsl", "-d", self.distro, "--", "pkill", "-9", "-f",
                            "llama-server"], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass

    def is_port_free(self, port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return False
        except Exception:
            return True
