"""lmctl — the one ops front-door for this box. Stop re-deriving WSL incantations.

Every session was re-typing `wsl.exe -d Ubuntu-24.04 -- bash -lc '...'`, killing servers
by port (`fuser -k 8090/tcp`, NOT `pkill -f ...8090` which self-matches its own cmdline),
copying judge serve scripts out of scratch/, and hand-rolling `nohup ... & disown`. All of
that now lives in ONE place, with the gotchas baked in as code instead of as memory notes.

Runs as WINDOWS Python (python.exe), on purpose: from here `wsl.exe` and `nvidia-smi.exe`
resolve cleanly and there is NO MSYS path-mangling (that footgun is the git-bash layer, not
Windows Python) and no empty-`$VAR` expansion (this file controls the bash string). Invoke
it from PowerShell or the Bash tool the same way:

    python lmctl.py serve mistral-judge          # bring a judge up, detached, health-checked
    python lmctl.py serve gemma-judge
    python lmctl.py ps                            # what's serving right now
    python lmctl.py stop --port 8090
    python lmctl.py gpu                           # VRAM / clocks / power / temp / util
    python lmctl.py sensors                       # FanControl readout (CPU/GPU/fans/Kraken)
    python lmctl.py build llama-bench             # cmake --build the target, in the right distro
    python lmctl.py wsl -- nvidia-smi             # escape hatch: run anything in Ubuntu-24.04
    python lmctl.py serve --list                  # list known profiles

`serve` takes a PROFILE name (serve_profiles.py) or a bare MODELS registry key. Detached by
default — the server outlives this process (a dead parent does NOT kill a WSL child), which
is exactly what you want when driving it from an agent. `--foreground` blocks instead.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.models import MODELS                       # noqa: E402
from model_lifecycle.serve_profiles import (                    # noqa: E402
    DEFAULT_BIN, SERVE_PROFILES, ServeSpec)

DISTRO = "Ubuntu-24.04"          # NOT the default `Ubuntu` (empty home, wrong user)
FCREAD = r"C:\CrashWatch\fcread\fcread.exe"
SRC_DIR = "/home/augus/src/llama.cpp-master"   # the deploy fork; build targets live here
KNOWN_PORTS = (8080, 8090, 8091)               # deploy + the two judges


# --------------------------------------------------------------------------- shell helpers
# HEADLESS, always. wsl.exe / nvidia-smi.exe / fcread.exe are console apps; launched without
# this flag they pop (or flash) a console window on the desktop -- which the owner sees and
# reasonably wants to close. CREATE_NO_WINDOW runs them with no window at all; output still
# flows through inherited/piped handles, so nothing is lost. (0x08000000; getattr keeps this
# importable off-Windows even though lmctl only ever runs on Windows.)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def _wsl(script: str, *, capture: bool = True, timeout: float | None = None,
         distro: str = DISTRO) -> subprocess.CompletedProcess:
    """Run a bash script inside the RIGHT distro. One code path for every WSL call so the
    distro pin and login shell (`-lc`) are never forgotten -- and so no window ever shows."""
    argv = ["wsl.exe", "-d", distro, "--", "bash", "-lc", script]
    return subprocess.run(argv, capture_output=capture, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          creationflags=_NO_WINDOW)


def _host_exe(argv: list[str], *, timeout: float = 15) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          creationflags=_NO_WINDOW)


def _health(port: int, timeout: float = 3) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _wait_healthy(port: int, timeout_s: float) -> float | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if _health(port):
            return time.monotonic() - t0
        time.sleep(1.0)
    return None


# --------------------------------------------------------------------------- serve resolve
def _resolve(name: str, port: int | None, extra: list[str]) -> tuple[ServeSpec, int, list[str]]:
    """A serve target is a profile name or a bare MODELS key. Build a ServeSpec for either.
    Returns (spec, port, argv-flags). CLI overrides in `extra` are appended AFTER the
    profile flags so llama.cpp's last-wins parsing lets you override any scalar."""
    if name in SERVE_PROFILES:
        spec = SERVE_PROFILES[name]
    elif name in MODELS:
        # A registry key with no named profile: serve with minimal sane defaults and let
        # --port / -- extra flags carry the rest. Dense vs MoE is the caller's to say.
        spec = ServeSpec(name=name, model_path=MODELS[name].path, port=port or 8080,
                         flags=("-fa", "on", "--ctx-size", "8192", "--jinja"), bin=DEFAULT_BIN,
                         note=f"bare MODELS key {name} ({MODELS[name].quant}); no named profile")
    else:
        raise SystemExit(
            f"unknown serve target '{name}'.\n"
            f"  profiles: {', '.join(sorted(SERVE_PROFILES))}\n"
            f"  or a MODELS key: {', '.join(sorted(MODELS))}")
    use_port = port or spec.port
    return spec, use_port, list(spec.flags) + extra


def _serve_bash(spec: ServeSpec, port: int, flags: list[str]) -> str:
    argv = [spec.bin, "-m", spec.model_path, "--host", "0.0.0.0", "--port", str(port), *flags]
    env_tokens = [f"{k}={v}" for k, v in sorted(spec.env.items())]
    full = (["env", *env_tokens] if env_tokens else []) + argv
    return shlex.join(full)


# Detaching on THIS box: a WSL2 distro is torn down when its last `wsl.exe --` session
# ends, so a Linux-side `nohup`/`setsid &` dies with it (verified: setsid printed a PID
# then the process was gone). The server's lifetime is instead the lifetime of a
# WINDOWS-side `wsl.exe` process holding the session open. To outlive lmctl AND stay
# headless it is started with CREATE_NO_WINDOW (no console window ever) + a new process
# group + a job breakaway so a parent shell/job that dies (or is a kill-on-close job) does
# not take it down. NOT DETACHED_PROCESS: that flag still lets wsl.exe pop its own console
# window. `stop --port` kills the server inside the distro; the holder then exits on its own.
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
_BREAKAWAY = 0x01000000  # CREATE_BREAKAWAY_FROM_JOB


def _serve_detached(bash_script: str, distro: str) -> None:
    argv = ["wsl.exe", "-d", distro, "--", "bash", "-lc", bash_script]
    kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL, close_fds=True)
    try:
        # Breakaway needs the enclosing job to allow it; if it doesn't, CreateProcess
        # raises -- fall back to no-window + new-group, which still survives here.
        subprocess.Popen(argv, creationflags=_NO_WINDOW | _NEW_GROUP | _BREAKAWAY, **kw)
    except OSError:
        subprocess.Popen(argv, creationflags=_NO_WINDOW | _NEW_GROUP, **kw)


# --------------------------------------------------------------------------------- verbs
def cmd_serve(a) -> int:
    if a.list or not a.name:
        print("serve profiles:")
        for n, s in sorted(SERVE_PROFILES.items()):
            print(f"  {n:15s} :{s.port}  {s.note.splitlines()[0] if s.note else ''}")
        print("\n(or serve any MODELS key: " + ", ".join(sorted(MODELS)) + ")")
        return 0

    spec, port, flags = _resolve(a.name, a.port, a.extra)
    cmd_str = _serve_bash(spec, port, flags)
    if spec.note:
        print(f"# {spec.note}")
    print(f"# argv: {cmd_str}\n")

    if _health(port):  # port already answering
        print(f"port {port} is already serving -- `lmctl stop --port {port}` first")
        return 2

    free_port = f"fuser -k {port}/tcp 2>/dev/null || true; sleep 1; "
    if a.foreground:
        print(f"serving {a.name} on :{port} (foreground; Ctrl-C to stop)\n", flush=True)
        return _wsl(free_port + f"exec {cmd_str}", capture=False).returncode

    log = f"/tmp/lmctl-serve-{port}.log"
    _serve_detached(free_port + f"exec {cmd_str} > {log} 2>&1", a.distro)
    print(f"launched (detached), waiting for /health (log: {log}) ...", flush=True)
    took = _wait_healthy(port, a.timeout)
    if took is None:
        print(f"SERVER NEVER HEALTHY in {a.timeout:.0f}s. last log lines:")
        for ln in _wsl(f"tail -n 20 {log}").stdout.splitlines():
            print(f"  | {ln}")
        return 1
    host = socket.gethostname().lower()
    print(f"UP in {took:.1f}s")
    print(f"  local    : http://127.0.0.1:{port}  (OpenAI: /v1)")
    print(f"  by-name  : http://{host}.local:{port}")
    print(f"  stop     : python lmctl.py stop --port {port}")
    return 0


def cmd_stop(a) -> int:
    if a.all:
        _wsl("pkill -9 -f llama-server 2>/dev/null || true")
        print("killed all llama-server processes")
        return 0
    if a.port is None:
        print("give --port N or --all")
        return 2
    _wsl(f"fuser -k {a.port}/tcp 2>/dev/null || true")
    print(f"freed port {a.port}")
    return 0


def _flag_value(tokens: list[str], names: tuple[str, ...]) -> str:
    """The token following the first occurrence of any flag in `names`, else '?'."""
    for i, tok in enumerate(tokens[:-1]):
        if tok in names:
            return tokens[i + 1]
    return "?"


def cmd_ps(a) -> int:
    r = _wsl("pgrep -a llama-server 2>/dev/null || true")
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        print("no llama-server running")
    else:
        print(f"{len(lines)} llama-server process(es):")
        for ln in lines:
            pid, _, cmd = ln.partition(" ")
            toks = cmd.split()
            port = _flag_value(toks, ("--port", "-p"))
            model = os.path.basename(_flag_value(toks, ("-m", "--model")))
            print(f"  pid {pid:>7}  :{port:<6} {model}")
    print("\nknown ports:")
    for p in KNOWN_PORTS:
        print(f"  :{p}  {'UP' if _health(p) else '--'}")
    return 0


def cmd_gpu(a) -> int:
    fields = ("name,memory.total,memory.used,memory.free,utilization.gpu,"
              "temperature.gpu,power.draw,clocks.sm,clocks.mem")
    r = _host_exe(["nvidia-smi.exe", f"--query-gpu={fields}",
                   "--format=csv,noheader,nounits"])
    if r.returncode != 0:
        print(f"nvidia-smi failed: {r.stderr.strip()[:200]}")
        return 1
    name, tot, used, free, util, temp, pwr, sm, mem = (
        x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
    print(f"{name}")
    print(f"  VRAM   {used}/{tot} MiB used  ({free} MiB free)")
    print(f"  util   {util}%   temp {temp} C   power {pwr} W")
    print(f"  clocks sm {sm} MHz   mem {mem} MHz")
    print("  (undervolt is MSI Afterburner-managed: ~1860 MHz @ 850 mV, applied at startup)")
    return 0


def cmd_sensors(a) -> int:
    if not os.path.exists(FCREAD):
        print(f"fcread not found at {FCREAD} (see memory: fancontrol-sensor-readout)")
        return 1
    argv = [FCREAD] + (["--json"] if a.json else [])
    r = _host_exe(argv, timeout=10)
    if r.returncode != 0:
        print(f"fcread failed: {r.stderr.strip()[:200]}")
        return 1
    print(r.stdout.strip())
    return 0


def cmd_build(a) -> int:
    cmd = f"cd {a.dir} && cmake --build build --target {a.target} -j"
    print(f"# {DISTRO}: {cmd}\n", flush=True)
    return _wsl(cmd, capture=False, timeout=a.timeout).returncode


def cmd_wsl(a) -> int:
    if not a.cmd:
        print("give a command: lmctl wsl -- <cmd ...>")
        return 2
    script = " ".join(a.cmd)
    return _wsl(script, capture=False).returncode


# ---------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lmctl", description=__doc__.splitlines()[0])
    p.add_argument("--distro", default=DISTRO, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)

    sv = sub.add_parser("serve", help="bring a model/judge up (detached, health-checked)")
    sv.add_argument("name", nargs="?", help="profile name or MODELS key")
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--foreground", action="store_true", help="block instead of detaching")
    sv.add_argument("--list", action="store_true", help="list known serve profiles")
    sv.add_argument("--timeout", type=float, default=1800, help="health-wait seconds")
    sv.add_argument("extra", nargs="*", help="extra llama-server flags after --")
    sv.set_defaults(func=cmd_serve)

    st = sub.add_parser("stop", help="stop a server by port, or --all")
    st.add_argument("--port", type=int, default=None)
    st.add_argument("--all", action="store_true")
    st.set_defaults(func=cmd_stop)

    sub.add_parser("ps", help="list running llama-server + known-port health"
                   ).set_defaults(func=cmd_ps)
    sub.add_parser("gpu", help="GPU VRAM/clocks/power/temp/util").set_defaults(func=cmd_gpu)

    se = sub.add_parser("sensors", help="FanControl sensor readout (fcread)")
    se.add_argument("--json", action="store_true", help="full 70-sensor JSON dump")
    se.set_defaults(func=cmd_sensors)

    bd = sub.add_parser("build", help="cmake --build a target in the deploy fork")
    bd.add_argument("target", help="e.g. llama-bench, llama-cli, test-backend-ops")
    bd.add_argument("--dir", default=SRC_DIR, help="llama.cpp source dir in the distro")
    bd.add_argument("--timeout", type=float, default=1800)
    bd.set_defaults(func=cmd_build)

    wc = sub.add_parser("wsl", help="run an arbitrary command in the right distro")
    wc.add_argument("cmd", nargs=argparse.REMAINDER)
    wc.set_defaults(func=cmd_wsl)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
