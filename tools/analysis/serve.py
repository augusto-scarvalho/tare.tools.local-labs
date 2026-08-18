"""Bring one configuration up and LEAVE IT UP, for the notebook to consume.

Every other entry point in this project has the benchmark lifecycle: start, measure,
tear down. `run_one.py`, `quality_bench.py` and `ab_isolate.py` all end by killing the
server, because a configuration that outlives its measurement contaminates the next one.
Serving is the opposite lifecycle and needed its own file rather than a flag on those.

Networking needs no portproxy: `.wslconfig` sets `networkingMode=mirrored` and there is
no forward configured on this host. What is NOT established is that the notebook can
reach this server -- see `config/environment.yaml`, `serving.reachability`. Measured so
far: loopback answers 200; the desktop cannot test its own LAN address (mirrored-mode
behaviour without `hostAddressLoopback`); a correct Hyper-V firewall allow-rule for TCP
8080 from LocalSubnet exists, behind a deny-by-default inbound policy.

If the notebook cannot connect, check the IPv4 literal BEFORE the firewall: `--host
0.0.0.0` binds IPv4 only, while mDNS answers with AAAA records ahead of the A record.

    python serve.py                          # proven default config
    python serve.py --model qwen36-35b-q4 --ctx 32768

Ctrl-C tears down. That teardown is not decorative: `wsl.exe` is a thin front for a
process living in the distro, so a dead parent does NOT kill the child. Two servers
survived three days elsewhere for exactly this reason.
"""
from __future__ import annotations

import argparse
import pathlib
import socket
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.collectors.host import HostReadError, sample     # noqa: E402
from model_lifecycle.control_plane.guard import (                     # noqa: E402
    Envelope, precheck, start_watch)
from model_lifecycle.models import MODELS                             # noqa: E402
from model_lifecycle.servers.llama_cpp import (                       # noqa: E402
    LlamaCppAdapter, ServerProfile)

LOCAL_BIN = "/home/augus/src/llama.cpp-local/build/bin/llama-server"

# MODELS is now the shared registry (model_lifecycle.models). Was a copied dict here.


def urls(port: int) -> tuple[str, str]:
    """(loopback, by-name). The second is what the notebook gets.

    By NAME and never by address: DHCP has already moved this host once (.107 -> .66)
    and the interface is WiFi with a DHCP-assigned suffix. mDNS re-resolves per query,
    so the name survives a move that would strand a written-down IP. Requires a resolver
    on the CLIENT -- native on Windows 11, macOS and avahi-equipped Linux.
    """
    return f"http://127.0.0.1:{port}", f"http://{socket.gethostname().lower()}.local:{port}"


def serve(model_key: str, *, port: int, ncmoe: int, ctx: int, kv: str,
          ubatch: int | None, flash: str | None, prefetch: bool,
          poll_s: float) -> int:
    # Pinning ON by default. Measured on this exact model in runs/ab-genpin-qwen36-35b:
    # prefill 214.5 -> 462.3 t/s over 6 paired rounds, generation unchanged (-0.08 t/s).
    #
    # Prefetch OFF by default, deliberately, despite being the fork's headline feature:
    # the evidence is contradictory. It measured -21.9% (n=6, sign p=0.031) when added to
    # an ALREADY PINNED baseline, and the L18 screen put it OFF fastest and non-monotonic.
    # A serving default is not the place to carry a disputed switch.
    env_vars = {"GGML_CUDA_REGISTER_HOST": "1"}
    if prefetch:
        env_vars["GGML_SCHED_PREFETCH_EXPERTS"] = "3"

    envelope = Envelope()
    # Refuse to START inside the reserve. Cheap, and it stops a busy desktop being
    # ambushed by a 21 GB read it cannot afford.
    if reason := precheck(envelope):
        print(f"REFUSING TO START: {reason}")
        return 2

    adapter = LlamaCppAdapter(server_bin=LOCAL_BIN, env=env_vars)
    if not adapter.is_port_free(port):
        print(f"port {port} is already serving -- stop that server first")
        return 2

    profile = ServerProfile(model_path=MODELS[model_key].path, port=port, n_cpu_moe=ncmoe,
                            ctx_size=ctx, ubatch=ubatch, cache_type_k=kv, cache_type_v=kv,
                            flash_attn=flash, extra_args=("--jinja",))

    print(f"starting {model_key}  ncmoe={ncmoe} ctx={ctx} kv={kv} "
          f"pinning=on prefetch={'on' if prefetch else 'off'}", flush=True)
    h = adapter.start(profile)
    try:
        if not adapter.wait_until_healthy(h, timeout_s=1800):
            print("SERVER NEVER HEALTHY. argv:")
            print("  " + " ".join(adapter.argv(profile)))
            for ln in h.stderr_tail[-15:]:
                print(f"  | {ln}")
            return 1

        loopback, by_name = urls(port)
        print(f"\nUP in {h.load_seconds:.1f}s")
        print(f"  this desktop : {loopback}")
        print(f"  the notebook : {by_name}")
        print(f"  OpenAI-compatible endpoint at {by_name}/v1")
        print("\nCtrl-C to stop.\n", flush=True)

        # The guard WARNS here; it does not kill. This is a deliberate departure from the
        # benchmark path, where a breach INVALIDATES the measurement and rejecting is the
        # whole point. While serving, the owner is deliberately using the machine, and a
        # tool that kills their loaded model because a browser opened would itself be the
        # thing taking the desktop down. Refusing to start stays hard; refusing to
        # continue does not.
        watch = start_watch(envelope)
        watch.mark_healthy()
        warned = False
        while h.proc.poll() is None:
            time.sleep(poll_s)
            try:
                ok = watch.observe(sample())
            except HostReadError as exc:
                # A reading that cannot be taken is reported, never silently treated as
                # healthy -- the mirror of host.py's rule against reading it as zero.
                print(f"  [guard] could not read host: {exc}", flush=True)
                continue
            if not ok and not warned:
                print(f"  [guard] WARNING sustained: {watch.breach}", flush=True)
                print("  [guard] still serving on purpose -- stop it yourself if the "
                      "desktop needs the room.", flush=True)
                warned = True

        print(f"server exited on its own (rc={h.proc.poll()})")
        for ln in adapter._drain(h)[-15:]:
            print(f"  | {ln}")
        return 1
    except KeyboardInterrupt:
        print("\nstopping ...")
        return 0
    finally:
        adapter.stop(h)
        adapter.force_stop(h)
        print("stopped.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen36-35b-q4")
    ap.add_argument("--port", type=int, default=8080)
    # The default is run_one.py's PROVEN configuration, not the fastest one found. A
    # serving default should be the one already observed to come up cleanly inside the
    # envelope; override it once you know what you want.
    ap.add_argument("--ncmoe", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--kv", default="q8_0")
    ap.add_argument("--ubatch", type=int, default=None)
    ap.add_argument("--flash", default=None)
    ap.add_argument("--prefetch", action="store_true",
                    help="enable GGML_SCHED_PREFETCH_EXPERTS=3 (disputed; see serve())")
    ap.add_argument("--poll-s", type=float, default=10.0)
    a = ap.parse_args()
    return serve(a.model, port=a.port, ncmoe=a.ncmoe, ctx=a.ctx, kv=a.kv,
                 ubatch=a.ubatch, flash=a.flash, prefetch=a.prefetch, poll_s=a.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
