"""First real measurement — validates Phase 4 end to end.

One configuration, chosen to be the most likely to SUCCEED rather than the most
interesting: Q4_K_M (the smallest quant above the owner's floor) with n-cpu-moe=8
(the sweet spot the prior benchmark already established) and a q8_0 KV cache. The
point of this run is to prove the instrument, not to discover anything.

    python run_one.py

Writes runs/<config_id>.json. A REJECTED verdict is a successful run of the
instrument -- it means the guard did its job.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.control_plane.guard import Envelope          # noqa: E402
from model_lifecycle.servers.llama_cpp import LlamaCppAdapter, ServerProfile  # noqa: E402
from model_lifecycle.workloads.throughput import run_config       # noqa: E402

MODEL_DIR = "/home/augus/models/qwen36-35b-a3b"

PROMPT = (
    "Explain, in about 120 words, why memory bandwidth rather than raw compute "
    "usually limits token generation speed for a large language model running on a "
    "single consumer GPU. Be concrete."
)


def main() -> int:
    profile = ServerProfile(
        model_path=f"{MODEL_DIR}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        port=8080,
        n_cpu_moe=8,          # prior benchmark: 8-10 is the sweet spot on this box
        ctx_size=8192,        # modest on purpose: KV is pure VRAM, and this run is
                              # about proving the harness, not stressing the envelope
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        no_mmap=False,
    )
    config_id = "qwen36-q4km__ncmoe8__kvq8__ctx8192"

    adapter = LlamaCppAdapter()
    if not adapter.is_port_free(profile.port):
        print(f"port {profile.port} already serving - stop the existing server first")
        return 2

    print(f"running {config_id} ...", flush=True)
    t0 = time.monotonic()
    # 1500, not 300. First run: three empty responses because the model spent the
    # whole budget reasoning before writing a word. On a thinking model the budget has
    # to cover reasoning AND the answer, and the floor is prompt-dependent -- which is
    # exactly why the platform must record a per-model minimum-token floor rather than
    # carry one global default.
    result = run_config(adapter, profile, config_id=config_id, prompt=PROMPT,
                        repetitions=3, max_tokens=1500, envelope=Envelope())
    elapsed = time.monotonic() - t0

    out_dir = pathlib.Path(__file__).parent / "runs"
    out_dir.mkdir(exist_ok=True)
    payload = result.as_dict() | {"wall_seconds": round(elapsed, 1)}
    (out_dir / f"{config_id}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"verdict     : {result.verdict}  {result.reason or ''}")
    print(f"load        : {result.load_seconds}s")
    print(f"pass rate   : {result.pass_rate:.0%} ({result.requests} requested)")
    if result.ttft:
        print(f"TTFT        : mean {result.ttft['mean']:.2f}s  p95 {result.ttft['p95']:.2f}s")
    if result.gen_tps:
        print(f"generation  : mean {result.gen_tps['mean']:.1f} t/s  cv {result.gen_tps['cv']:.2f}")
    if result.prompt_tps:
        print(f"prompt      : mean {result.prompt_tps['mean']:.1f} t/s")
    print(f"min free    : vram {result.min_free_vram_mb}MB  ram {result.min_available_ram_mb}MB")
    if result.failures:
        print(f"failures    : {result.failures[:3]}")
    print(f"wall        : {elapsed:.0f}s -> runs/{config_id}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
