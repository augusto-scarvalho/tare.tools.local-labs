#!/usr/bin/env python3
"""Clean successor for the RSH-02 Triton namespace correction."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.research import run_rsh02_packed_gpu as base

TASK_ID = "BACKLOG-RSH02-PACKED-GPU-02"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    base.TASK_ID = TASK_ID
    base.EXPECTED = {
        path: digest for path, digest in base.EXPECTED.items()
        if "BACKLOG-RSH02-PACKED-GPU-01" not in path.as_posix()
    }
    base.EXPECTED.update({
        ROOT / "config/research_backlog_admissions/BACKLOG-RSH02-PACKED-GPU-02.json": "f00006fedbf20fc6d090366894322ee7b81ad91db02f91e79771cfec5fa688f5",
        ROOT / "runs/research/BACKLOG-RSH02-PACKED-GPU-02/PRE_REGISTRATION.md": "147c4ff36d097fec785f47ed8b3e822f7595e6857203448a1e529a0a2977a60b",
        ROOT / "runs/research/BACKLOG-RSH02-PACKED-GPU-01/ABORTED.md": "f6f59c8e18b219ecd6a28d14a0849de026627a509896ec486be0ac60545cacca",
    })
    outdir = args.outdir.resolve()
    receipt = base.run(outdir)
    metrics = json.loads((outdir / "raw/actual_scores.json").read_text(encoding="utf-8"))
    all_pass = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "RSH02_FALSE_NEGATIVE_CONFIRMED_R2" if all_pass else "RSH02_NEGATIVE_RETAINED_R2"
    failed = [name for name, gate in receipt["gates"].items() if not gate["pass"]]
    (outdir / "RESULT.md").write_text(f"""# {TASK_ID} result

## Verdict

`{claim}` pending independent AGY review.

The successor encoded `{metrics['actual_model_weight_elements']}` real Qwen weight symbols into a physical Huffman bitstream and decoded them exactly on the RTX 3090. Physical storage was `{metrics['physical_bits_per_element']:.4f}` bits/element including restart offsets and lookup table. Huffman latency was `{metrics['huffman_latency_ms']:.4f}` ms and input throughput `{metrics['decoder_input_throughput_gbs']:.3f}` GB/s, versus `{metrics['int4_latency_ms']:.4f}` ms for physical INT4 (`{metrics['latency_penalty_vs_int4']:.3f}x`).

Failed gates: `{', '.join(failed) if failed else 'none'}`. Claim scope is limited to this physical codec-kernel screen.
""", encoding="utf-8")
    print(json.dumps(receipt["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
