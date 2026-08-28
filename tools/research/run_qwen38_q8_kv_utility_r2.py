#!/usr/bin/env python3
"""Clean-provenance physical repeat of Qwen3.8 Q8_0 KV utility."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research import run_qwen38_q8_kv_utility as r1


TASK_ID = "BACKLOG-QWEN38-Q8-KV-UTILITY-02"
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-QWEN38-Q8-KV-UTILITY-02.json": "9f833f08530d4060caab73e706c3fb06cc80a74c0cdffcc6b325ba1f5a23ecba",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/PRE_REGISTRATION.md": "7a743739bf5329a150bea4088c48788f9adc488f1d79c05bc5f7f651831879ad",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-01/raw/receipt.json": "3a7e0eaabc678e6514e13875906005d89789adccca1edd44b6b880e7d72dbb24",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-01/raw/samples.jsonl": "f072f9e297a5ac1f1681a4c849c163ee99956536c26a057fa1feaa40dc5036ef",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-01/raw/actual_scores.json": "9aed2ce35d30c5e9b35b4f86848eb2d3a7d6e47bfebc3a9063e12563a25e7a4f",
    "runs/research/BACKLOG-QWEN38-KV-PRECISION-02/raw/receipt.json": "aba1cc2685f5b74ab01a16937b13d7c063898e9cf2df6ac97bb30564a4074bd7",
    "runs/research/BACKLOG-QWEN38-KV-PRECISION-02/raw/samples.jsonl": "24151e0077e34172f25eafdba5ea24377cb4293ae225bca1b90b85edd10be10a",
    "runs/research/BACKLOG-QWEN38-KV-PRECISION-02/raw/actual_scores.json": "17369a2faaa67899d249772dc96c03f6ef2fb94b0c37b9eea990bcbc29b50b37",
    "tools/research/run_qwen38_q8_kv_utility.py": "e1e0c30a201dc87b0d924067d59b5fe17fa6b81049c696cd1ad8e723e2de58d6",
    "tools/research/run_qwen38_kv_precision.py": "84da3a32cccf309a6cc7106e25a7afb282f9c57acc4dbd3ab2c6c7694a22baf9",
    "tools/research/run_qwen38_kv_precision_r2.py": "fdf2d469a652381821af23d7d4612898d9ec65d1f4fae5323ee9e50efa7d8158",
    "tools/research/run_mtp_persistence_first_instance.py": "2a70897b1a9d73fb5fbf77159fa8c6706a41786c95bd428061e8963b8abde7b1",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def verify_sources_r2() -> tuple[dict[str, Any], list[pathlib.Path]]:
    ledger: dict[str, Any] = {"host": {}, "wsl": {}}
    paths: list[pathlib.Path] = []
    for relative, expected in HOST_INPUTS.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen source mismatch: {relative}: {actual} != {expected}")
        ledger["host"][relative] = {"bytes": path.stat().st_size, "sha256": actual}
        paths.append(path)
    for path, expected in r1.base.infra.EXPECTED_WSL.items():
        size = r1.base.infra.stat_wsl(path)
        digest = r1.base.infra.sha256_wsl(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"frozen WSL identity mismatch: {path}: {size} {digest}")
        ledger["wsl"][path] = {"bytes": size, "sha256": digest}
    return {"q8_sources_and_artifacts_verified": True, "inputs": ledger}, paths


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    r1.TASK_ID = TASK_ID
    r1.verify_sources = verify_sources_r2
    receipt = r1.execute(outdir)

    raw = outdir / "raw"
    scores_path = raw / "actual_scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    scores["q8_sources_and_artifacts_verified"] = True
    r1.base.write_json(scores_path, scores)
    receipt["gates"]["source_integrity"]["metric"] = "q8_sources_and_artifacts_verified"

    _, frozen_paths = verify_sources_r2()
    evidence_files = sorted(
        path for path in raw.rglob("*")
        if path.is_file() and path.name != "receipt.json"
    )
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started,
        started_monotonic=mono,
        input_paths=[*frozen_paths, *evidence_files],
        packages=[],
        runtime={
            "execution_mode": "physical_q8_utility_noninferiority_clean_provenance",
            "requests": scores["recorded_requests"],
            "mutable_handoff_excluded": True,
        },
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    receipt["provenance"] = provenance
    receipt.pop("receipt_fingerprint", None)
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r1.base.write_json(raw / "receipt.json", receipt)

    failed = [name for name, gate in receipt["gates"].items() if not gate["pass"]]
    claim = (
        "QWEN38_Q8_KV_UTILITY_NONINFERIOR_R2"
        if not failed else "QWEN38_Q8_KV_UTILITY_NOT_NONINFERIOR_R2"
    )
    comparison = scores["paired_q8_minus_f16_accuracy"]
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"F16/Q8 accuracy `{scores['f16_accuracy']:.4f}`/`{scores['q8_accuracy']:.4f}`; "
        f"Q8-minus-F16 paired-bootstrap 95% interval "
        f"`[{comparison['lower_95']:.4f}, {comparison['upper_95']:.4f}]`. "
        f"Q8 saved `{scores['vram_saving_mib']:.1f}` MiB at "
        f"`{scores['q8_vs_f16_throughput_ratio']:.4f}x` throughput. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        verify_sources_r2()
        assert len(r1.panel()) == 128
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
