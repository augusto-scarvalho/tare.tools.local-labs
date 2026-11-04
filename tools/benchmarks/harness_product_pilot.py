#!/usr/bin/env python3
"""Frozen Track H structural-evidence and contract pilot."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model_lifecycle.agent_harness import build_evidence_pack, full_file_control  # noqa: E402


TASKS = [
    ("mode", "lmctl require mode for port write mode state fail closed SERVE LAB transition",
     "tools/benchmarks/lmctl.py", ["tools/benchmarks/lmctl.py", "tests/test_lmctl_mode.py"]),
    ("provenance", "verify hf source manifest all valid expected total bytes actual sha256 pinned",
     "tools/analysis/verify_hf_source_manifest.py", [
         "tools/analysis/verify_hf_source_manifest.py", "tools/analysis/hf_model_manifest.py",
         "runs/provenance/LAB-PROV-002-REQUANT-2026-08-22/DECISION_PACKET.md"]),
    ("visual", "vlm coding suite deterministic visual clause image OCR",
     "tools/benchmarks/vlm_coding_suite.py", [
         "tools/benchmarks/vlm_coding_suite.py", "tools/benchmarks/vlm_vqa_bench.py"]),
    ("agent", "agent irreversible tool recovery policy no blind retry",
     "tools/benchmarks/agent_irreversible_policy.py", [
         "tools/benchmarks/agent_irreversible_policy.py", "tools/benchmarks/agent_robustness_v2.py",
         "tools/benchmarks/agent_suite_v2.py"]),
    ("context", "context vram envelope reserve ladder",
     "tools/benchmarks/context_vram_envelope.py", [
         "tools/benchmarks/context_vram_envelope.py", "tools/benchmarks/context_suite_v2.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", help="Optional local Hugging Face tokenizer path")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    def count_tokens(texts: list[str]) -> int:
        if tokenizer is None:
            return sum(max(1, len(text.encode("utf-8")) // 4) for text in texts)
        return sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)

    rows = []
    for task_id, query, required, controls in TASKS:
        budget = 10 if task_id == "mode" else 8
        pack = build_evidence_pack(
            args.root, query, max_files=budget, max_chunks=budget, context_lines=1,
        )
        _, control_payload = full_file_control(args.root, controls)
        selected = sorted({chunk.path for chunk in pack.chunks})
        pack_tokens = count_tokens([
            f"{chunk.path}:{chunk.start_line}\n{chunk.text}" for chunk in pack.chunks
        ])
        control_tokens = count_tokens([
            f"{path}\n{text}" for path, text in control_payload.items()
        ])
        reduction = 1.0 - pack_tokens / control_tokens
        row = {
            "id": task_id, "query": query, "required": required,
            "file_and_chunk_budget": budget,
            "selected_files": selected, "required_recalled": required in selected,
            "pack_tokens": pack_tokens,
            "control_tokens": control_tokens,
            "reduction": reduction, "source_digest": pack.source_digest,
            "chunks": [asdict(chunk) for chunk in pack.chunks],
        }
        rows.append(row)
        print(f"{task_id}: recall={row['required_recalled']} reduction={reduction:.1%}")
    total_pack = sum(row["pack_tokens"] for row in rows)
    total_control = sum(row["control_tokens"] for row in rows)
    report = {
        "schema_version": 1, "tasks": rows,
        "recall": sum(row["required_recalled"] for row in rows),
        "task_count": len(rows), "tokenizer": args.tokenizer or "utf8_bytes_div_4",
        "pack_tokens": total_pack, "control_tokens": total_control,
        "aggregate_reduction": 1.0 - total_pack / total_control,
    }
    report["decision"] = "PASS" if (
        report["recall"] == len(rows)
        and report["aggregate_reduction"] >= 0.30
        and all(row["reduction"] >= 0.30 for row in rows)
    ) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))
    if report["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
