#!/usr/bin/env python3
"""Run the context envelope with associative near-label and near-code decoys."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import sha256_file
from tools.research import run_fleet_context_envelope as core
from tools.research import run_fleet_context_envelope_r3 as backend

TASK_ID = "BACKLOG-FLEET-CONTEXT-INTERFERENCE-01"
PRE_REG_SHA256 = "bf5ef2c20ba6b08852ab14fcd2761c54230b389b34058caa0487f062d645dbe7"
SOURCE_RECEIPT = ROOT / "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/receipt.json"
SOURCE_HASHES = {
    "config/research_backlog_admissions/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01.json": "8f25d5d0d023b4bd1809ade207edf27137ec261a9408befbb2ef8464a3cbc112",
    "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/receipt.json": "17f0ec8b541f6d769dd5909ca4a44bc3f7c2813d13cc51ec8b206d74090d15d6",
    "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/case_manifest.json": "63936f59148535a54ca54221d29dd669c387c87e07f0f28361899f18ee914111",
    "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/RESULT.md": "4ebc023dae15da6be96ea2ab2c62da34d9d6471a83c7be05d6f66c47ee0783a0",
    "tools/research/run_fleet_context_envelope.py": "1ebb0c07145edd48f1fcc7d8f97b248a954ce4faa554e4d6f220c6d693eb857b",
    "tools/research/run_fleet_context_envelope_r3.py": "a674174eeeb4aebb2a5cd871ed6211085b5e1c28040d1e8041e519100ba857cf",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
}


def interference_prompt(filler_count: int, position: str, code: str) -> str:
    records = [core.filler_line(index) for index in range(filler_count)]
    prefix = "-".join(code.split("-")[:3])
    for index in range(1, 32):
        insertion = min(len(records), round(index * len(records) / 32))
        records.insert(insertion, (
            f"SECURE ACCESS RECORD [ORION-DELTA-{index:02d}]: "
            f"The access code is {prefix}-D{index:02d}.\n"))
    target = f"SECURE ACCESS RECORD [ORION-DELTA]: The access code is {code}.\n"
    insertion = 0 if position == "start" else len(records) // 2 if position == "middle" else len(records)
    records.insert(insertion, target)
    return (
        "Read the archive. Return the access code for the exact record label "
        "[ORION-DELTA]. Do not use numbered labels such as [ORION-DELTA-01]. "
        "Return only the code, with no explanation.\n\n"
        + "".join(records)
        + "\nQuestion: What is the access code for exact label [ORION-DELTA]? Return only the code."
    )


def configure() -> None:
    original_write = core.fleet.write_json
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))

    def enriched_write(path: pathlib.Path, value: Any) -> None:
        if path.name == "case_manifest.json" and isinstance(value, dict):
            value = value | {"generator": "associative_decoy_archive_v2",
                             "secure_records_per_case": 32,
                             "target_label": "ORION-DELTA", "near_label_decoys": 31}
        elif path.name == "source_execution_receipt.json":
            value = {"source_task_id": "BACKLOG-FLEET-CONTEXT-ENVELOPE-03",
                     "receipt_sha256": sha256_file(SOURCE_RECEIPT),
                     "receipt_fingerprint": source["receipt_fingerprint"]}
        original_write(path, value)

    core.TASK_ID = TASK_ID
    core.PRE_REG_SHA256 = PRE_REG_SHA256
    core.SOURCE_HASHES = SOURCE_HASHES
    core.token_count = backend.backend_token_count
    core.make_prompt = interference_prompt
    core.fleet.write_json = enriched_write
    core.__file__ = __file__


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    configure()
    receipt = core.execute(outdir)
    result = outdir / "RESULT.md"
    text = result.read_text(encoding="utf-8").replace(
        "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_MEASURED_R2",
        "QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_MEASURED_R1",
    ).replace(
        "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R2",
        "QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_NOT_CONFIRMED_R1",
    )
    result.write_text(text, encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        prompt = interference_prompt(100, "middle", "NX-Q38-04000-M0")
        assert prompt.count("SECURE ACCESS RECORD") == 32
        assert prompt.count("[ORION-DELTA]") >= 2
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
