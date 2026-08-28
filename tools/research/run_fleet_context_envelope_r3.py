#!/usr/bin/env python3
"""Route only context-envelope tokenization to the verified active backend."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.research import run_fleet_context_envelope as r2

TASK_ID = "BACKLOG-FLEET-CONTEXT-ENVELOPE-03"
PRE_REG_SHA256 = "d0a62bbec7f7b64840098abdbd869cc3cdc783a613fbad88a54f58daed9bcac4"
SOURCE_HASHES = {
    "config/research_backlog_admissions/BACKLOG-FLEET-CONTEXT-ENVELOPE-03.json": "439ff053d46c1f0099300ec695d7ce23cd5b413926b40da70593f0ba153904d0",
    "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-02/PRE_REGISTRATION.md": "e851482e89abd1bc417db405921a1b0d4a4195f17452689eebb41b6f23e845ae",
    "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-02/PIPELINE.json": "5d7ee0aef3b9ed30ef6b15fc1cbca9257d8d23bf01b83932e35b2206ffaf37a6",
    "tools/research/run_fleet_context_envelope.py": "1ebb0c07145edd48f1fcc7d8f97b248a954ce4faa554e4d6f220c6d693eb857b",
    "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-02/runner.stderr.log": "54640b924eb5820fd719db98ab60b2de9920771e81c8f2681058152ca2aff25a",
    "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-02/raw/recovery_state.json": "23cc4f5f154dc1f0fbd215c9acde15e710388a46e5cc6afe43930507d91f12ab",
    "runs/autonomous/EXPERIMENT-WATCH-2026-08-27-FLEET-CONTEXT-R2/FINAL.json": "3bbdb733bf8e9930b5d1086467de452869bc7298e06ad9084174d4df20fa6426",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
}


def backend_token_count(text: str) -> int:
    gateway = r2.fleet.gateway_status()
    port = int(gateway.get("backend_port") or 0)
    if port != 18080 or not gateway.get("backend_healthy"):
        raise RuntimeError(f"unexpected tokenizer backend identity: {gateway}")
    status, response = r2.fleet.http_json(
        f"http://127.0.0.1:{port}/tokenize",
        {"content": text, "add_special": False},
        timeout=180.0,
    )
    tokens = response.get("tokens") if status == 200 else None
    if not isinstance(tokens, list):
        raise RuntimeError(f"backend tokenizer failure: {status} {response}")
    return len(tokens)


def configure() -> None:
    r2.TASK_ID = TASK_ID
    r2.PRE_REG_SHA256 = PRE_REG_SHA256
    r2.SOURCE_HASHES = SOURCE_HASHES
    r2.token_count = backend_token_count
    r2.__file__ = __file__


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    configure()
    receipt = r2.execute(outdir)
    result = outdir / "RESULT.md"
    text = result.read_text(encoding="utf-8")
    text = text.replace(
        "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_MEASURED_R2",
        "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_MEASURED_R3",
    ).replace(
        "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R2",
        "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R3",
    )
    result.write_text(text, encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert TASK_ID.endswith("-03")
        assert PRE_REG_SHA256
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
