from __future__ import annotations

import json
import pathlib

from tools.research import run_fleet_context_envelope as envelope
from tools.research import run_retained_fleet_rebind_r2 as target


def test_historical_context_prompts_reconstruct_with_canonical_digest():
    raw = pathlib.Path("runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw")
    rows = [json.loads(line) for line in (raw / "samples.jsonl").read_text().splitlines()]
    cases = json.loads((raw / "case_manifest.json").read_text())["cases"]
    registry = json.loads(pathlib.Path("config/qualified_model_fleet.json").read_text())
    artifacts = json.loads((raw / "artifact_hashes.json").read_text())

    metrics, analysis = target.context_metrics(
        rows, cases, registry, artifacts, envelope.make_prompt
    )

    assert metrics["prompt_hash_reconstruction_rate"] == 1.0
    assert analysis["prompt_hashes_reconstructed"] == 72
