from __future__ import annotations

import json
import pathlib

import pytest

from tools.research import run_fleet_context_envelope as envelope
from tools.research import run_fleet_context_interference as interference
from tools.research import run_retained_fleet_rebind_r3 as target


@pytest.mark.parametrize(
    ("source_task", "generator"),
    [
        ("BACKLOG-FLEET-CONTEXT-ENVELOPE-03", envelope.make_prompt),
        ("BACKLOG-FLEET-CONTEXT-INTERFERENCE-01", interference.interference_prompt),
    ],
)
def test_historical_context_prompts_reconstruct_without_recursive_patch(source_task, generator):
    raw = pathlib.Path("runs/research") / source_task / "raw"
    rows = [json.loads(line) for line in (raw / "samples.jsonl").read_text().splitlines()]
    cases = json.loads((raw / "case_manifest.json").read_text())["cases"]
    registry = json.loads(pathlib.Path("config/qualified_model_fleet.json").read_text())
    artifacts = json.loads((raw / "artifact_hashes.json").read_text())

    metrics, analysis = target.context_metrics(rows, cases, registry, artifacts, generator)

    assert metrics["prompt_hash_reconstruction_rate"] == 1.0
    assert analysis["prompt_hashes_reconstructed"] == 72


def test_successor_config_does_not_mutate_frozen_base_config():
    original = target.base.CONFIGS["BACKLOG-FLEET-CONTEXT-ENVELOPE-04"]["claim_pass"]
    configs = target.successor_configs()
    configs["BACKLOG-FLEET-CONTEXT-ENVELOPE-05"]["claim_pass"] = "mutated"
    assert target.base.CONFIGS["BACKLOG-FLEET-CONTEXT-ENVELOPE-04"]["claim_pass"] == original
