#!/usr/bin/env python3
"""Run the harness-bound adapter treatment behind the qualified gateway."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from collections.abc import Callable
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import sha256_file
from tools.research import run_adapt06_slop_live_r5 as r5
from tools.research import run_adapt06_slop_live_r6 as r6

TASK_ID = "BACKLOG-ADAPT06-SLOP-LIVE-07"
FLEET = ROOT / "config/qualified_model_fleet.json"
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT06-SLOP-LIVE-07.json": "ed89ec2adf6e08b1f839b4959cbbd8513b6535b3cbe35d3ec36e735ce0591bed",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-07/PRE_REGISTRATION.md": "e8865301663195eac4082db90af6c623e4f3261e72d0487223d11ffc4524f59f",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-06/raw/run.terminal.json": "e7e83bb158ced87e973b71e1fa0b18252ab8d13a585e21a78a89ab09c2a841b0",
    ROOT / "tools/research/run_adapt06_slop_live_r6.py": "4f1c94b180366123e65e94e2ccd395e29108bbf92813417d6a02f4ad39d80e20",
    FLEET: "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/receipt.json": "871fd8aeb94ff4b2e4eeb6432ba10305591c01b6270462686af0a116ec8d3a28",
    ROOT / "runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_model.safetensors": "05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122",
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_model.safetensors": "174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7",
}


def stable_gateway_exec(value: str) -> str:
    return value.split(" ; ignore_errors=", 1)[0].strip()


def make_gateway_service(
    actual_service: Callable[[], dict[str, Any]], binary: str
) -> tuple[Callable[[], dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    expected_gateway_exec: str | None = None

    def compatible() -> dict[str, Any]:
        nonlocal expected_gateway_exec
        actual = actual_service()
        values = actual.get("values", {})
        gateway_exec = str(values.get("ExecStart", ""))
        normalized = stable_gateway_exec(gateway_exec)
        if "qualified_model_gateway.py" not in gateway_exec:
            raise RuntimeError("systemd service is not the qualified-model gateway")
        if expected_gateway_exec is None:
            expected_gateway_exec = normalized
        elif normalized != expected_gateway_exec:
            raise RuntimeError("qualified-model gateway command drifted during handoff")
        observation = {
            "gateway_exec_start": gateway_exec,
            "gateway_main_pid": values.get("MainPID"),
            "gateway_nrestarts": values.get("NRestarts"),
            "gateway_active_state": values.get("ActiveState"),
        }
        observations.append(observation)
        result = copy.deepcopy(actual)
        result.setdefault("values", {}).update({
            "GatewayExecStart": gateway_exec,
            "GatewayMainPID": values.get("MainPID", ""),
            "ExecStart": (
                f"{{ path={binary} ; argv[]={binary} -m gateway-registry-placeholder.gguf "
                "; ignore_errors=no }"
            ),
        })
        result["gateway_identity"] = observation
        return result

    return compatible, observations


def execute(outdir: pathlib.Path):
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {path}: {actual}")
    fleet = json.loads(FLEET.read_text(encoding="utf-8"))
    binary = fleet["models"]["qwen38"]["runtime"]["binary"]
    binary_check = r5.run(["wsl", "-d", "Ubuntu-24.04", "--", "test", "-x", binary])
    if binary_check["returncode"] != 0:
        raise RuntimeError(f"registered llama-server is not executable: {binary_check}")

    actual_service = r5.service
    compatible, observations = make_gateway_service(actual_service, binary)
    previous_task, previous_expected = r6.TASK_ID, r6.EXPECTED
    r5.service = compatible
    r6.TASK_ID = TASK_ID
    r6.EXPECTED = EXPECTED
    try:
        receipt, metrics = r6.execute(outdir)
    finally:
        r5.service = actual_service
        r6.TASK_ID = previous_task
        r6.EXPECTED = previous_expected
    if len(observations) < 2:
        raise RuntimeError("gateway before/after identity was not observed")
    return receipt, metrics, observations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics, observations = execute(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "ADAPT06_GATEWAY_BOUND_CLIENT_AFFINITY_QUALIFIED_R7" if passed else "ADAPT06_GATEWAY_BOUND_CLIENT_AFFINITY_REJECTED_R7"
    failed = [name for name, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Gateway observations: `{len(observations)}`; route-correct parity: "
        f"`{metrics['route_correct_counterfactual_match_rate']:.4%}`; switch reduction: "
        f"`{metrics['requested_route_switch_reduction']:.4%}`; failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`. Claim is client-affinity only.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "gates": receipt["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
