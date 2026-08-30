#!/usr/bin/env python3
"""Independently score retained HYPER-01 R2 generated tensors."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

from hyper01_factorized_worker_r2 import TARGET_KEY_A, TARGET_KEY_B


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs=4, required=True)
    parser.add_argument("--generated", type=pathlib.Path, required=True)
    parser.add_argument("--states", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    targets = []
    for path_text in args.checkpoints:
        with safe_open(path_text, framework="pt", device="cpu") as handle:
            targets.append((handle.get_tensor(TARGET_KEY_A).float(), handle.get_tensor(TARGET_KEY_B).float()))

    rows = []
    with safe_open(str(args.generated), framework="pt", device="cpu") as generated:
        for seed in args.seeds:
            cosines = []
            for index, (target_a, target_b) in enumerate(targets):
                a = generated.get_tensor(f"seed_{seed}.target_{index}.a").float()
                b = generated.get_tensor(f"seed_{seed}.target_{index}.b").float()
                cosine = torch.nn.functional.cosine_similarity(
                    (b @ a).flatten(), (target_b @ target_a).flatten(), dim=0,
                )
                cosines.append(float(cosine.item()))
            rows.append({"seed": seed, "target_cosines": cosines, "mean": statistics.mean(cosines)})
    with safe_open(str(args.states), framework="pt", device="cpu") as state_file:
        retained_seed_states = len({key.split(".", 1)[0] for key in state_file.keys()})
    payload = {
        "schema": "hyper01-factorized-independent-score-r2",
        "rows": rows,
        "mean_weight_delta_cosine": statistics.mean(row["mean"] for row in rows),
        "worst_seed_mean_cosine": min(row["mean"] for row in rows),
        "retained_seed_states": retained_seed_states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
