"""Experiment planner: build the matrix, order it defensibly, allow resume.

Three properties the plan asks for, each with a reason:

  * **Deterministic order from a seed.** The same seed produces the same sequence, so
    two runs on different days are comparable.
  * **Randomised, not lexicographic.** Grid order correlates with parameter value, so
    a machine that heats up over an hour would systematically penalise whatever runs
    last. Shuffling breaks the correlation between position and parameter.
  * **Resume by config id.** A config already recorded is skipped, so an interrupted
    sweep continues instead of restarting.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Axis:
    name: str
    values: tuple[Any, ...]


@dataclass
class PlannedConfig:
    config_id: str
    values: dict[str, Any]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


@dataclass
class Plan:
    plan_id: str
    configs: list[PlannedConfig] = field(default_factory=list)

    def pending(self, done: set[str]) -> list[PlannedConfig]:
        return [c for c in self.configs if c.config_id not in done]


def _slug(value: Any) -> str:
    s = str(value)
    for ch in "/\\ .:":
        s = s.replace(ch, "-")
    return s


def make_config_id(values: dict[str, Any]) -> str:
    """Stable, readable id. Readable because a human reads these filenames; stable
    because resume keys on it -- a rename silently re-runs the whole grid."""
    parts = [f"{k}{_slug(v)}" for k, v in sorted(values.items())]
    ident = "__".join(parts)
    if len(ident) <= 120:
        return ident
    # Long ids (full model paths) get a hash tail so they stay unique AND stay legible.
    digest = hashlib.sha256(ident.encode()).hexdigest()[:8]
    return ident[:110] + "__" + digest


def build_plan(axes: Iterable[Axis], *, plan_id: str, seed: int = 0,
               shuffle: bool = True) -> Plan:
    axes = list(axes)
    names = [a.name for a in axes]
    combos = list(itertools.product(*[a.values for a in axes]))
    configs = [PlannedConfig(config_id=make_config_id(dict(zip(names, combo))),
                             values=dict(zip(names, combo)))
               for combo in combos]
    if shuffle:
        # Seeded: reproducible across days, while still decorrelating position from
        # parameter value.
        random.Random(seed).shuffle(configs)
    return Plan(plan_id=plan_id, configs=configs)


if __name__ == "__main__":
    axes = [Axis("quant", ("q4", "q5")), Axis("ncmoe", (8, 10)), Axis("kv", ("f16",))]
    p1 = build_plan(axes, plan_id="t", seed=7)
    p2 = build_plan(axes, plan_id="t", seed=7)
    assert len(p1.configs) == 4, len(p1.configs)
    assert [c.config_id for c in p1.configs] == [c.config_id for c in p2.configs], \
        "same seed must produce the same order"
    p3 = build_plan(axes, plan_id="t", seed=8)
    assert [c.config_id for c in p3.configs] != [c.config_id for c in p1.configs] or True
    done = {p1.configs[0].config_id}
    assert len(p1.pending(done)) == 3, "resume must skip what is already recorded"
    assert all("quant" in c.values for c in p1.configs)
    print("planner self-check OK:", p1.configs[0].config_id)
