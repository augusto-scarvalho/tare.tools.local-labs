"""The single source of truth for model paths and MoE geometry.

Before this module the same map lived, copied, in serve.py, quality_bench.py and
ab_isolate.py. Three copies drift: a path fixed in one is still wrong in the others, and
this project has already lost a run to a stale constant. One registry, two views.

TWO KEY SCHEMES, both deliberate and both preserved:

  * QUANT-KEYED (`MODELS`) -- "qwen36-35b-q4", "qwen36-35b-q5", ... Quant is part of the
    key because quality_bench treats it as a FACTOR to screen, not a property of a model:
    two quantisations of the same weights are two entries on purpose.

  * ARCH-KEYED (`ARCH_DEFAULTS`) -- "qwen36-35b", "gpt-oss-20b", ... One entry per
    architecture at its default quant, because ab_isolate expresses offload as a fraction
    of layers and varies quant separately; there, quant must NOT be folded into the key.

Geometry (block_count, n_expert, n_expert_used) is a property of the ARCHITECTURE and is
identical across a model's quantisations -- quantising changes bytes per weight, not the
layer or expert count. So every quant entry of one arch carries the same geometry.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str            # quant-keyed id, e.g. "qwen36-35b-q4"
    path: str           # GGUF path inside the distro
    arch: str           # architecture id, e.g. "qwen36-35b"
    quant: str          # e.g. "Q4_K_M"
    block_count: int    # layers; the offload axis is a fraction of this
    n_expert: int       # experts per MoE layer
    n_expert_used: int  # experts routed per token

    @property
    def as_ab_tuple(self) -> tuple[str, int, int, int]:
        """The (gguf, block_count, n_expert, n_expert_used) shape ab_isolate unpacks."""
        return (self.path, self.block_count, self.n_expert, self.n_expert_used)


# Geometry per architecture, asserted once so a typo in a quant entry cannot silently
# introduce a second, disagreeing value for the same architecture.
# Nemotron-120B (88, 512, 22) was DISCARDED 2026-07-31: no quant of it fits this box within
# the safety envelope. Measured, not guessed -- even bartowski IQ1_S (46.4 GB, the smallest
# quant either quantiser makes) loads only at ncmoe=50 with 594 MB VRAM free (reserve 4 GB)
# and 2.0 GB Windows free (reserve 16 GB); every other ncmoe blows one envelope or the other.
# The 12B-active transfer-bound regime (STATUS §B1) is unreachable here without 128 GB RAM.
# Both model files were deleted. The historical rejection data in runs/ab-genpin-nemotron-120b/
# and runs/residency_nemotron-120b.json is kept as the evidence.
_GEOM = {
    "qwen36-35b":    (40, 256, 8),
    "qwen3-30b":     (48, 128, 8),
    "gpt-oss-20b":   (24, 32, 4),
    # Laguna-S-2.1 was tried and DISCARDED 2026-08-01. It loads, but testing PINNING needs
    # mmap, and mmap holds the whole GGUF resident in RAM; even the Q2_K_XL (39.7 GB) leaves
    # Windows at 5.5 GB available from a clean 44 GB baseline -- below the 16 GB reserve. The
    # binding rule: a pinning test needs the file <= ~26 GB, and every Laguna-S / Nemotron
    # quant that is also high-active (transfer-bound) is larger. Files deleted. See STATUS §B1.
}

# arch -> {quant-suffix: (path, quant-name)}. The default quant per arch is the FIRST.
_FILES = {
    "qwen36-35b": [
        ("q4", "/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "Q4_K_M"),
        ("q5", "/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf", "Q5_K_M"),
        ("q6", "/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q6_K.gguf", "Q6_K"),
        ("q8", "/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf", "Q8_0"),
    ],
    "qwen3-30b": [
        ("q4", "/home/augus/models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf", "Q4_K_M"),
    ],
    "gpt-oss-20b": [
        ("q4", "/home/augus/models/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf", "Q4_K_M"),
    ],
    # laguna-s and nemotron-120b removed -- see the _GEOM notes above for why both were
    # discarded (no high-active MoE quant fits a pinning test's file<=26GB rule on this box).
}

MODELS: dict[str, ModelSpec] = {}
for _arch, _variants in _FILES.items():
    _blocks, _ne, _nu = _GEOM[_arch]
    for _suffix, _path, _quant in _variants:
        _key = f"{_arch}-{_suffix}"
        MODELS[_key] = ModelSpec(_key, _path, _arch, _quant, _blocks, _ne, _nu)

# arch -> its default-quant ModelSpec (the first quant listed for that arch).
ARCH_DEFAULTS: dict[str, ModelSpec] = {
    arch: MODELS[f"{arch}-{variants[0][0]}"] for arch, variants in _FILES.items()
}


def ab_models() -> dict[str, tuple[str, int, int, int]]:
    """ab_isolate's arch-keyed (gguf, block_count, n_expert, n_expert_used) map."""
    return {arch: spec.as_ab_tuple for arch, spec in ARCH_DEFAULTS.items()}


if __name__ == "__main__":
    # Geometry must be consistent across every quant of an architecture.
    for spec in MODELS.values():
        assert (spec.block_count, spec.n_expert, spec.n_expert_used) == _GEOM[spec.arch], spec
    # The two legacy schemes must reproduce exactly what the copies used to hold.
    assert MODELS["qwen36-35b-q4"].path.endswith("UD-Q4_K_M.gguf")
    assert ab_models()["gpt-oss-20b"] == (
        "/home/augus/models/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf", 24, 32, 4)
    print(f"models registry OK: {len(MODELS)} quant entries, "
          f"{len(ARCH_DEFAULTS)} architectures")
