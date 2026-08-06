"""Named serve profiles — the exact `llama-server` line for a model we run repeatedly.

Kept SEPARATE from `models.py` on purpose. `models.py` is the geometry registry that
`ab_isolate` consumes (block_count / n_expert drive the offload axis); folding a judge or
a deploy config into it would pollute that and break its self-checks. A serve profile is a
different thing: the full, blessed command line for ONE way of running a model, so that
bringing up a judge quorum is `lmctl serve gemma-judge` instead of copying a shell script
out of scratch/ and re-deriving the flags.

Every profile here was lifted verbatim from a source that already proved it works:
  * the two judges from `scratch/serve_{mistral,gemma}_judge.sh` (A2 Gate 3);
  * `deploy-moe` from DEPLOY.md's TL;DR (validated end-to-end 2026-08-04, 127-130 t/s);
  * `deploy-fable` from the A2 deploy candidate (dense l1.0).

FLAG NAMES: the long forms `--batch-size` / `--ubatch-size` are used everywhere, not the
short `-b`/`-ub`. DEPLOY.md records that the current deploy binary (lifecycle @ 068764d92)
REJECTS the short forms (`error: invalid argument: --batch`); the long forms are canonical
and always accepted, so they are the safe choice across builds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import MODELS

# The consolidated fork / deploy binary (branch `lifecycle`). This is THE binary now, newer
# than the older llama.cpp-local that serve.py still points at. Judges + deploy both use it.
DEFAULT_BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"


@dataclass(frozen=True)
class ServeSpec:
    """One blessed way to run one model. `flags` are the llama-server args AFTER
    `-m <model> --host 0.0.0.0 --port <port>`, which lmctl adds itself."""
    name: str
    model_path: str          # GGUF path INSIDE the distro
    port: int
    flags: tuple[str, ...]
    bin: str = DEFAULT_BIN
    env: dict[str, str] = field(default_factory=dict)   # env vars INSIDE the distro
    note: str = ""


SERVE_PROFILES: dict[str, ServeSpec] = {
    # --- A2 Gate 3 writing-quality judges (from scratch/serve_*_judge.sh) -------------
    "mistral-judge": ServeSpec(
        name="mistral-judge",
        model_path="/home/augus/models/mistral-small-24b-heretic/"
                   "Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2.i1-Q4_K_M.gguf",
        port=8090,
        flags=("-ngl", "99", "-fa", "on", "--ctx-size", "8192",
               "--batch-size", "2048", "--ubatch-size", "2048"),
        note="A2 Gate 3 writing judge. Dense 24B Q4=14GB fits fully in VRAM (-ngl 99). "
             "Heretic abliteration (coherence-preserving). Prefill-heavy workload."),
    "gemma-judge": ServeSpec(
        name="gemma-judge",
        model_path="/home/augus/models/gemma4-26b-heretic/"
                   "Gemma-4-26B-A4B-it-heretic-antislop.i1-Q4_K_M.gguf",
        port=8091,
        flags=("-ngl", "99", "-fa", "on", "--ctx-size", "16384",
               "--batch-size", "2048", "--ubatch-size", "2048"),
        note="A2 Gate 3 2nd local judge. Gemma-4-26B MoE (~4B active), Q4=16.8GB in VRAM. "
             "-c 16384 (not 8192): THINKING model, 4 concurrent reqs overflow an 8k KV."),

    # --- Deploy configs ----------------------------------------------------------------
    "deploy-moe": ServeSpec(
        name="deploy-moe",
        model_path=MODELS["qwen36-35b-mtp-q4"].path,
        port=8080,
        flags=("-fa", "on", "--n-cpu-moe", "8", "--ctx-size", "8192",
               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
               "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
               "--batch-size", "2048", "--ubatch-size", "2048", "--jinja"),
        note="Project deploy (DEPLOY.md TL;DR, validated 2026-08-04: 127-130 t/s, 83% "
             "accept). Qwen3.6-35B MoE, ncmoe=8 + MTP self-draft. For 128k: -c 131072 "
             "--cache-type-k/v q4_0."),
    "deploy-fable": ServeSpec(
        name="deploy-fable",
        model_path=MODELS["fable-tc-l1.0-q4"].path,
        port=8080,
        flags=("-ngl", "99", "-fa", "on", "--ctx-size", "8192", "--jinja"),
        note="A2 deploy candidate: concise+uncensored Fable dense l1.0 (Qwen3.6-27B). "
             "Q4 fits fully in VRAM. MTP omitted (verify the merge carries a draft head "
             "before adding --spec-type draft-mtp)."),

    # --- Track M-A · VLMs (vision) ------------------------------------------------------
    # A VLM needs its multimodal projector passed with --mmproj alongside the model; the
    # deploy-fork llama-server has vision built in (libmtmd, supports QWEN3VL + GEMMA4V —
    # verified in tools/mtmd/clip.cpp). All on port 8092 (the "vision slot") since only one
    # serves at a time. Current-gen picks (2026-08): Qwen3-VL (unsloth Q4) + Gemma-4 (ggml-org).
    "qwen3-vl-30b": ServeSpec(
        name="qwen3-vl-30b",
        model_path="/home/augus/models/qwen3-vl-30b/Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
        port=8092,
        flags=("--mmproj", "/home/augus/models/qwen3-vl-30b/mmproj-F16.gguf",
               "-ngl", "99", "-fa", "on", "--ctx-size", "8192", "--jinja"),
        note="M0 FLAGSHIP. Qwen3-VL-30B-A3B (current-gen Qwen vision, MoE ~3B active = fast). "
             "UD-Q4_K_XL 17.7GB + mmproj 1.1GB fits ~19GB in VRAM (~5GB headroom). Top OCR."),
    "qwen3-vl-8b": ServeSpec(
        name="qwen3-vl-8b",
        model_path="/home/augus/models/qwen3-vl-8b/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
        port=8092,
        flags=("--mmproj", "/home/augus/models/qwen3-vl-8b/mmproj-F16.gguf",
               "-ngl", "99", "-fa", "on", "--ctx-size", "8192", "--jinja"),
        note="Nimble dense Qwen3-VL-8B (~5GB + 1.2GB mmproj). Fast iteration / low-VRAM."),
    "gemma-4-12b-vision": ServeSpec(
        name="gemma-4-12b-vision",
        model_path="/home/augus/models/gemma-4-12b-vision/gemma-4-12B-it-Q4_0.gguf",
        port=8092,
        flags=("--mmproj",
               "/home/augus/models/gemma-4-12b-vision/mmproj-gemma-4-12B-it-Q8_0.gguf",
               "-ngl", "99", "-fa", "on", "--ctx-size", "8192", "--jinja",
               "--reasoning-budget", "256"),
        note="Gen-4 Gemma vision (Google lineage), dense 12B Q4_0 ~7GB. Cross-family "
             "comparison vs Qwen3-VL. THINKING model -> its wall-clock is dominated by "
             "reasoning tokens, not decode rate. --reasoning-budget 256 caps the think "
             "phase: MEASURED on MMStar-150 to KEEP full accuracy (0.580 vs 0.573 "
             "unbounded) at 1.4x faster (528s vs 740s); 256 saturates accuracy, 128 "
             "drops it to 0.520. `--reasoning off` = 9.3x faster (80s) but 0.480 = "
             "8B-tier (Qwen-8b dominates there). Repo also ships an MTP draft head "
             "(google/...assistant) that draft-mtp can wire for ~2x more decode, "
             "stacking with the budget cap (see M_A_VLM_PERF.md). Tune the budget up "
             "for harder visual-reasoning workloads than MCQ."),
}


def resolve_model_path(name: str) -> str | None:
    """A serve target may be a profile name OR a bare MODELS registry key. Return the
    GGUF path for either, else None."""
    if name in SERVE_PROFILES:
        return SERVE_PROFILES[name].model_path
    if name in MODELS:
        return MODELS[name].path
    return None


if __name__ == "__main__":  # self-check: python -m model_lifecycle.serve_profiles
    for spec in SERVE_PROFILES.values():
        assert spec.model_path.startswith("/home/augus/models/"), spec
        assert spec.port > 0 and spec.flags, spec
    print(f"serve profiles OK: {len(SERVE_PROFILES)} profiles "
          f"({', '.join(sorted(SERVE_PROFILES))})")
