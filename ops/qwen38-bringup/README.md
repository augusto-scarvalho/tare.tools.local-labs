# Qwen3.8-27B bring-up runbook (agentic coding, single 3090)

**Goal:** stand up Qwen3.8-27B (dense GDN-hybrid, released 2026-08-14) as our agentic-coding
endpoint on the 24GB 3090, fully GPU-resident, with MTP spec-decode and working prefix reuse.

**Owner method (see memory `sweep-first-and-squeeze`):** sweep to find the point before A/B;
each phase below is DEPENDENCY-GATED — do not proceed until the acceptance test passes. Document
the observed numbers back into each script's header (that is how `kv-quant-bench.sh` /
`spec-drafter-bench.sh` became standing gates).

Hardware: RTX 3090 24GB, ~23GB usable (headless via iGPU, memory `hardware-ram-64gb` /
`gpu-prefs-igpu-change`); **64GB** system RAM; WSL Ubuntu-24.04, models in `/home/augus/models`.

---

## What our OWN prior work already settled (do not re-derive; these override the generic advice)

This model is the same architecture FAMILY as our deploy model (Qwen3.6-35B-A3B = GDN-hybrid+MTP),
so these transfer (re-confirm on the dense 27B in Phase 3, but expect them to hold):

1. **KV quant: symmetric ONLY.** Our build compiles only symmetric FA-KV kernels
   (f16/f16, q4_0/q4_0, q8_0/q8_0, bf16/bf16). **K≠V (e.g. K=q8/V=q4) drops attention to the CPU
   backend → −57% decode.** So the "keep K at q8, compress V" split that generic guides (and our
   own first-pass plan) recommend is WRONG for this build unless rebuilt with
   `GGML_CUDA_FA_ALL_QUANTS=ON` — and even then it's dominated. Source: `ops/kv-quant-bench.sh`.
2. **q4_0 KV is ~lossless on GDN hybrids** (QK-Norm kills the outliers INT4 fights). So for 256k
   use **q4_0/q4_0 symmetric** = smallest (4.1GB @256k) AND no measurable quality loss. No need to
   pay q8's 8.2GB unless Phase 3 recall says otherwise on the dense 27B.
3. **Spec-decode: `--spec-type draft-mtp` ALONE** wins ~1.7×; never stack a second drafter.
   Source: `ops/spec-drafter-bench.sh`.

Net effect on the launch config: because q4_0 KV is lossless and cheap, we may not even need to
drop to 128k — **UD-Q4_K_XL (17.9GB) + q4_0/q4_0 KV @256k (4.1GB) ≈ 22GB** could sit fully
resident. Phase 3 decides 256k-vs-128k and weight quant on measured recall, not a priori.

---

## Phase 0 — download + inventory  (gate: files present, sha ok)
```bash
huggingface-cli download unsloth/Qwen3.8-27B-GGUF \
  --include "*UD-Q4_K_XL*" --include "*IQ4_XS*" \
  --local-dir /home/augus/models/qwen38-27b
# vision NOT needed for code -> skip mmproj.
```
Also grab a fallback MTP-only draft in case the main GGUF lacks the head (see Phase 1):
`a4lg/Qwen3.8-27B-MTP-ONLY-GGUF` (Q4_K_M / Q5_K_M — NOT Q8, too big for our budget).

## Phase 1 — MTP tensors present?  (gate: MTP head found, else graft draft)
```bash
bash ops/qwen38-bringup/mtp_tensor_check.sh
```
Community reports CONFLICT on whether Unsloth's GGUF ships the MTP tensors. If absent, use the
a4lg MTP-only GGUF via `--model-draft` (or graft per their README). MTP is our +33–70% decode
lever — do not skip this check.

## Phase 2 — prefix reuse actually works?  (gate: "restored context checkpoint" in log)  ← CRITICAL
This is the single biggest agentic-latency lever AND the biggest risk. Hybrid (recurrent) models
can't do token-granular `--cache-reuse`; reuse happens only via **context checkpoints**, and there
is an OPEN upstream regression (#24055) that can force full re-prefill on some builds.
```bash
bash ops/qwen38-bringup/checkpoint_reuse_gate.sh
```
PASS = turn-2 server log shows `restored context checkpoint` and turn-2 prompt-eval time is a small
fraction of turn-1. FAIL (`forcing full prompt re-processing due to lack of cache data`) = wrong
build; pin a known-good one (Fable flagged a b9309-era build + `--checkpoint-every-n-tokens`) and
re-run. Do NOT tune MTP until this passes — under cache thrash MTP acceptance craters.

**Client hygiene (equally important):** any volatile byte early in the prompt (timestamp,
session-id, attribution header, reordered tool schemas) kills the prefix. Keep the system prompt +
tool schemas byte-stable; put volatile content at the END. (Documented real case: Claude Code's
attribution header broke caching; fix `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.)

## Phase 3 — KV-quant recall at long context  (gate: pick smallest KV that holds recall)
Confirm on the DENSE 27B that q4_0/q4_0 is still lossless (our gate proved it on the 35B MoE).
Symmetric arms only (asymmetric falls off GPU — see prior-work #1).
```bash
bash ops/qwen38-bringup/kv_recall_sweep.sh
```
Decide: if q4_0 recall == f16 recall at your target depth → ship q4_0/q4_0 @256k. If it degrades
on the dense variant → step up to q8_0/q8_0 and, if needed, drop to 128k.

## Phase 4 — MTP throughput on OUR traces  (gate: draft-mtp beats no-spec floor)
Reuse the standing gate, just point it at qwen38:
```bash
MODEL=/home/augus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  bash ops/spec-drafter-bench.sh
```
Escalate `--spec-draft-n-max` 2→3 (+ `--spec-draft-p-min 0.6`) only if measured acceptance on real
agent traces supports it. Record numbers in that script's header for qwen38.

## Phase 5 — freeze the launch config
Bake the Phase-3 KV choice and Phase-1 MTP path into the launch line. Candidate (pending Phases):
```bash
llama-server -m /home/augus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -c 262144 -ngl 999 -fa 1 --no-mmproj \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 -np 1 \
  --ctx-checkpoints 64 --cache-ram 12000 \
  --jinja --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0
# instruct/agent loop; reasoning_effort=medium via chat_template_kwargs; consider presence_penalty 0-0.5 for code.
# NOTE: checkpoint flag NAMES depend on the pinned build — Phase 2 discovers the real ones from --help.
```

---

## Custom quantization: NO-GO now (see decision below)
Community UD-imatrix quants already exist and calibrate on code; 4-bit knee ≈ 94% of FP with Q5
marginal; 64GB RAM makes the BF16 (54.7GB) workflow feasible-but-tight. Flip to GO only if a
trigger fires — full triggers + recipe in `ops/qwen38-bringup/CUSTOM_QUANT_DECISION.md`.

## Open items to verify (honest register)
- MTP tensors in Unsloth GGUF (Phase 1 resolves).
- #24055 checkpoint regression scope on the pinned build (Phase 2 resolves).
- q4_0-lossless holds on the DENSE 27B, not just the 35B MoE (Phase 3 resolves).
- Quant-quality numbers borrowed from Qwen3.6-27B studies (assumed transfer).
- ThinkingCap-style / coder fine-tunes for 3.8 — see `VARIANTS.md` (Fable research, pending).
