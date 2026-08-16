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

## Phase 1 — MTP tensors present?  ✅ RESOLVED 2026-08-16: PRESENT in all 3 quants
```bash
bash ops/wsl/wslx.sh ops/qwen38-bringup/mtp_tensor_check.sh                       # default = UD-Q4_K_XL
bash ops/wsl/wslx.sh ops/qwen38-bringup/mtp_tensor_check.sh -- MODEL=<other.gguf> # any file
```
**Result:** UD-Q4_K_XL, IQ4_XS, and bartowski Q4_K_M ALL carry the MTP head — the `nextn` layer at
`blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight` (866 tensors, arch `qwen35`,
`nextn_predict_layers=1`). So `--spec-type draft-mtp` runs on the single GGUF; the a4lg MTP-only graft
is NOT needed. (Gotcha found here: the base/login python3 has no numpy → the reader false-negatives;
the gate now auto-discovers `/home/augus/sglang-venv/bin/python3`, which has numpy+gguf.)

## Phase 2 — prefix reuse actually works?  ✅ RESOLVED 2026-08-16: PASS (build 068764d92)  ← CRITICAL
This is the single biggest agentic-latency lever AND the biggest risk. Hybrid (recurrent) models
can't do token-granular `--cache-reuse`; reuse happens only via **context checkpoints**, and there
is an OPEN upstream regression (#24055) that can force full re-prefill on some builds.
```bash
bash ops/wsl/wslx.sh ops/qwen38-bringup/checkpoint_reuse_gate.sh
```
**Result:** on `/home/augus/src/llama.cpp-master` (v10159, 068764d92), a 2-turn test over a ~51k-token
shared context gave **TURN1 prompt_n=51015 (52.0s) → TURN2 prompt_n=517, cache_n=50499 (0.66s)** — the
warm turn reused 50,499 tokens and reprocessed only 517 (~78× faster prefill). On a 48-recurrent-layer
hybrid that is only possible via state checkpointing, so checkpoints work and **#24055 does NOT affect
this build.** Flags present: `--ctx-checkpoints`, `--checkpoint-min-step` (default 8192; gate lowers it
to 256 so checkpoints form within the test), `--cache-reuse`. NB this build logs reuse as "selected
slot by LCP similarity", not "restored context checkpoint" — the authoritative signal is turn-2 cache_n.
Client hygiene below still applies. (If a future build regresses: pin a known-good one and re-run.)

**Client hygiene (equally important):** any volatile byte early in the prompt (timestamp,
session-id, attribution header, reordered tool schemas) kills the prefix. Keep the system prompt +
tool schemas byte-stable; put volatile content at the END. (Documented real case: Claude Code's
attribution header broke caching; fix `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.)

## Phase 3 — KV-quant recall at long context  ✅ RESOLVED 2026-08-16: q4_0 lossless → SHIP q4_0/q4_0
Symmetric arms only (asymmetric falls off GPU — see prior-work #1).
```bash
bash ops/wsl/wslx.sh ops/qwen38-bringup/kv_recall_sweep.sh -- DEPTH=65536 NEEDLES=32
```
**Result:** at real depth 41,165 tok / 32 needles, **f16 == q8_0 == q4_0 == 100%** → q4_0/q4_0 is
**lossless** on the dense 27B (matches the 35B-MoE result in `ops/kv-quant-bench.sh`: QK-Norm kills the
outliers, so q4 KV is lossless on GDN hybrids). VRAM aside: q8_0 KV at real ~168k needed ~23.9/24GB
(borderline); q4_0 is half → another reason to ship q4. **Ship `--cache-type-k q4_0 --cache-type-v q4_0`.**

*Off-topic observation (not a KV effect):* at real ~168k with 32–48 near-identical needles, recall
collapses for ALL KV types (multi-needle interference); a SINGLE needle at 166k retrieves fine. That's
a model long-context/interference property (forgetting-regime research), out of scope for the KV choice.

## Phase 4 — MTP throughput  ✅ RESOLVED 2026-08-16: draft-mtp ~2.1x, n-max 3 wins
```bash
bash ops/wsl/wslx.sh ops/qwen38-bringup/mtp_throughput.sh    # dense analogue of spec-drafter-bench (no ncmoe)
```
**Result** (UD-Q4_K_XL, 3090, 5 reps, temp 0, enable_thinking:false):

| regime | no-spec | draft-mtp n2 | draft-mtp n3 |
|---|---|---|---|
| GEN  | 39.5 t/s | 76.6 (+94%)  | **83.6 (+112%)** |
| EDIT | 39.4 t/s | 83.3 (+111%) | **88.0 (+123%)** |

**WINNER: `--spec-type draft-mtp --spec-draft-n-max 3`** — ~2.1–2.2× on code, far exceeding the +33%
public reference (nextn head has high acceptance on deterministic code). n3 > n2 clearly; n2→n3 is a
free win here. (The MoE-flavored `ops/spec-drafter-bench.sh` is the deploy-model gate; this dense one
supersedes it for Qwen3.8.)

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
- ~~MTP tensors in Unsloth GGUF~~ ✅ RESOLVED 2026-08-16: PRESENT in all 3 quants (UD-Q4_K_XL, IQ4_XS, bartowski Q4_K_M).
- ~~#24055 checkpoint regression scope~~ ✅ RESOLVED 2026-08-16: build 068764d92 reuses prefix correctly (52s→0.66s warm turn); no regression.
- q4_0-lossless holds on the DENSE 27B, not just the 35B MoE (Phase 3 resolves).
- Quant-quality numbers borrowed from Qwen3.6-27B studies (assumed transfer).
- ThinkingCap-style / coder fine-tunes for 3.8 — see `VARIANTS.md` (Fable research, pending).
