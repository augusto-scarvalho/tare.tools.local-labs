# Qwen3.8-27B fine-tune / merge landscape for agentic coding (checked 2026-08-16)

**Headline: the ecosystem is 2 days old and mostly empty. Nothing yet beats BASE for agentic coding.
ThinkingCap for 3.8 does NOT exist yet (only 3.6). Use base; watch the list below.**

Correction to our own memory: **ThinkingCap is by BottleCapAI (bottlecapai), not DavidAU.** It's a
token-EFFICIENCY tune (same accuracy, ~50% fewer thinking tokens) — separate from DavidAU's
NEO/Brainstorm merge family. (See memory `tc-humaneval-harness-bug` — our ~93%-on-code result was the
3.6/earlier ThinkingCap line.)

## Ranked for agentic coding on the 3090

| # | Variant | What / verdict | MTP |
|---|---|---|---|
| 1 | **Base Qwen3.8-27B** — Unsloth UD-Q4_K_XL **or bartowski imatrix** | **USE THIS.** 3.8 *is* the coding/agentic upgrade (+15.1% Terminal-Bench vs 3.6, better nested tool-call parsing, developer-role). bartowski's imatrix calibrates on tool-calling+reasoning convos → best-matched community quant; A/B it vs Unsloth. | **CONFIRMED present** (nextn @ blk.64) in UD-Q4_K_XL, IQ4_XS, bartowski Q4_K_M — 2026-08-16 |
| 2 | JonathanColetti/Qwen3.8-27B-Uncensored-GGUF | Heretic refusal-removal, cap cost ~−0.5pt (no code bench). **Only if refusals actually block the agent** (security/exploit-adjacent code). Else pure tail-risk. | **Retained + verified** (inspection script; noMTP variants labeled) |
| 3 | bottlecapai/ThinkingCap — **3.6 only** | Halves thinking tokens at ~same accuracy. **Wrong base** — swapping loses 3.8's +15.1%, and 3.8's `reasoning_effort=medium/low` kwarg is a native tune-free lever for the same goal. **Not for deploy; #1 watch item.** | Retained (bf16 in FP8 build) |
| 4 | DavidAU NEO-CODE / Heretic / Fable-Fusion | Pipeline active on 3.6 (some repos carry an explicit `-MTP-` tag) but **nothing for 3.8 yet.** Watch. | Tracks MTP in 3.6 naming |
| — | orcarouter FP8 / Zynerji Ektome / NVFP4 builds | **Avoid:** FP8/NVFP4 don't fit a 3090 or are Blackwell-targeted; Ektome has no GGUF / unverified MTP. | n/a |

**No official or community "Qwen3.8-Coder" exists** — 3.8 base appears to have absorbed the coder role.

## Universal MTP mitigation
[a4lg/Qwen3.8-27B-MTP-ONLY-GGUF](https://huggingface.co/a4lg/Qwen3.8-27B-MTP-ONLY-GGUF) restores the MTP
head to any derivative that lacks it (via `--model-draft` or grafting). So an MTP-stripped future merge
is recoverable — at the cost of a `spec-drafter-bench.sh` run to confirm acceptance holds.

## Watch list (re-check in 2–4 weeks)
1. **`bottlecapai/ThinkingCap-Qwen3.8-27B`** — most likely high-value arrival; org was active 3 days
   ago. If it lands, first tune to test (our prior ThinkingCap ≈93% on code) — gguf-dump MTP on day one.
2. DavidAU `Qwen3.8-27B-NEO-CODE-*` / Heretic lines.
3. huihui-ai abliteration of any 3.8 ThinkingCap (they did 3.6 within days).
4. Any official Qwen3.8-Coder (none announced; may never come).

Sources: bottlecapai org · JonathanColetti Uncensored GGUF · bartowski/Qwen3.8-27B-GGUF ·
DavidAU NEO-CODE 3.6 · a4lg MTP-ONLY GGUF · Qwen/Qwen3.8-27B (full URLs in the research transcript).
