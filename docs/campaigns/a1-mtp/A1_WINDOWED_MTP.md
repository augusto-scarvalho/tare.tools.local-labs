# A1 — Windowed / adaptive MTP: the full record (2026-08-04)

Consolidated write-up of IDEAS_BACKLOG **A1**. Canonical short entry: `STATUS.md §A1`; pre-registration card:
`EXPERIMENTS.md §A1`; backlog disposition: `IDEAS_BACKLOG.md` A1; deploy note: `DEPLOY.md`. Raw data:
`runs/a1-mtp-depth/`. Gate/tool: `ops/a1_mtp_depth_bench.py`.

## Verdict (one line)

**CUT.** Windowing the MTP draft attention has **no context depth we can serve** where it helps: the MTP
decode-t/s edge only **grows** with depth — all the way to the model's **native 262k ceiling** (+176%) — and
draft acceptance stays pristine. The draft-KV tax the lever removes only dominates near ~1M tokens, which is
unreachable (needs YaRN far past training, where quality is gone) and unusable.

## The question / premise

The panorama research doc (§22/§35) warns MTP self-speculation isn't "free speed": *"em contexto longo, o draft
também pode pagar custo de KV; arquiteturas híbridas podem reduzir o custo do target e diminuir a vantagem
relativa do MTP"*, and flags **`windowed-MTP`** [ref-105] as the bleeding-edge fix — restrict the draft/nextn
head's attention to a recent window to remove the "full-context draft-KV tax" on hybrids. A1 asked: does this
help our deploy models (Qwen3.6-35B-A3B MoE + Qwen3.6-27B dense, both GDN hybrids)?

## Mechanistic grounding (pre-registered, before any run)

`qwen35moe` = 41 blocks, a **GDN hybrid**: `full_attention_interval=4` → only **10 of 40 base layers bear a KV
cache** (full-attn at blk 3,7,…,39); the other 30 are GDN/SSM linear (no KV). **No SWA** (no `sliding_window` /
`n_swa` key). The **`nextn`/MTP head (blk 40) IS a full-attention KV-bearing layer** — verified in
`src/models/qwen35moe.cpp`: the MTP block builds `build_attn_inp_kv()` + `build_attn(...)` over the unified KV
cache ("a full-attention Qwen3.5 decoder block with MoE FFN"). So "the draft pays full-context KV" is
**mechanically real in the runtime** — the only open question was the *regime* at which it matters.

## Results

All via `ops/a1_mtp_depth_bench.py` (llama-server, ncmoe=8, q8_0 KV @8k / q4_0 @deep, isolated arms + cooldown —
the GPU-A/B variance rule; fixed generation task so only context depth varies; decode t/s = `predicted_per_second`,
accept = `draft_n_accepted/draft_n`).

### Edge vs depth — both hybrids (extremes; `a1_depth.csv`)

| model | depth (prompt_n) | no-spec t/s | draft-mtp t/s | **edge** | accept |
|---|---|---:|---:|---:|---:|
| MoE 35B-A3B | 8k (6.7k) | 84.9 | 148.4 | **+74.9%** | 99.2% |
| MoE 35B-A3B | 128k (122k) | 48.3 | 113.3 | **+134.2%** | 99.2% |
| dense 27B | 8k (6.3k) | 31.8 | 70.6 | **+121.8%** | 97.8% |
| dense 27B | 48k (44k) | 25.0 | 58.3 | **+133.1%** | 94.9% |

### Strengthened accept-vs-depth curves, MoE (`a1b_curve.csv`, `a1c_256k.csv`)

**T1 — context-independent task (~99% accept), 6 depths to the native ceiling:**

| depth | 8k | 32k | 64k | 96k | 128k | **256k** |
|---|---:|---:|---:|---:|---:|---:|
| mtp t/s | 151 | 142 | 132 | 122 | 115 | **90** |
| edge (vs no-spec) | +75% | — | — | — | +134% | **+176%** (32.7→90.4) |
| accept | 99.17% | 99.17% | 99.17% | 99.17% | 99.17% | **99.17%** |

Accept **byte-identical at all six depths** (drafts are deterministic under greedy and context-independent, so a
dip could only be a cache/slot BUG) → **rules out the KV-slot-boundary acceptance-oscillation bug (#23658)** on
our base, across depths spanning ~125 2048-token boundaries. **256k runs and fits VRAM** (ncmoe=8, q4 KV, ub1024).

**T2 — realistic free-form reasoning (~50% accept, real headroom to fall):**

| depth | no-spec | mtp | edge | accept |
|---|---:|---:|---:|---:|
| 8k | 87.4 | 97.7 | +11.8% | 51.2% |
| 64k | 64.1 | 72.5 | +13.1% | 52.7% |
| 128k | 49.0 | 69.2 | +41.4% | 48.1% |

Even a low-accept task holds ±2pp with the edge still growing → **rules out the SWA/hybrid accept-collapse report
(#23322)** for our config, and shows the growth is not a near-ceiling-99%-task artifact.

## Mechanism (why the edge grows) — literature-backed

Three distinct O(S) cost terms dominate in sequence:
- **(A) target-forward cost** grows O(S) (our 10/40 full-attn layers + base): no-spec decode 85→48→33 (MoE 8k→
  128k→256k). MTP amortizes it over ~4 verified tokens/pass → the *relative* edge **widens with depth**. Textbook
  **Leviathan** (`IF=(1−α^{γ+1})/[(1−α)(γc+1)]`, c=t_draft/t_target: costlier target → lower c → higher speedup at
  fixed α) and empirically **MagicDec** (arXiv:2408.11049: 1.02×@4k→2.0×@32k — our curve's shape). **Dominates the
  whole range we can reach.**
- **(A′) the draft's OWN attention cost** also grows O(S) (nextn is full-attn) — this is what **Windowed-MTP**
  (arXiv:2607.21535, "…*at Million-Token Context*", NVIDIA single-author preprint, Jul-2026) removes. But its tax
  is +27% on the *draft phase* @261k rising to net-negative near 1M; the draft phase is ~1 of 41 layers, a tiny
  slice → at 256k it is **swamped by (A)** (we measured +176% edge, pristine accept). "Vanishes at short context
  by construction." Worst on hybrids — but we can't reach the regime where it wins.
- **(B) draft prediction quality over long-range deps** — small for *native jointly-trained* MTP heads (EAGLE-3.1
  accept flat 1K→32K; DeepSeek-V3 MTP 85–90% stable; "Hidden States Drift" 2604.26412: the mild decay is over
  draft *depth γ*, not context length). Our T2 accept-stability confirms B is negligible here.

## Corrections banked during the double-check (honest trail)

The verdict evolved across three commits — recorded so we don't re-lITigate:
1. **`9dc2edd`** — first pass: "premise reversed, CLOSED null." *Wrong framing on two points.*
2. **`e1631d1`** — double-check: "right paper, wrong regime; don't build at 128k, revisit ≥256k." Corrected: the
   doc's [ref-105] is a **real** paper (verified on arxiv, not fabricated — the doc's error was
   **regime-misattribution**, dropping the "million-token" scope); and **windowing the draft is LOSSLESS by
   construction** (the full-attn target verifies every token, so a window changes only which tokens are *proposed*,
   never *accepted*; draft top-1 unchanged 86–94% per the paper) — a cost-saver, **not** an accept-killer (my
   first-pass "can only lower accept" was wrong).
3. **`0c585a9`** — measured native 256k (reachable without YaRN: GGUF `context_length=262144`, fits VRAM): edge at
   its MAX (+176%), accept pristine → the tax never bites anywhere we can serve → **CUT** (supersedes "revisit
   ≥256k").

## Upstream / external verification

- **No PR/issue anywhere in ggml-org/llama.cpp proposes windowing the MTP/nextn draft attention** — clean negative
  (two in-repo GitHub searches, "0 results"). Adaptive-draft-length is only an **open, unanswered** discussion
  (#23738, which does note n=2/n=3 curves *converging* ~80–100k — about optimal draft *length*, tangential to our
  edge-vs-no-spec claim). So windowed-MTP is un-built AND un-rejected upstream.
- llama.cpp had May-2026 Qwen3.6 MTP accept bugs (#23322 SWA/hybrid low accept, #23658 slot-boundary oscillation,
  #23302 n-max≥3 determinism) — but our base **`720d7fa40` = 2026-07-25** postdates them, we're **not SWA**, and
  our accept curves are clean, so they don't apply.
- **Accept metric verified correct:** `tools/server/server-context.cpp` — `draft_n = n_draft_total`,
  `draft_n_accepted += ids.size()-1` (the −1 drops the target's always-correct bonus token) → the standard
  fraction-of-drafts-accepted. Deterministic under greedy (one value per (task,depth) is correct).
- **Verified papers (all resolve on arxiv):** 2211.17192 Leviathan · 2408.11049 MagicDec · 2503.01840 EAGLE-3 ·
  2412.19437 DeepSeek-V3 MTP · 2607.21535 Windowed-MTP · 2503.05096 AdaSpec · 2604.26412 Hidden-States-Drift ·
  2502.17421 LongSpec · 2505.20776 SpecExtend · 2605.01106 Component-Aware Hybrid · 2607.16673 SpecLA.

## Method / stats notes & limits

- Effect sizes (+12% to +176%) dwarf the <1% within-instance CV; the monotone, two-model, two-regime, six-depth
  signal is far outside any plausible variance. Reps reuse the cached prefix (they measure decode jitter, not
  reload variance) — acceptable given the effect magnitude; accept is deterministic (zero rep-variance).
- Not measured: the ~512k–1M regime (unreachable: native ceiling 262k; beyond needs YaRN past training where
  multi-hop quality is already gone — see CONTEXT_PLAN). That is the ONLY regime where the lever would pay, and
  it's off-limits by architecture and by quality → no gap remains that matters.

## Deploy consequence (positive, keep)

`--spec-type draft-mtp` is worth **MORE** at long context — **+134% @128k, +176% @256k (MoE), +41% even on
low-accept reasoning @128k** — exactly the regime the agentic deploy runs in. **Keep it ON.**

## Re-open trigger

Only if a future model ships a **≥512k _native_ window** (no YaRN) **with a real use case at that length**. Then
window the nextn head (StreamingLLM window+sink — lossless), and re-run the gate:
`MODELSET=… TASK=… DEPTHS=… UBATCH=… ops/a1_mtp_depth_bench.py`.
