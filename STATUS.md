# STATUS — what this project has settled, and what it has not

Referenced from `collectors/host.py` and implied across the experiment scripts; this is
its actual home. Every number below was produced by `python analyze_ab.py` over the
`runs/ab-*/records.json` directories, paired by `(round, ncmoe)`, using the project's own
robust machinery (exact sign test, seeded percentile bootstrap, Cliff's delta). Medians,
never means — this host produces cold-start outliers and a mean was trusted once already.

Regenerate everything here with:

    python analyze_ab.py            # human report, straight off disk
    python analyze_ab.py --json     # provenance blob
    python -m model_lifecycle.cli report -o reports/EVIDENCE.md   # same numbers, from the Store

The number tables below are **authored prose citing regenerable evidence**. That evidence —
the noise floor and every paired comparison — is regenerated into [`reports/EVIDENCE.md`](reports/EVIDENCE.md)
by `model_lifecycle.reports.status`, reading the A/B records out of the Store (where the
backfill unified them). The disk reader (`analyze_ab.py`) and the Store reader share one
pairing core and produce byte-identical numbers, so this prose can be checked against the
data at any time and can never again quietly drift from it.

---

## The instrument's resolution: the noise floor

From `ab-null-qwen36-35b` — the same binary and the same environment in both arms, so the
true delta is **zero by construction**. This is the single most important number in the
project, because every published delta is read against it.

| quantity | value | reading |
|---|---|---|
| null prefill Δ, median | **+0.29%** | the pair is unbiased |
| null prefill Δ, median \|%\| | **~2.3%** | round-to-round paired scatter |
| null order effect (sign_p) | 1.0000 | interleaving IS balanced |

Two consequences, both load-bearing:

1. **A prefill delta below ~2.3% is not evidence, however tidy its median.** The floor is
   ~2.3%, NOT the sub-1% previously assumed offhand.
2. `sign_p = 1.0` on the null means arm order does not bias the pair. This validates the
   interleaving for *every* paired comparison below — without it, they would all inherit a
   systematic bias invisible to replication.

At n=6 the exact sign test bottoms out at **p = 0.0312** (every round agreeing). That is
the strongest claim a 6-round paired run can make distribution-free, and where you see it
below it means the effect was unanimous across rounds.

---

## SETTLED

### 1. Pinning (`GGML_CUDA_REGISTER_HOST=1`) roughly doubles prefill — in three geometries

`cudaHostRegister` on the model-loader's mmap is the fork's real value. Prefill delta of
`pin - base`, paired:

| model | experts | Δ prefill | sign_p | Cliff's δ |
|---|---|---|---|---|
| qwen36-35b | 256 | **+104.9%** | 0.031 | +1.00 |
| qwen3-30b | 128 | **+123.3%** | 0.031 | +1.00 |
| gpt-oss-20b | 32 | **+114.6%** | 0.031 | +1.00 |

Three independent geometries, unanimous every round, ~50× the noise floor. This is the
most solid result in the project. **Pinning is on by default in `serve.py`.**

### 2. Adding the prefetch to an already-pinned baseline COSTS prefill

The switches are not independent and there is no cheap middle. `pinpf - pin`, paired:

| model | experts | Δ prefill | sign_p |
|---|---|---|---|
| qwen36-35b | 256 | **−22.1%** | 0.031 |
| qwen3-30b | 128 | **−11.3%** | 0.031 |
| gpt-oss-20b | 32 | **−8.6%** | 0.031 |

The tax scales with expert count (256 > 128 > 32), unanimous every round, all above the
floor. **Prefetch is off by default in `serve.py`.**

### 3. The published "+58%" was one number hiding two effects

`pinpf - base` on qwen36-35b = **+60.2%** (sign_p 0.031) — the headline, reproduced. It
decomposes exactly: pinning (+105%) followed by prefetch (−22%). The fork's entire value
is the `cudaHostRegister` call; its headline overlap is a tax, not a feature.

### 4. Generation does not move — IN THE NEAR-RESIDENT REGIME

Every `gen_tps` comparison lands inside the noise floor. `pin - base` generation:
−0.15% / +0.24% / +0.04% across the three models (all sign_p ≥ 0.69); the dedicated
`genpin` 35B run gives −0.18% (sign_p 1.0). Pinning buys prefill and nothing else here.

> **SCOPE — do not overread.** All of this is the *near-resident* regime (35B at
> ncmoe=24 generates at ~43 t/s, most weight near the card). The question pinning was
> *supposed* to answer — does it help generation when generation is **transfer-bound** —
> is NOT settled by this data. See OPEN §B1.

### 5. The fork is worth its mechanism, and the rebase reproduces it exactly (n=18)

| comparison | Δ prefill | sign_p | reading |
|---|---|---|---|
| fork − base | +59.3% | <0.0001 | the fork earns its keep |
| rebased − base | +60.4% | <0.0001 | so does the patch on today's upstream |
| rebased − fork | +0.48% | 0.096 (within floor) | **rebase ≡ fork** |

(Cliff's δ is diluted here because this run spans ncmoe {8,24,40} and pools raw rates
across doses; the *paired* sign test and median% are the trustworthy read.) **Actionable:
the 3-week-stale fork can be dropped and the patch carried on current upstream with no
measurable loss** — 266 commits of upstream work for free.

### 6. `turbo-mma-decode` does nothing measurable — do not carry it

`turbo - base`, both models, both metrics, all within the floor: generation −0.04% (35B)
and +0.15% (30B), sign_p 1.0; prefill likewise. The fork author's fused-GQA decode path
is inert on this hardware. n=6 here could detect roughly a >5% effect; this is <0.4%.

### 7. Question 4 resolved: the L18 / A-B disagreement is NOT the build

The 2×2 (`ab-stack`, BUILD × PREFETCH):

| | prefetch off → on |
|---|---|
| **rebase build** | −22.1% (sign_p 0.031) |
| **stack build** | −23.3% (sign_p 0.031) |

The prefetch is a ~22% tax on *both* builds — essentially identical. Build alone
(prefetch off) is +0.65%, within the floor. So the stack's prefill changes do **not**
interact with the prefetch, and the L18's non-monotonic ordering was confounding (its
`cache` factor was inert by construction), not a build×prefetch effect.

---

## OPEN — as prominent as the answers, on purpose

- **§B1 — transfer-bound generation.** The one regime that would test whether pinning
  helps *generation* (Nemotron-120B at ncmoe=99, ~0.64 t/s, 22 experts/token over PCIe)
  returned **12/12 REJECTED** — `ram ~4-5 GB < 16384 MB`. Prime suspect: `.wslconfig`
  caps WSL at 44 GB of 63.8, leaving ~19.8 GB for Windows against a 16 GB reserve, so the
  guard rejects the moment WSL claims its cap. This is a `.wslconfig` × envelope conflict,
  not a property of the model. **Unresolved — B1 is unanswered.**

- **The −10.4% no-mmap residual.** One of the three historically disputed deltas. Not
  covered by any `ab-*` directory here; **still open**, needs its own clean paired A/B.

- **`stackpf − prefetch` = −1.57%, sign_p 0.031 but below the floor.** Sign-consistent
  across all 6 rounds yet sub-2.3% in magnitude — a candidate real-but-tiny build effect
  when prefetch is on. **Not promotable at n=6**; would need more rounds to separate from
  noise. Recorded, not claimed.

- **The quality axis is dark.** `quality_bench` starved on 33/40 HumanEval+ problems (the
  thinking model spends the 1024-token budget reasoning before emitting code). No pass@1
  is measurable until the budget/thinking issue is fixed.

- **SGLang head-to-head on gpt-oss-20b.** New (2026-07-31). The only model both engines
  can serve on this box; llama.cpp numbers exist (base 189 → pin 406 t/s prefill). Awaits
  the SGLang setup now building in WSL.

---

## Stale markers found while writing this (fix in place)

- `config/environment.yaml`: `transient_handling: UNRESOLVED` predates `guard.py`, which
  now implements sustained-breach over K samples plus a load-phase exemption. The marker
  lags the code.
- `collectors/host.py:45`: "There is no Python on the Windows side of this desktop" — there
  is, 3.12.10. The `.exe` collector suffix is still correct for a different reason (it
  measures the Windows host, and the suffix also works when run natively), but the stated
  premise is false.

_Last regenerated: 2026-07-31 from `analyze_ab.py`._
