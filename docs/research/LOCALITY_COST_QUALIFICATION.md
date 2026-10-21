# EXP-3: locality, model loading and total-cost coverage

Status: **descriptive pilot completed**, evidence grade **C**, observed 2026-10-20/21.
The [bounded result](LOCALITY_COST_QUALIFICATION.json) and
[`locality_study.py`](../../tools/analysis/locality_study.py) belong to Local Labs.
This closes the operator's measurement slice, not the historical proposal for
automatic placement, zero local cost or a 70% cloud-cost reduction.

Six real public TUI direct sessions read the same frozen analyzer source for two
source-review questions. Qwen38-GSQ on the existing RTX 3090 gateway and the
existing GPT-6.1 Sol native Responses subscription used identical per-case requests,
read-only roles and four-call/240-second ceilings with zero automatic retries.
Each trial used three generation calls: two nonoverlapping file pages and one
answer. There were **18 measured calls**, plus **13 separate implementation/repair
calls** in one preserved session. Browsing and source review did not invoke models.
No model was downloaded, cache reset, service restarted or placement rule changed.

## Observed result

| Trial | Residency | Start to terminal (s) | Native active (s) | Load (s, included) | Input / output tokens | Cache read | GPU gross (Wh) | Words / 120 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| local-A-first | cold_load | 26.61 | 25.52 | 12.11 | 16749 / 777 | 8881 | 1.464 | 133 |
| remote-A-first | unknown | 13.68 | 11.96 | — | 13100 / 303 | 0 | Unknown | 104 |
| local-A-repeat | warm | 13.10 | 11.98 | — | 16749 / 939 | 13567 | 0.938 | 176 |
| remote-A-repeat | unknown | 12.05 | 10.94 | — | 13100 / 302 | 4480 | Unknown | 103 |
| local-B | warm | 14.07 | 13.31 | — | 16765 / 889 | 8891 | 1.133 | 159 |
| remote-B | unknown | 16.13 | 15.04 | — | 13121 / 361 | 1920 | Unknown | 139 |

The initial gateway had no resident backend and no held/queued GPU work; loading
took **12.11 s**. The same healthy PID/model remained for both local warm trials.
Cold here means an empty model backend; disk/page-cache coldness was not observed.
The switch time is **already inside** request/native wall time and is not added again.
Start-to-terminal includes public dispatch/preparation and observer latency; native
active time excludes operator browsing and review. The cloud's physical residency
and first-content timing were unobserved and remain unknown.

Local warm A was close to cloud A; local B was faster in this one observation.
Models, reasoning configurations and tokenizers differ, and two tasks establish
no statistically supported winner or general coding/Work quality. All six source
diagnoses were checked against the cited code. Only **2/6** met the requested
120-word target: **0/3 local, 2/3 cloud**. The local repeat duplicated its substantive
answer across prose and delivery summary, which the adapter intentionally preserves.
Keep these efficiency limits; do not report a strict all-pass benchmark.

Prompt-cache hits are separate from residency. Native provider counters record
the values above, including zero on the first cloud trial. Keys/history were kept;
no hidden warmup, synthetic prefix padding or keepalive request was used.

## Cost and decision

Sampled local request-window GPU gross energy was **3.535 Wh** in total. Across
the **577.46 s** campaign window, including review/browsing/remote intervals while
the local model remained resident, GPU gross energy was **7.252 Wh**. These are
approximately 1 Hz power samples integrated at client receipt times, not exclusive
task energy or a calibrated whole-host meter. Cloud compute energy is unknown.
The 594-line raw trace is retained privately by SHA-256.

**Total monetary cost is unknown for both routes.** No electricity tariff,
whole-host energy, hardware allocation or subscription allocation was supplied.
An empty known-cost map and unpriced calls mean unknown, not zero. Tokens, cached
reads, implementation overhead and idle GPU energy have destinations in the result;
none is silently discarded to manufacture a local-cost advantage.

Decision: keep current explicit routing. A cold local request pays measurable
startup; a resident local route is a useful candidate for these source reviews.
This pilot does not justify expanding the scheduler, promoting a model, replacing
frontier implementation/audit, or starting a paid broad benchmark.

## Repairs and evidence

The initial offline tests passed but used ISO invocation timestamps. Actual native
receipts use numeric epochs; this made active wall time disappear. The implementer
repaired reader and tests through the **same public TUI session**, keeping its
profile, journal, deadline and limits. A retained numeric receipt was replayed
offline and **60 focused tests passed on Windows**. CI runs the portable suite.

The private pilot screen first rejected valid paragraph-local citations and the
synonym "gaps/complete=false". The failed screens and word overruns are preserved.
Their remedy was **offline rescoring of retained answers**, followed by exact
answer/source-hash-bound maintainer review, with **zero inference reruns**. Hard
execution/source/limit failures still stop the pilot. The original preregistered
all-checks gate was not satisfied; this is the disclosed descriptive outcome.

The analyzer reads existing receipts only, hashes consumed files, preserves failed
rows, separates active wall/generation/switch measurements, keeps missing values
unknown, and refuses identity/duplicate-session/byte-limit errors. It does not
execute external assessment claims or certify their independence. Only direct,
uncompacted sessions were measured; this is not total multi-session Work accounting.

## Reuse and retention

```powershell
python tools/analysis/locality_study.py C:\evidence\study.json --output C:\evidence\new-report.json
```

Input shape, optional fleet/quality/energy fields and limits are in the module
docstring; synthetic examples are in [the focused tests](../../tests/test_locality_study.py).
Replay is offline and output creation is exclusive. Failed or missing costs stay
visible; do not respend inference to recompute a report.

Small maintained result: this document plus its JSON sibling, linked from the OS
backlog. Private journals, power samples and full VT stay under the operator's
64 MiB experiment ceiling, with a retention review on **2026-11-20**. The 4 MiB
per-file/32 MiB aggregate read limits bound the analyzer. Neither source maintenance
nor installation changes the operator's model fleet, profiles or resident cache.
