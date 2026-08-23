# QWEN38-2026-08-20 — requalification report

**Decision:** PASS for the three declared measurement questions. This is grade-2,
single-substrate evidence, not a model promotion and not evidence that the same
capabilities survive a different engine, quantization, template, or KV format.

## Frozen substrate

- model: `Qwen3.8-27B-IQ4_XS.gguf`
- GGUF SHA-256: `9fd40d7036f5e0918e20aaeebf11468fafd06bb53d4d980eef6bb7e4e4ace666`
- llama.cpp build: `b9863-5e7f6271c`
- live context: 32,768 tokens; four slots; Q8_0 K/V cache; no speculative decoder
- chat-template SHA-256: `d1f22a89eac3609dcfaa7b471b1f7d23bee2f084d275d26f4f8231d1d7908f4e`
- runner source SHA-256: `c23e67e2983033d4755614d7170d17dd61060f84f9aaebeff52734bf8bc10669`
- result-bearing Git HEAD: `153150acabab292bb0feb7398e93be615950b754`

All three manifests report the same identity before and after their run. Prompt
cache was disabled, the response path was greedy, and every raw completion is
preserved in an incremental JSONL receipt.

## Results against pre-registered criteria

| Question | Result | Wilson 95% | Other hard criteria | Decision |
|---|---:|---:|---|---|
| MQAR strict recall | 240/240 (100%) | 98.42–100% overall | 40/40 at every dose; 100% format | PASS |
| NIAH positive retrieval | 60/60 (100%) | 93.98–100% | 10/10 negative controls; no repeated cell failure | PASS |
| GSM8K strict accuracy | 95/100 (95%) | 88.82–97.85% | 99% format; 1% truncation | PASS |

### MQAR

The valid restarted matrix spans `P={4,32,128,512,1024,1792}`, five depths,
and eight replicates per cell block. Actual rendered prompt sizes were 268,
716, 2,252, 8,396, 16,588, and 28,876 tokens. Every complete response was the
expected fixed-width value and nothing else. At each dose, 40/40 gives a Wilson
lower bound of 91.24%, clearing the declared 90% high-pressure threshold.

The original terminal dose `P=2048` rendered to 32,972 tokens, above the live
32,768-token context. The runner stopped before sending that request. Its five
earlier rows are retained as an invalid instrumentation attempt and were not
pooled. Amendment A1 was committed before the clean restart at `P=1792`.

### NIAH

The calibrated prompts ranged from 4,064 to 29,999 actual rendered tokens.
Every one of the 60 positive probes returned the exact random code, and all ten
negative controls returned exact `NOT_PRESENT`. Each target had three seeds at
depths 0.10, 0.50, 0.75, and 0.90.

The preliminary failure at nominal 24k/depth 0.75 does not reproduce. That old
probe actually rendered only about 18k tokens, used one sample per cell, and had
no negative controls. It is superseded as a defect signal by this calibrated
matrix; it remains useful as the reason the stronger test was run.

### GSM8K

The sample is a deterministic shuffle of 100 rows from the 1,319-row test
export (`seed=20260820`). Dataset file SHA-256 is
`68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`;
score-bearing logical-content SHA-256 is
`ae9f014e00b29e8cdbeb65d2b8d37fc4fb42d70dd691052e682c65f2f65dedf2`.
Strict and lenient diagnostic scores agree on all 100 rows.

The five strict failures separate into two mechanisms:

| Task | Gold | Output | Classification |
|---|---:|---:|---|
| `gsm8k/584` | 2 | 4 | divided the price of two boxes by pairs in only one box |
| `gsm8k/1019` | 98 | 113 | treated late checkout as never leaving |
| `gsm8k/241` | 6 | 3 | counted distinct books instead of reader-book events |
| `gsm8k/1312` | 180 | 30 | omitted the six reservations multiplier |
| `gsm8k/153` | 48 | no final line | derived 48 correctly, then truncated during verification at 512 tokens |

The historical 29/30 first-row scout is compatible with this result but is no
longer the decision artifact: it was smaller, non-random, and used a permissive
last-number scorer. The current result meets the declared lower-bound, format,
and truncation thresholds.

## Qualification and remaining limits

The benchmark scorer self-test suite passes 20/20, including adversarial cases
for substring/wrapper false positives, GSM8K final-line enforcement, answer-hash
invalidation, and interval calculations.

These results establish exact retrieval and arithmetic competence on the stated
fixtures only. MQAR values are synthetic fixed-width codes; NIAH is a single
needle family; GSM8K has only 100 sampled rows. The next confirmation step for
any deployment claim is a frozen second substrate or an independently repeated
run. The next research experiment must be pre-registered separately and must not
reuse these rows as tuning feedback.
