# Qwen3.8 imatrix and Cold Fusion candidate assessment — 2026-08-21

**Status:** assessed / experiment proposed / not yet executed

**Scope:** `tare.tools.local-labs`, Qwen3.8-27B GGUF artifacts, `slop.cpp` runtime qualification

**Hardware target:** RTX 3090 24 GB, 64 GB host RAM, WSL2

**Operational decision:** preserve the identity of already-qualified artifacts as historical evidence, not
their operational primacy. Admit the current Unsloth revision and Cold Fusion as revision-pinned candidates;
if stronger evidence wins, supersede the old quant, experiment conclusion, or operating rule explicitly.

## 1. Why this assessment exists

Two external artifacts were proposed after the IQ4_XS endpoint requalification and the broader local
inference portfolio:

1. Unsloth's Qwen3.8 importance matrix:
   `https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/imatrix_unsloth.gguf`
2. DavidAU's Cold Fusion GGUF repository:
   `https://huggingface.co/DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-MTP-GGUF`

They answer different questions. The Unsloth file is a quantization input, not a runnable model. Cold
Fusion changes the trained weights and behavior, so it is a new model candidate rather than another
quantization arm of the Qwen3.8 base.

The assessment also found material upstream artifact drift: the current Unsloth files differ in size and
content identity from the files already present in the lab. Historical results therefore remain bound to
their original local hashes and cannot be transferred automatically to the current Hub revision. This
binding is a provenance rule, not a veto on evolution: the new revision may supersede the historical Pareto
frontier as soon as it passes a stronger or equally qualified comparison.

## 1.1 Supersession posture

The lab is evidence-seeking, not artifact-preserving. Models, quantization recipes, experiment designs,
promotion rules, and working assumptions are all provisional. A new result should supersede an old one when
it has better controls, stronger measurement, broader relevant coverage, or a clearly better operational
trade-off.

Supersession means:

1. preserve the old artifact hash, receipts, and conclusion as historical evidence;
2. identify the exact new candidate and the evidence that defeats the old conclusion;
3. mark the old decision `SUPERSEDED`, not `WRONG` or silently deleted;
4. promote the new operational default when its declared gates pass;
5. reopen even the new default when later evidence challenges it.

A hash change is neither positive nor negative evidence by itself. It only prevents accidental pooling.
The purpose of version pinning is to make a genuine improvement attributable and reproducible.

## 2. Frozen external identities

The following identities were read from the Hugging Face model API and, for the imatrix, verified by a
local metadata-only inspection. They must be rechecked immediately before any download because Hub `main`
is mutable.

| Artifact | Revision / SHA-256 | Bytes | Classification |
|---|---|---:|---|
| Unsloth repository | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | — | mutable upstream revision |
| `imatrix_unsloth.gguf` | `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1` | 13,642,656 | quantization importance matrix |
| Current Unsloth `UD-IQ4_XS` | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` | 14,252,845,984 | base-model quant candidate |
| Current Unsloth `UD-Q2_K_XL` | `fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0` | 9,828,981,664 | base-model quant candidate |
| Cold Fusion GGUF repository | `21dd13a4a43d7570a9496948f6265310681fa9f4` | — | mutable upstream revision |
| Cold Fusion BF16 source repository | `9c44193f07782c85c0f437a5d8466ba5c95c95fe` | — | fine-tuned source identity |
| Cold Fusion regular `NEO-IQ4_XS` | `d1a8f3595955e132b3f60da5fb4e138d0f1ba11b1d04729f4ba74e8b4eb2852e` | 16,582,359,616 | tuned-model quant candidate |
| Cold Fusion `NEO-MTP-IQ4_XS` | `523bf4fbe2a2e0ce7aa54f812d85746294483b579443dd6e50e8ab684d7852f9` | 17,033,680,384 | recommended first Cold Fusion artifact |
| Cold Fusion `NEO-LOW-MTP-IQ4_XS` | `b5d3c2abbccf2bacf9dcbbe8b0a192a34dae01d91e8a0f527e5a309349e32f2c` | 15,309,039,104 | deferred: naming/card semantics need inspection |

The already-qualified local base IQ4_XS remains:

- path: `/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-IQ4_XS.gguf`;
- bytes: `15,705,861,088`;
- SHA-256: `9fd40d7036f5e0918e20aaeebf11468fafd06bb53d4d980eef6bb7e4e4ace666`;
- embedded MTP head: previously verified present;
- qualification substrate: llama.cpp `5e7f6271c06b9104862ab799278a1b7f1323a449` (`b9863`).

Do not replace this file with a same-label download. New artifacts belong in a revision-qualified directory.

## 3. Unsloth imatrix: direct findings

The file is a valid GGUF v3 object whose `general.type` is `imatrix`. Direct inspection found:

| Field | Value |
|---|---:|
| `imatrix.datasets` | `unsloth_calibration_dataset` |
| `imatrix.chunk_count` | 1,251 |
| `imatrix.chunk_size` | 8,192 |
| GGUF tensor records | 992 |
| explicit tensor names containing `nextn` | 0 |

The declared chunk geometry represents approximately 10.25 million token-slots of calibration. The file
contains per-weight input-importance accumulators and counts for the base model, but its embedded metadata
does not disclose the composition of `unsloth_calibration_dataset`.

### 3.1 What it can and cannot do

It can be passed to `llama-quantize --imatrix` while quantizing a BF16/F16 source. It can influence which
weight columns retain more precision in supported mixed quant types.

It is **not**:

- a model that can be served;
- a LoRA or runtime adapter;
- a patch that improves an already-quantized GGUF;
- evidence that a produced quant preserves agent, context, or tool-calling quality;
- an importance profile for the MTP `nextn` head.

Requantizing an existing low-bit file is not the qualified path. A controlled custom quant must start from
the frozen BF16/F16 source. Because the supplied imatrix contains no `nextn` records, any integrated MTP
head should remain at an explicitly high precision such as Q8/F16 or be distributed separately.

### 3.2 Upstream drift and its consequence

The local IQ4_XS is 15,705,861,088 bytes while the current Hub `UD-IQ4_XS` is 14,252,845,984 bytes. The
local Q2_K_XL is 10,676,423,744 bytes while the current Hub file is 9,828,981,664 bytes. The current Hub
also publishes an MTP head under a separate `MTP/` path, while the historical local quants were verified
with an embedded head. The repository commit history also records removal of the earlier `UD-IQ2_M` file.

Therefore:

1. the old seven-quant frontier remains valid only for the frozen historical artifacts;
2. the current Unsloth Dynamic release is a new artifact generation;
3. `Q2_K_XL is the Pareto floor` is a hypothesis to requalify, not a fact to copy to the new hashes;
4. current downloads must never overwrite the old paths;
5. custom quantization should be considered only after testing the ready-made current-revision quants;
6. if a current-revision requant wins, it should supersede the old deployment candidate without waiting for
   artificial continuity with the historical file;
7. if the new imatrix changes the observed quantization frontier, the old frontier and any derived dogma
   should be relabeled `SUPERSEDED` while their receipts remain intact.

## 4. Cold Fusion: evidence posture

Cold Fusion is a fine-tuned Qwen3.8-27B. Its model card claims:

- median reasoning-token reduction around two thirds, with some cases between one half and one tenth of
  the base reasoning length;
- improved instruction following and problem solving;
- support for `xhigh`, `medium`, and `low` reasoning modes;
- NEO-imatrix quants with a high-precision output tensor;
- MTP variants whose MTP tensors remain Q8;
- improved results over the base model on several classic language-model benchmarks.

These claims make the model relevant because the lab independently observed a concrete base-model failure:
Qwen3.8 `xhigh` thinking was verbose, truncation-prone, and worse than instruct mode on the coding curve.
Cold Fusion proposes to modify exactly that behavior.

The public evidence is not sufficient for promotion. The card does not provide a local-lab-compatible
packet with exact harness commit, per-sample receipts, complete seeds, fixed template identity, confidence
intervals, or a matched GGUF comparison using the same quantization recipe. Claims such as `2–4% imatrix
improvement`, `99% of 8-bit performance`, and the reported reasoning reductions are hypotheses for this
lab, not accepted measurements.

**Classification:** promising external candidate; provenance identifiable; behavioral and quantization
claims unqualified locally.

## 5. Recommended first Cold Fusion artifact

Use only the revision-pinned `NEO-MTP-IQ4_XS` in the first campaign:

`Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-NEO-MTP-IQ4_XS.gguf`

Reasons:

- IQ4_XS is close to the already-qualified base substrate;
- the file fits the 24 GB target with more headroom than Q6/Q8;
- the embedded MTP head allows a same-file MTP-off versus MTP-on comparison;
- loading the same file in both MTP arms avoids treating head-presence or file layout as the MTP factor;
- starting at IQ2_M would confound the fine-tune with an already-suspect aggressive quant regime.

Defer `LOW-MTP` until a tensor inventory reconciles its name with the card statement that LOW variants omit
some MTP/output-tensor modifications. Do not infer behavior from filenames.

## 6. Proposed experiment program

### E0 — artifact admission and static inspection

Before the first server restart:

1. download by immutable revision into a new directory;
2. verify bytes and SHA-256 against the frozen API identity;
3. record GGUF architecture, tensor types, MTP/`nextn` inventory, tokenizer, chat template, context metadata,
   output tensor type, quantizer metadata, and mmproj relationship;
4. reject or hold on any unexplained tensor or template difference;
5. retain the existing IQ4_XS and its service argv unchanged as the restoration baseline.

### E1 — current Unsloth revision screen

Arms:

1. historical local IQ4_XS;
2. current-revision Unsloth `UD-IQ4_XS`;
3. historical local `UD-Q2_K_XL`;
4. current-revision Unsloth `UD-Q2_K_XL`.

Run the compact, high-discrimination packet first:

- agent suite plus tool perturbations;
- cache/cancel/reuse correctness;
- paired context retrieval, multikey, multihop, and aggregation at 8k/16k/32k/64k;
- MBPP+ failure-focused subset, including `Mbpp/260`;
- strict GSM8K failure replay;
- MTP acceptance and throughput sentinel if the artifact carries or is paired with a head.

Promote only the non-inferior candidates to the full MBPP+ and wider context matrices.

### E2 — base versus Cold Fusion practical candidate A/B

Use a 2 × 2 design on the same engine, KV type, context, sampling condition, and request order:

| Weights/artifact | MTP off | MTP on |
|---|---:|---:|
| qualified base IQ4_XS | yes | `n-max=3` |
| Cold Fusion `NEO-MTP-IQ4_XS` | yes | `n-max=2`, then `n-max=3` |

Run two sampling tracks:

1. deterministic qualification (`temperature=0`) for correctness and divergence attribution;
2. model-recommended sampling for descriptive human-facing behavior, never pooled with track 1.

The first A/B is an end-product comparison, not a causal estimate of the fine-tune. Weights, imatrix,
output-tensor treatment, converter revision, embedded template, and MTP layout can all differ.

### E3 — reasoning-efficiency falsification

This is the primary Cold Fusion claim and the highest-value experiment.

For base and Cold Fusion, run instruct/off, low, medium, and xhigh with enough output budget to distinguish
true reasoning efficiency from token starvation. Measure:

- strict task success;
- reasoning tokens, final-answer tokens, and total tokens;
- truncation and non-termination rate;
- correctness per 1,000 generated tokens;
- time and joules per correct answer;
- self-correction that changes a correct answer into an incorrect one;
- tool-call validity and irreversible-action safety;
- repeated hesitation/loop markers as descriptive diagnostics, not standalone quality scores.

The claim is falsified if shorter reasoning comes with lower correctness, more tool errors, hidden
truncation, or reduced long-context retention.

### E4 — context, cache, and agent behavior

Cold Fusion and the new Unsloth quants must pass the risks that classic benchmarks do not cover:

- retrieval, multikey, multihop, and aggregation at 8k/16k/32k/64k, then 128k if qualified;
- information placed near the beginning, middle, and end;
- many similar facts, not only a single distinctive needle;
- cold versus warm prefix, context checkpoint reuse, cancel then reuse, and session switching;
- 8-tool functional suite followed by rephrase/reorder/rename/irrelevant-tool perturbations;
- MTP acceptance, functional success, TPOT, energy, and concurrency interaction.

MTP acceptance below 50% is only a warning from the model card. The lab decision must use measured
end-to-end latency, energy, and task correctness; acceptance alone is not a promotion metric.

### E5 — matched-requant causal follow-up

Only if Cold Fusion wins E2–E4, produce a causal quantization comparison from frozen BF16 sources:

- same converter commit;
- same quant target;
- same calibration corpus and imatrix-generation procedure;
- same explicit output-tensor and MTP-head precision;
- same chat template and metadata policy;
- base and Cold Fusion quantized independently but by the identical recipe.

This stage separates the effect of the fine-tuned weights from DavidAU's NEO imatrix and tensor-preservation
choices. It is expensive and is not required to answer the practical question `which final artifact is best
for this box?`; it is required before attributing a measured gain specifically to Cold Fusion training.

## 7. Proposed decision rules

Freeze exact numeric non-inferiority margins in a preregistration after the pilot estimates variance. The
qualitative gates are already binding:

1. no critical agent/tool regression;
2. no blind retry of irreversible operations;
3. cache cold/warm oracle equality and no cross-session contamination;
4. no increase in truncation or non-termination;
5. no context cliff earlier than the qualified base at the same usable memory envelope;
6. claimed reasoning-token reduction must coexist with non-inferior strict correctness;
7. MTP promotion requires an end-to-end latency or energy win, not merely a reported acceptance rate;
8. any result remains bound to model SHA, Hub revision, engine commit, template SHA, quant recipe, and runtime
   lever vector;
9. an admitted candidate that passes stronger gates and improves the relevant Pareto trade-off supersedes the
   prior candidate; prior promotion is not a tie-breaker in its favor.

## 8. Explicit non-actions

- Do not overwrite the already-qualified local Qwen3.8 files.
- Do not treat preservation of an old artifact as preservation of its deployment status.
- Do not treat hash or size drift as a defect; treat it as a new experimental identity.
- Do not apply an imatrix to an already-quantized GGUF and call the result qualified.
- Do not begin with Cold Fusion IQ2_M.
- Do not treat the model-card benchmarks as local evidence.
- Do not attribute a practical base-versus-Cold win solely to the fine-tune.
- Do not download both regular and MTP Cold Fusion files solely for the MTP A/B; the MTP file can be run with
  speculation disabled and enabled.
- Do not change deployment defaults before the full candidate packet closes.
- Do not restart a soak campaign as part of this work.

## 9. Recommended execution order

1. Create a controlled LAB maintenance mode that records and restores the exact current service argv.
2. Admit the current Unsloth IQ4_XS and Q2_K_XL under revision-qualified paths.
3. Run E1 and decide whether the current Q2_K_XL remains, improves, or supersedes the historical quant Pareto
   candidate; promote the winner rather than privileging continuity.
4. Admit only Cold Fusion `NEO-MTP-IQ4_XS` and run E0.
5. Run E2–E4 with MTP off/on and the new agent/context/termination packet.
6. Run E5 only if the end-product candidate wins and causal attribution remains valuable.
7. Restore the original server configuration and verify health after every maintenance tranche.

## 10. Evidence boundary

This document records a repository/API inspection and an experiment design. It does not claim that the
current Unsloth quants, the supplied imatrix, NEO imatrix, Cold Fusion weights, or Cold Fusion MTP variants
have passed local qualification. At the time of writing, the historical IQ4_XS identified above is the only
locally qualified Qwen3.8 artifact in this assessment. That status is explicitly temporary: it carries no
presumption against a newer requant, model, method, or result that earns supersession through evidence.
