# LAB-CTX-003 — LongBench RepoBench-P full baseline

**Frozen before inference:** 2026-08-22 10:55 America/Sao_Paulo  
**Question:** can the incumbent use retrieved cross-file repository context for next-line code completion?

## Frozen substrate and data

- LongBench official repository commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`;
- Hugging Face dataset revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`;
- `data.zip` SHA-256 `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`;
- `repobench-p.jsonl` SHA-256 `919a4439e2a84ebb25bacc39ac3b3269a7641af6e02ae205ed78d8c53dfe3568`;
- 500/500 unique examples, Python and Java, one reference per example;
- reported source length min/median/max 810/3,503.5/18,754;
- model SHA-256 `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`;
- server slop.cpp `b9863-5e7f6271c`, context 131,072, q4 KV, MTP n3, one RTX 3090.

## Frozen protocol

- exact official prompt: `Please complete the code given below.\n{context}{input}Next line of code:`;
- raw `/completion`, matching LongBench's explicit decision to omit chat wrappers for RepoBench-P;
- deterministic generation: 64-token official budget, temperature 0, top-k 1, top-p 1, seed 42,
  cache disabled;
- exact live tokenizer preflight; no prompt may exceed the server context;
- official `code_sim_score`: first generated non-comment/non-fence line, FuzzyWuzzy ratio against the reference;
- official mean edit similarity ×100 is primary; exact normalized first-line match and length strata are auxiliary;
- first run indices `0, 25, ..., 475` (20 examples). Expand resumably to all 500 only if outputs are
  nonempty, scoring imports pass, and there is no endpoint/context failure.

## Gates

1. Dataset and output cardinality must be exact and IDs unique.
2. The full run is a useful baseline if official edit similarity is at least 55.0, approximately the strongest
   published model in the original LongBench table (ChatGLM3-6B-32k: 54.76), and at least 99% of raw outputs
   are nonempty. This historical comparison is contextual, not a same-model claim.
3. A pilot below the threshold does not stop the full run unless it reveals an infrastructure or prompt-mode
   error; the full denominator is cheap enough and needed to avoid selection noise.
4. Preserve every raw completion and do not post-process before official scoring.

## Prompt-mode amendment — frozen after raw smoke, before chat inference

The raw 20-example smoke scored 1.90 and produced two empty EOS completions. The other outputs mostly began
with meta-explanations such as “Based on the provided code,” rather than a next line. This is a systematic
prompt-mode mismatch for this instruct model, not a context failure. Stop the raw arm at n=20 and preserve it.

Run the same 20 IDs through the model-native chat template with thinking disabled and the same user prompt,
sampling, and 64-token budget. Expand the chat arm to all 500 only if it improves official code similarity by
at least 10 percentage points over raw and produces at least 19/20 nonempty outputs. The chat result must be
labeled **official-data/model-native-template**, not directly comparable to the stock raw LongBench table.
