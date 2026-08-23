# LAB-CTX-002 — Official RULERv1 64k/128k pilot

**Frozen before generation/inference:** 2026-08-22 09:10 America/Sao_Paulo  
**Question:** does the historical Qwen3.8-27B Q4_K_XL retain broad official RULERv1 capability at 64k and 128k on one RTX 3090?

## Frozen substrate

- NVIDIA/RULER `rulerv1-ns`: `e8bbff677ca2c239640dc90f93310dcf32408c93`;
- NVIDIA-NeMo/Skills `chsieh/ruler-remove-prefix`: `f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3`;
- official RULER generator/config snapshot used by NeMo Skills: `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3`;
- model SHA-256: `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`;
- server: slop.cpp `b9863-5e7f6271c`, context 131,072, q4 KV, MTP n3, parallel 1;
- model tokenizer: `Qwen/Qwen3.5-27B` tokenizer.json SHA-256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.

## Corpus identity

- Paul Graham essays SHA-256 `8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd`;
- SQuAD v2 dev SHA-256 `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`;
- HotpotQA distractor validation comes from the Hugging Face dataset mirror revision
  `1908d6afbbead072334abe2965f91bd2709910ab`, parquet SHA-256
  `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`, because the upstream
  CMU HTTP host did not respond. Conversion changes container shape only, not fields/content.

## Frozen pilot

- lengths: 65,536 and 131,072 tokens;
- all 13 official RULERv1 tasks, one sample per task/length, seed 42 (26 cells);
- official prompts, answer prefixes, per-task output budgets, and `string_match_all`/`string_match_part` semantics;
- deterministic inference: temperature 0, top-p 1, thinking disabled, no prompt cache;
- NeMo Skills' default 50-token chat-template reserve, followed by exact `/apply-template` + `/tokenize`
  preflight against the live GGUF. Any cell exceeding 131,072 is an infrastructure failure and is not sent.

## Gates

1. Tokenizer parity must be exact on the fixed smoke set before dataset generation.
2. Generation must produce all 26 unique cells and every live prompt must fit the server context.
3. A length is a provisional broad-context pass only if its 13-task macro score is at least 85.6%, the
   original RULER qualitative threshold. With n=1/task this remains a pilot, not a publication-grade score.
4. Infrastructure/model failures stay separate. No blind retries; a reproducibility rerun is labeled explicitly.
5. Expand beyond the pilot only if it is discriminating and resource-proportionate.

## Gate-triggered replication amendment — frozen before replication

The completed pilot scored 82.82% at 64k and 100% at 128k. The three non-perfect 64k tasks were VT
(0%), CWE (10%), and FWE (66.7%), while the corresponding 128k cells were all 100%. This strong
non-monotonic reversal triggers a bounded replication of exactly those three tasks:

- regenerate the same official datasets with `num_samples=3`, seed 42, at both lengths;
- verify that sample index 0 has the same prompt hash as the original pilot;
- reuse index 0 and infer only indices 1 and 2, adding 12 independent cells;
- report n=3 task means and the 3-task macro by length; do not generalize this replication to the other
  ten tasks or to the full 100/500-sample official protocol.
