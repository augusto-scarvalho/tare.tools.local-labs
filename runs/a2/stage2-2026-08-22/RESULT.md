# A2 Stage-2 E1/E2 result

Date: 2026-08-23

Verdict: **G0 KILL / NO ELIGIBLE REFUSAL DIRECTION / DO NOT EDIT WEIGHTS**

## What ran

- Exact resident Qwen3.6 dense-27B base in NF4/BF16 compute on the RTX 3090.
- Frozen pools: 128 harmful + 128 harmless train; disjoint 32 harmful + 32 harmless validation.
- Residual activations captured at all 65 positions (embeddings plus 64 layers), hidden size 5,120.
- Difference-of-means directions scored at every pre-registered candidate layer 8--51.
- Per layer: harmful bypass under residual ablation, harmless refusal induction at alpha 8, and
  harmless next-token `KL(baseline || ablated)`.

The first calibration accidentally measured the model's open `<think>` block within the 32-token
budget. It was stopped before layer scoring and is retained as invalid evidence. The official
tokenizer option `enable_thinking=False` was frozen in `METHODOLOGY_ADDENDUM.md`; activations and the
complete selection sweep were then rerun at the answer-channel position.

## Results

- Corrected baseline refusal: harmful **25/32 = 0.78125**; harmless **0/32 = 0.00**.
- Eligible directions: **0/44**.
- Refusal induction on harmless prompts: **0.00 at all 44 layers**; required `> 0.00`.
- Lowest harmless KL: layer 49, **0.565684**; required `< 0.1`.
- Layer 49 reduced harmful refusal from 0.78125 to 0.53125 (absolute drop 0.25), but failed
  both the induction sign check and KL gate.
- Largest bypass drops reached 0.78125 (refusal to zero) at multiple layers, but those directions
  had destructive KL values of roughly 9--20 and still failed induction.
- Layers 27--32 and 34--35 often increased refusal to 0.94--1.00, another sign that the extracted
  axes were not a clean causal refusal direction under this intervention.

## Decision

The pre-registered E2 gate says that no eligible layer blocks editing. No `layer_star` was minted,
so the TC transfer check, A0/base ablation, A1 `l1.0_ablit`, A2/A3 carrier arms, conversion,
requantization, and downstream behavioral gates are dependency-blocked and were not run.

Increasing `INDUCE_ALPHA` after seeing the result is not justified: every candidate independently
missed the KL threshold, including the best-KL layer by 5.66x. Stage-1 `fable-tc-l1.0` remains the
deploy artifact; Stage-2 closes as a decisive negative bounded methods experiment.

## Receipts

- `DECISION_PACKET.md`
- `METHODOLOGY_ADDENDUM.md`
- `refusal_diagnostic_thinking_default.json` (invalid calibration, retained)
- `refusal_diagnostic.json` (corrected answer-channel sample)
- `EXTRACT_BASE_ANSWER_CHANNEL.log`
- `acts/base__*.pt`
- `SELECT_ANSWER_CHANNEL.log`
- `select_report.json`
- `rhat_base_all_layers.pt`
- `LOCAL_ARTIFACTS.sha256` (content hashes and sizes for the 813.78 MiB of local, regenerable
  activation tensors; the tensors themselves are intentionally excluded from Git)
