# ADAPT-00C behavioral finalist panel - result

## Verdict

`NO_ARM_PROMOTED`; ADAPT-01 trace distillation remains blocked.

The loss screen was directionally informative but insufficient for behavioral
qualification. LoKr improved exact GSM8K behavior substantially over the base,
but missed the frozen correctness floor by one problem and had too many
non-terminating generations. LoRA and IA3 also failed the promotion rule.

## Frozen roles and panel

- Control: unadapted `Qwen/Qwen3.5-0.8B-Base` at revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Conventional anchor: LoRA
- Target-loss leader: LoKr
- Footprint leader: IA3
- Target: 32 disjoint GSM8K records from ADAPT-00A/B, greedy, 192-token cap
- Protected: 16 frozen normal-QA records, greedy, 128-token cap
- Machine receipt: `raw/results.json`

## Results

| Arm | Target correct | Required format | Protected pass | Natural EOS (48) | Median target tokens | Teacher ratio | Promotion |
|---|---:|---:|---:|---:|---:|---:|---|
| Base | 4/32 | 0/32 | 3/16 | 40/48 | 14.0 | 0.10x | control |
| LoRA | 10/32 | 26/32 | 3/16 | 38/48 | 121.0 | 0.85x | `FAIL` |
| LoKr | **15/32** | **27/32** | **4/16** | 35/48 | 136.5 | 0.96x | `FAIL` |
| IA3 | 4/32 | 0/32 | 3/16 | 35/48 | 55.5 | 0.39x | `FAIL` |

The teacher median was 142.5 tokens on these target records.

## Gate localization

- LoRA improved target correctness by six and preserved the base protected
  count, but failed both the 16/32 correctness floor and 46/48 EOS floor.
- LoKr improved target correctness by eleven, preserved/provisionally improved
  protected behavior, and stayed inside the teacher-length limit. It failed the
  correctness floor at 15/32 and the EOS floor at 35/48.
- IA3 matched the base's 4/32 and failed the required gain, correctness, and EOS
  gates. Its lower loss did not produce useful answer behavior at this budget.

The low absolute protected scores are expected for a base checkpoint rather
than an instruction model; the preregistered gate treated retention relative to
that base. This does not make 3–4/16 a deployable ordinary-QA result.

## Decision

- Do not open ADAPT-01 training from these adapters.
- Preserve LoKr as the strongest research candidate, not a promoted one.
- A future retry requires a new preregistered training-budget or model-scale
  hypothesis. It cannot be justified by relaxing the missed 16/32 or EOS gates.
- Do not infer production Fable/27B behavior from these 0.8B results.

## Service restoration

Only `llm-inference.service` was stopped; embeddings stayed active. Both
services were active afterward, GPU allocation returned to approximately
20.9 GiB, and the Fable canary again returned exactly
`adapt00-baseline-restored-ok`. Fan Control and MSI Afterburner were untouched.
