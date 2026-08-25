# ADAPT-00C behavioral finalist panel - pre-registration

## Finalist rule frozen before generation

ADAPT-00B left every passing arm on a multi-objective Pareto frontier. This
panel chooses three role representatives without inspecting generated output:

- LoRA: conventional anchor and exact ADAPT-00A configuration;
- LoKr: highest held-out target-loss improvement;
- IA3: fewest trainable parameters and lowest peak VRAM.

The unadapted base is the causal control. LoHa, BOFT, and trainable tokens are
not behaviorally run because they did not win one of these frozen roles. DoRA
is already rejected by its non-finite training result.

## Frozen panels

Target panel: the same 32 disjoint GSM8K IDs selected by seed 20260824 in
ADAPT-00A/B. Generation is greedy with at most 192 new tokens. Record numeric
answer correctness using the standing GSM8K extractor, required `####` format,
natural EOS, and generated token count. The teacher completion token count on
the same IDs is the concision reference.

Protected behavioral panel: the first preregistered IDs by category from the
existing frozen normal-QA set:

`f01,f02,f03,m01,m02,m03,r01,r02,r03,i01,i02,i03,c01,c02,s01,s02`

Use each task's existing deterministic grader, greedy decoding, and at most 128
new tokens. No task is used for training.

## Promotion gate

A finalist opens ADAPT-01 only if all are true:

1. at least 16/32 target answers are correct;
2. it improves by at least 3 target answers over the unadapted base;
3. protected pass count is no worse than base by more than one task;
4. at least 46/48 target-plus-protected generations end naturally;
5. median target output length is no more than 1.25 times the teacher median.

The gate is intentionally behavioral and demanding. Loss improvement alone
cannot promote a distillation geometry. No precision, learning-rate, step,
prompt, or token-budget rescue is permitted after results.

## Operations

Run base, LoRA, LoKr, and IA3 sequentially. Stop only
`llm-inference.service`; keep embeddings active and restore/verify Fable after
the complete panel. No production config or desktop GPU-control setting changes.
