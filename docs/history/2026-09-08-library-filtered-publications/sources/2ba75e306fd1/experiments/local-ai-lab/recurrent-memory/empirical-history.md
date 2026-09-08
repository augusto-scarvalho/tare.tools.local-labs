# Recurrent Memory — empirical history and falsifications

[← Research-line index](README.md) · [Experiments](../../README.md) · [Navigation](../../../NAVIGATION.md)

## Epistemic history

The research began with synthetic dense post-hoc Memory Caching ideas. Several rounds failed to establish a usable causal mechanism; some apparently promising results were invalidated by methodology/harness issues. That branch is **PARKED**, not quietly rewritten as success.

The program then moved to a real recurrent-model substrate and decomposed the problem:

1. prove continuation/lifecycle semantics;
2. induce controlled unique-binding load forgetting;
3. test whether earlier/same-aged control recurrent states retain accessible target information;
4. only then test target-agnostic recovery utility.

## Qualified substrate snapshot

AntonV/mamba2-1.3b-hf at the qualified pin; Transformers 4.48.3 native Mamba2ForCausalLM; torch 2.6.0+cu124; bf16; chunk_size=32; no mamba_ssm/causal_conv1d fast path for the qualified evidence.

## Qualified results

- continuation lifecycle: QUALIFIED;
- controlled unique-binding load forgetting: QUALIFIED;
- RNN-06B3 controlled load-loss effect: ~0.417 in the recorded U1→U176 setup;
- RNN-06C historical/control advantage N−L: ~0.453 with recorded CI [.385,.526];
- historical recurrent state information **presence/accessibility**: QUALIFIED;
- recovery utility: NOT_TESTED;
- Qwen/GDN transplant: DEFER.

Full recurrent cache state in the qualified substrate was roughly 49.59 MiB/sequence.

## Interpretation discipline

Presence ≠ deployable recovery. Oracle access ≠ target-agnostic selector. One substrate/config ≠ generic RNN result. Backend/kernel/dtype/chunk/state-boundary are part of execution identity.

---

**Next:** [Scientific interpretation →](scientific-interpretation.md) · **Methodology context:** [Local Model Lab](../../../research/local-inference/local-model-lab-methodology.md)
