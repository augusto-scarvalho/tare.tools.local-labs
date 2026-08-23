# LAB-HARNESS-003 - anti-slop and independent critic

## Objective

Qualify a deterministic anti-slop screen and measure a cross-family local critic
on an eight-case code-review panel with a frozen accept/reject oracle.

## Frozen design

- Static gate rejects syntax errors, broad/swallowed exceptions, blocking
  sleeps, unseeded randomness, TODO/FIXME/HACK markers, functions over 80 lines,
  and branch counts over 12.
- Independent critic: Gemma-4-12B-it Q4_0, temperature 0, reasoning budget 256,
  maximum 512 output tokens.
- Panel: four behavior-preserving/strengthening patches and four unsafe or
  maintainability-regressing patches. The critic sees the requirement, before
  code and after code; it does not see the oracle label.

## Gates

- Static anti-slop unit sentinels pass.
- Critic returns parseable `ACCEPT`/`REJECT` for 8/8 cases.
- Critic accuracy is at least 7/8 and unsafe accepts are zero.
- This is deterministic synthetic calibration, not human-preference or broad
  process-reward-model calibration.

