# LAB-HARNESS-002 - independent test-writer mutation gate

## Objective

Measure whether a cross-family local model acting only as an independent test
writer can create a compact suite that passes the unmodified Track H harness
primitives and kills a frozen set of plausible implementation mutations.

## Frozen roles

- Code under test: `src/model_lifecycle/agent_harness.py` from LAB-HARNESS-001.
- Independent test writer: resident Gemma-4-12B-it Q4_0, temperature 0,
  reasoning budget 256, maximum 2,048 output tokens.
- The model receives source plus behavioral requirements, but not the mutation
  implementations or the author-written tests.
- Oracle/executor: deterministic Python `unittest` subprocesses. No model grades
  its own tests.

## Frozen mutations

1. Remove stale base-digest rejection.
2. Remove cross-contract rejection.
3. Stop incrementing the contract version.
4. Drop the parent-digest chain.
5. Replace evidence instead of appending it.
6. Ignore missing baseline tests.
7. Ignore passing-to-failing regressions.

## Gates

- Generated code is a valid `unittest` module, uses no network, subprocess,
  sleep or randomness, and has no more than 180 non-empty lines.
- The suite passes against the unmodified implementation.
- Mutation kill rate is at least 5/7, with both stale-digest and regression
  mutations required to be killed.
- Results qualify this exact independent-test-writing mechanism only; they do
  not establish broad model critic calibration or human preference agreement.

