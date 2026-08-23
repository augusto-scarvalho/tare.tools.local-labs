# LAB-SERVE-001d — closed-loop MoE concurrency × MTP TPOT isolation

Status: **FROZEN BEFORE GENERATION**  
Date: 2026-08-22

## Question

Does the Qwen3.6-35B-A3B MoE MTP TPOT effect cross from beneficial at low concurrency to harmful
under sustained multi-slot decode, while still improving aggregate throughput and E2E latency?

## Fixed controls

- Model: `/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Model SHA-256: `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`
- Engine: lifecycle llama.cpp commit `068764d927ecd6d39665a46d31b1ee533eedabe7`
- Common flags: `-fa on --n-cpu-moe 8 --ctx-size 32768 --parallel 8 --cache-type-k q8_0
  --cache-type-v q8_0 --batch-size 2048 --ubatch-size 2048 --jinja`
- Only arm difference: `--spec-type draft-mtp --spec-draft-n-max 4` in MTP-on.
- Workload: SGLang serving benchmark, random fixed 1,024-token input, forced 128-token output,
  temperature 0, request rate infinite, exact Qwen3.6 tokenizer.
- Concurrency: `N ∈ {1,2,4,6,8}`; eight request waves per cell.
- Statistical unit: five independent paired server-level blocks, alternated arm order.

## Hypotheses and decisions

- H1: MTP median TPOT delta `(on - off)` is below zero at N=1 and N=2.
- H2: the same delta is above zero at N=4, N=6 and N=8.
- H3: aggregate output throughput remains higher with MTP at every N.
- A crossover is supported only if at least 4/5 paired blocks agree with H1/H2 at the relevant N;
  otherwise it is unresolved or absent. Report medians and every block, not only an aggregate fit.
- One paired diagnostic at N=4 sweeps forced output length 32 and 512 to check whether the sign is
  merely an output-duration artifact; it is descriptive, not a separate promotion gate.

Every cell must complete all requests, return exit zero, and retain sane token accounting. Invalid
cells are rerun before interpretation. The canonical Qwen3.8 service is restored after the campaign.

