# LIGER-FEASIBILITY-2026-08-20 — pre-registration

**Status:** PRE-REGISTERED before dependency materialization, import, model
construction, or outcome-bearing tensor execution.

## Prior scout observations

These facts were observed while selecting the campaign and are not outcomes:

- the primary paper names `OpenSparseLLMs/Linearization` as its code artifact;
- upstream `main` resolved to commit
  `0b364eb81d2159cc0fd9818b95d2d07d75522043` on 2026-08-20;
- the repository is Apache-2.0 and was cloned read-only in detached-HEAD form to
  `/home/augus/src/Linearization`;
- its README instructs `git clone --recurse-submodules`, but the commit contains
  two gitlinks and no `.gitmodules` file. The README names the intended FLA and
  lm-evaluation-harness projects, so exact gitlink commits may still be recovered
  manually and will be treated as an explicit local provenance repair;
- the published Qwen3 path targets Qwen3-8B. No model weights have been
  downloaded and no Python environment has been changed.

## Question and hypothesis

Can the pinned official Qwen3 Liger implementation be made into a reproducible,
mechanically valid local substrate before any expensive fine-tuning?

The hypothesis is that the exact upstream source plus its exact gitlink revisions
can (1) produce a fully locked environment, (2) import on the RTX 3090/Ampere
stack, (3) construct a reduced Qwen3-shaped Liger model, and (4) execute finite
training and cached-recurrent forward paths with declared state-dict differences.

This is a **mechanism feasibility** campaign. It does not test the paper's quality
claims, does not tune on the completed Qwen38 fixtures, and cannot authorize a
large-model transplant or deployment.

## Frozen inputs

- Liger source commit: `0b364eb81d2159cc0fd9818b95d2d07d75522043`
- FLA gitlink commit: `72aa949f27dba47767f13226c45de29600d77312`
- lm-evaluation-harness gitlink commit:
  `1ba35e623b9bd9ca48df926f1a028043e159a6f2`
- hardware: one RTX 3090, compute capability 8.6, 24 GiB
- environment location: a new isolated venv under
  `/home/augus/.venvs/liger-feasibility-20260820`
- no change to the running llama.cpp Qwen38 server is allowed during dependency
  qualification; tensor tests that need GPU memory may stop it only after its
  exact restart argv and health contract are captured in the result record.

## Ordered gates

1. **Provenance gate:** manually materialize each README-named repository at the
   exact gitlink SHA; verify both worktrees are clean and record licenses and
   hashes. A missing or unreachable commit blocks the campaign.
2. **Environment gate:** create a new venv; resolve a lock without mutating prior
   RNN/TPTT environments. Record Python, CUDA, PyTorch, Triton, Transformers,
   FlashAttention, and FLA versions. Unbounded silent upgrades are forbidden.
3. **Static compatibility gate:** compare the reduced base-Qwen3 and Liger state
   dictionaries. Every missing/unexpected tensor must be enumerated; random
   initialization hidden behind `from_pretrained` is a hard failure.
4. **Construction gate:** instantiate a two-layer reduced config with Qwen3 head
   geometry, count parameters, and run one finite forward/backward microcase.
5. **Recurrence gate:** compare a full causal forward with tokenwise cached
   recurrent decoding under deterministic inputs. Report maximum absolute and
   relative error; do not invent a tolerance after seeing results.

## Metrics and thresholds

- provenance: exact SHA match and clean worktree for all three repositories;
- dependency qualification: all required imports succeed from the isolated venv;
- state transfer: zero unexplained missing/unexpected tensors; declared Liger-only
  tensors must be initialized by a documented rule;
- construction: expected logits shape, all finite outputs and gradients;
- recurrence: full-vs-tokenwise logits `max_abs <= 5e-2` and
  `max_rel <= 5e-2` in BF16, plus identical cache sequence length. The tolerance
  is deliberately loose for a kernel feasibility test, not a numerical parity
  claim;
- repeatability: two fresh-process executions with the same seed produce the
  same pass/fail vector and errors within `1e-3` absolute.

The decision is lexicographic. Later successes cannot compensate for failed
provenance, hidden random weights, non-finite values, or an invalid cache.

## Abandon and reversal

Stop before model-weight download or fine-tuning if any of the first three gates
fail. A local compatibility patch may be proposed only in a separate, explicitly
forked campaign with a patch digest; it may not be reported as an upstream
replication. All new state is confined to the isolated venv, pinned source clone,
and this run directory. No remote push or publication is authorized.
