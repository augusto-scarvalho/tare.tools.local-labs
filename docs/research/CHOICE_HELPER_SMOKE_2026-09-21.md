# Laya task-demand helper: CPU smoke

Date: 2026-09-21. Classification: adapter smoke, **UNQUALIFIED** for production
routing. No research backlog entry or fleet promotion was changed.

## Pinned inputs

- Model: `convaiinnovations/laya`, revision
  `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, subfolder `typed-decisions`.
- SDK: Laya 0.3.4, source commit
  `42626c348753fbb17572a813127df2278a1ec527`.
- Environment: Windows Python 3.12, torch `2.8.0+cpu`, transformers `4.57.6`,
  numpy `2.2.6`, safetensors `0.6.2`; two CPU threads on acer. No GPU access.
- Three cases frozen in `config/choice_helper_smoke.json`, SHA-256
  `4a6bd7c418f2a2500d129f8ec677cafd6fc68834b6006a9dc8186b4d634ae408`.
- One model load, exactly three forward calls, 120-second external deadline.
  No prompt tuning, retries, remote inference, training or Jev calls.

The SDK packaging initially failed under Windows' default text encoding before
any inference. Installing the same source with `PYTHONUTF8=1` resolved it.

## Observed results

| Case | Expected and observed | Input tokens | Warm elapsed |
| --- | --- | ---: | ---: |
| English spelling correction with exact check | bounded | 107 | 938 ms |
| Portuguese GPU lease / signal race | reasoning | 164 | 1,344 ms |
| Request with no project/scope/criteria | insufficient | 81 | 718 ms |

Load and artifact verification: **18,187 ms**. Total: **21,187 ms**.
Output tokens: zero. Full distributions and identities are retained in
[the raw receipt](evidence/choice-helper-20260921.json).

This proves that the pinned CPU adapter can consume these three inputs and
return useful typed choices. It does not prove general multilingual accuracy,
calibration, superiority to Jev or an LLM, optimal effort selection, or savings
per accepted Work. The highest probability was only 0.3817 on the ambiguous
case; a winning option is not a calibrated probability of correctness.

Decision: proceed to a small task-specific comparison if routing is pursued;
keep automatic selection disabled. Cold startup is material. A worker retained
in memory would need separate measured end-to-end evidence before claiming
subsecond dispatch overhead.

## Reproduction

Use an isolated CPU environment. Install the pinned SDK and dependencies, and
download only the five checkpoint files at the revision above. Create a
`tare.local-labs/laya-checkpoint/1` manifest with `directory`, model identity,
`sdk_version` and the SHA-256 of each file listed by Kernel's
`compute_plane.laya_choice_worker.FILES`. The adapter verifies bytes before
loading, refuses SDK text truncation and cannot silently choose CUDA.

```text
python -B tools/analysis/qualify_choice_helper.py \
  --kernel-root /path/to/tare.tools.kernel \
  --checkpoint /path/to/checkpoint.json \
  --output /external/evidence/new-result.json
```

The output must not already exist. The harness first writes a STARTED receipt;
interrupted attempts are preserved. Bound the command externally to 120 seconds.
The smoke harness cannot emit a QUALIFIED/PROMOTED result, even with three
matches. Local retained setup plan and dependency inventory are under
`.reconciliation/evidence-store/reports/adaptive-helper-20260921`.
