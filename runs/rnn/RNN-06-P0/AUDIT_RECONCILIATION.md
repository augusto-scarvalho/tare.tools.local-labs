# RNN-06-P0 — AUDIT_RECONCILIATION (append-only closure)

**Status:** CLOSURE + PROVENANCE reconciliation only. **No GPU rerun. No measured result, curve, threshold, classification, or `P0_RESULTS_*` changed. Nothing pushed.** All original P0 protocol/results commits are left intact (no amend/rebase/rewrite).

**Source of this reconciliation:** the independent audit's requirements as supplied inline in the closure request (8 numbered sections). **The referenced file `AUDIT_RECONCILIATION_RNN-06-P0_2026-08-11.md` was NOT present in the session uploads** — the uploads folder for this session contained only `dc7bee51-Screenshot_20260811_161228_Claude.jpg` (the background-tasks screenshot). Its points are therefore transcribed from the request and reconciled against **live Git**, exactly as done for the RNN-06 research/design reconciliation-1 precedent (whose referenced audit file was likewise absent). `PACKET_AUDIT_FILE = NOT_PRESENT_IN_UPLOADS`.

**Accepted P0 result (unchanged, exploratory):**
- GDN (`linear-moe-hub/Gated-Deltanet-1.3B`): `MODEL_NOT_RUNNABLE`; scientific phenomenon **NOT TESTED**.
- DeltaNet (`fla-hub/delta_net-1.3B-100B`): `TASK_COMPETENT = NO`, `P0_GRADED_BAND = NOT_FOUND_WITHIN_BUDGET`.
- Mamba-2 (`AntonV/mamba2-1.3b-hf`): `TASK_COMPETENT = YES`, `P0_GRADED_BAND = PLAUSIBLE` (exploratory).
- `FIXED_BACKBONE_GRADED_REGION = NOT_QUALIFIED` (only RNN-06B may mint it).
- `QWEN_GDN_TRANSPLANT_GATE = DEFER`.

---

## §1 — Final Git truth (live)
- **current HEAD (before closure commit):** `2607c9921036efc3532e34866dd96737c1babc58`
- **branch:** `master` · **upstream:** none · **pushed:** NO (no remote-tracking branch; no `git push` executed this session or in closure).
- **staged:** none · **unstaged (tracked modifications):** none.
- **untracked relevant to RNN-06-P0 (before closure):** only `runs/rnn/RNN-06-P0/RNN-06-P0-audit-bundle.zip` (the first delivered bundle; a derived deliverable). Other untracked entries in the repo (`.harness/`, pre-existing `git_evidence.txt`/`stdout.log`/adapter dirs under RNN-04/05*/08*) are unrelated and pre-existing.

**Commit `46098e5` verified from live Git:**
- exists (`git cat-file -t` → `commit`).
- full SHA: `46098e552caaca9fe9a71e2cc73e1d4765dd910e`
- parent SHA: `7d7feedb24ba35f2b931532339a5dde1c609f799` (the pre-run protocol commit)
- tree SHA: `fa023582296e225f7781f1201085b45883806879`
- `git show --name-status 46098e5`: `M ops/rnn_06_p0_mqar.py`, `A ops/rnn_06_p0_bundle.py`, `A runs/rnn/RNN-06-P0/{HANDOFF.md, MODEL_IDENTITY_DELTANET.json, MODEL_IDENTITY_MAMBA2.json, P0_CURVES.csv, P0_DECISION.md, P0_RESULTS_DELTANET.json, P0_RESULTS_MAMBA2.json, calibration_examples.json, git_evidence.txt, stdout_sweep.log, stdout_sweep_mamba.log}`, `M runs/rnn/RNN-06-P0/P0_PROTOCOL.md` (14 files, +1246/−26).

### Handoff self-reference resolution — **case A + B (not C)**
The delivered handoff says *"Commit 2 (results): 46098e5 … adds … this handoff …"* while itself containing the string `46098e5`. Live Git shows:
- **`46098e5`** committed `HANDOFF.md` with the **placeholder** `` `<RESULTS_COMMIT>` `` (line 10) — an **earlier** version.
- **`2607c99`** (later, `docs(rnn): … fill results-commit hash`) committed the **delivered** `HANDOFF.md` with `` `46098e5` `` filled in (line 10) plus the parenthetical "(This handoff's own final commit updates this hash…)".
- Only these two commits touched `HANDOFF.md`.
- **Resolution:** `46098e5` contains an earlier handoff (**A**); the delivered handoff bytes live in the **later** commit `2607c99` (**B**). The delivered handoff is **committed and clean** at HEAD (not uncommitted → not C). History is **not** normalized; the self-reference is a benign forward-reference (the results-commit hash is only knowable after that commit exists, so the handoff naming it was necessarily committed one step later).
- Cross-check: the first delivered ZIP's `MANIFEST.json` records `HANDOFF.md` sha256 `82990be13b55074bb8bfb82b2cafb699e7ae220c0b8d9db9dbc94a3026ad0b9e` = the committed `2607c99` blob (below). So the **delivered** handoff == the `2607c99` bytes.

---

## §2 — Bytes of the delivered artifacts (Git blob SHA-1 + content SHA-256)
Working tree is clean, so for every file except `P0_CURVES.csv` the committed blob == the working-tree/bundled bytes. All committed at `46098e5` unless noted.

| File | committed? | delivered bytes in | Git blob (SHA-1) | content SHA-256 | bytes |
|---|---|---|---|---|---|
| `runs/rnn/RNN-06-P0/P0_PROTOCOL.md` | yes | `46098e5` | `ab9b9ef7c2d8141d678c93cfc0c27ce86038d90a` | `3122af44ec386bbc563481968090a5d74554ae5fd54b7592971d3923f85f9543` | 14399 |
| `runs/rnn/RNN-06-P0/P0_RESULTS_DELTANET.json` | yes | `46098e5` | `54f1a161dbb9d2ee9ed308c62bde56fe539a1334` | `5b45433087843ff3fc2ff440dd0e69e44d3ab5e2de43662251af448e5589b480` | 6249 |
| `runs/rnn/RNN-06-P0/P0_RESULTS_MAMBA2.json` | yes | `46098e5` | `d35db76438801d54cb8b6fac7541371db8118b20` | `97a6c9454c1e0095cd20db4811907c94556586a2a71125e90154f18e96a00956` | 6508 |
| `runs/rnn/RNN-06-P0/P0_DECISION.md` | yes | `46098e5` | `97362796c7a21e8a90f7f66265fb158623050de1` | `7869cf3ec33649bb11507104002c86959c5fb154a843151d751ea197ac374809` | 7365 |
| `runs/rnn/RNN-06-P0/HANDOFF.md` | yes | **`2607c99`** | `a07d29e9f05fd32a785d02aa61cb0ceee88b82dd` | `82990be13b55074bb8bfb82b2cafb699e7ae220c0b8d9db9dbc94a3026ad0b9e` | 11728 |
| `ops/rnn_06_p0_mqar.py` | yes | `46098e5` | `00eaeb39377ee935857d9b16b7a398dc5a6098dd` | `a5023872400f13966208f37b9b58738ec3e07bdce238d43187083b97845b4263` | 26456 |
| `runs/rnn/RNN-06-P0/P0_CURVES.csv` (committed, **LF**) | yes | `46098e5` | `e913b3600ae5be0f4a30ac392d4b9512ab87ac39` | `8edad565fcac2b97faf25daad2aa20c4099ab3ceee759a2372e124d0d54cde51` | 1597 |
| `runs/rnn/RNN-06-P0/P0_CURVES.csv` (working-tree / **first ZIP**, **CRLF**) | — | first ZIP | — | `b98b2979d910fc3300511950c61dd2f1632f7092e8b32760cf40f22e9a56920d` | 1610 |

### CSV line-ending provenance note (benign, fully explained)
`.gitattributes` declares `* text=auto eol=lf`. The scout's CSV writer (`ops/rnn_06_p0_curves.py`, Python `csv` on `open(...,newline="")`) emits **CRLF** terminators → the working-tree/first-bundled `P0_CURVES.csv` is **1610 B (CRLF, 13 CRLF), sha256 `b98b2979…`**, while Git normalized the committed blob to **LF → 1597 B, sha256 `8edad565…`**. The 13-byte delta is exactly the 13 line terminators. **Row/column values are byte-identical; only EOLs differ.** No measured value changed. The final bundle below records BOTH hashes and (for consistency) bundles the committed **LF** copy via `git show`.

---

## §3 — Executed-source identity
The scout source `ops/rnn_06_p0_mqar.py` was adapted **during** the session for `--impl`, the Transformers-native Mamba backend + concrete `Mamba2ForCausalLM`, `chunk_size`/config overrides, seq-length **autobatch**, and per-batch **OOM resilience**. Two committed blobs bound the adaptation:
- `7d7feed:ops/rnn_06_p0_mqar.py` → Git blob `ae1c27d0d9963d95f12605a8a055eb15a232678b`, sha256 `fa90087a00ac018e75dbfbefb8986fed4888437b384bc4a1dde44346adf22a29`, 24241 B (**pre-adaptation**: no `--impl`/autobatch/OOM-resilience).
- `46098e5:ops/rnn_06_p0_mqar.py` → Git blob `00eaeb39377ee935857d9b16b7a398dc5a6098dd`, sha256 `a5023872400f13966208f37b9b58738ec3e07bdce238d43187083b97845b4263`, 26456 B (**post-adaptation**: all of the above). `2607c99` carries the identical blob (that commit changed only `HANDOFF.md`).

**Best-available circumstantial binding (HIGH confidence, from committed config fields — corroboration, not cryptographic proof):**
- **DeltaNet substantive sweep** → `P0_RESULTS_DELTANET.json` has `config.impl = <ABSENT>`, `config.autobatch_budget = <ABSENT>`, `config_overrides_applied = {}` ⇒ produced by a script **without** the adaptation features ⇒ consistent **only** with the `7d7feed` blob `ae1c27d0` (which was committed immediately before the DeltaNet run, with no intervening edit).
- **Mamba-2 substantive sweep** → `P0_RESULTS_MAMBA2.json` has `config.impl = "transformers"`, `config.autobatch_budget = 1536`, `config_overrides_applied = {"chunk_size": {"was":256,"now":32}}` ⇒ produced by a script **with** the adaptation features ⇒ consistent **only** with the `46098e5` blob `00eaeb39`.

**Classification (strict):** `PER_CANDIDATE_EXECUTED_SOURCE_IDENTITY = NOT_PROVEN`.
Rationale: the runtime captured **no source hash**, and there is **no reflog/stash entry recording the exact on-disk bytes at execution instant**; the binding above is inferred from commit timeline + committed config-field discriminators, not from an execution-bound cryptographic artifact. Per the audit this is **acceptable for exploratory P0**; the executed source is **not reconstructed from memory/prose** — only committed Git objects (with exact SHAs) and committed result-config fields are cited.
- `executedSourceSha256[DeltaNet]` = **NOT_PROVEN** (best candidate: `fa90087a…`, blob `ae1c27d0`, object `7d7feed`).
- `executedSourceSha256[Mamba-2]` = **NOT_PROVEN** (best candidate: `a5023872…`, blob `00eaeb39`, object `46098e5`).
**Remediation carried forward:** RNN-06A/06B MUST hash and pin the exact executed source (commit the runner, then have it self-record its own blob SHA / a source SHA-256 into the results JSON) **before** any outcome-bearing run.

---

## §4 — Scientific caveats preserved (Mamba P0 is exploratory, NOT confirmatory)
- **Sequence length and the write→query gap co-vary with pressure P** (seq_len 18→514 as P 4→128). P0 therefore **does NOT isolate recurrent-state capacity** from length/position; a length-matched-packing + position-only control is required in 06B.
- **Constrained recall, unconstrained-exact success, and format adherence are DISTINCT channels** — reported separately; format adherence falls with pressure for both models. A model that merely stops emitting the value format is different from one that forgets. Constrained accuracy is the reported band metric.
- **Mamba `n_eval = 64` is exploratory** (SE ≈ 0.06; single master seed; single template family; `τ_hi/τ_lo` are P0 heuristics).
- **`AntonV/mamba2-1.3b-hf` is the exact tested candidate — NOT "generic Mamba-2".** It is an **unofficial HF-format conversion** of `state-spaces/mamba2-1.3b` (repo sha `c5b59d00ec85`).
- **P0 used Transformers-native NAIVE execution with NO `mamba_ssm` / `causal_conv1d`.** `chunk_size = 32` was an **execution accommodation** (mathematically-equivalent chunk tiling; sanity output verified identical to `chunk_size=256`), not a model change.
- **Changing backend/kernels later requires its own qualification** and does **NOT** automatically inherit P0 semantic equivalence.
- `PLAUSIBLE` **must not** be reinterpreted as confirmatory evidence. `FIXED_BACKBONE_GRADED_REGION` stays `NOT_QUALIFIED`.

---

## §5 — Next scope kept narrow
- `GDN_COMPATIBILITY_GAP = OPEN` — **not** resolved in this closure. The `linear-moe-hub/Gated-Deltanet-1.3B` fused-MLP-vs-fla-0.5.2 incompatibility remains a **separate** gap/packet so it cannot redesign or delay the Mamba lifecycle experiment.
- **Next implementation packet (recommended, NOT executed):** **`RNN-06A-MAMBA — State Observability & Lifecycle Qualification`** on the exact Mamba candidate/backend selected for qualification (`AntonV/mamba2-1.3b-hf`; pin backend + kernels + source SHA before outcome-bearing work).
- `QWEN_GDN_TRANSPLANT_GATE = DEFER`.

---

## §6 — Confirmations
`NO_GPU_RERUN = TRUE` · `NO_MEASURED_RESULT_MODIFIED = TRUE` (no `P0_RESULTS_*`, curve, threshold, or classification changed) · `NO_HISTORICAL_COMMIT_REWRITTEN = TRUE` (7d7feed/46098e5/2607c99 intact) · `NOTHING_PUSHED = TRUE` · `RNN-06A NOT started`.
