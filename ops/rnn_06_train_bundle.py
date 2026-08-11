#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the full RNN-06 Mamba Qualification Train audit deliverable (mobile ZIP).

Bundles both backlog items (RNN-06A2-MAMBA-CONTINUATION + RNN-06B-MAMBA-BASE) plus
train-level protocol/amendment/handoff/git-evidence, with TRAIN_MANIFEST.json and
SHA256SUMS.txt. Because 06B is BLOCKED, includes an explicit BLOCKED.json marker rather
than pretending a qualified region exists. Excludes weights / HF cache / venv / .git /
large temporaries. Prints the ZIP's own SHA-256. Pure packaging; no GPU/model.
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ZIP = os.path.join(REPO, "runs", "rnn", "RNN-06-MAMBA-TRAIN",
                       "RNN-06-MAMBA-qualification-train-audit-bundle.zip")

# arcname -> repo-relative source
FILES = {
    # train level
    "RNN-06-MAMBA-qualification-train/TRAIN_HANDOFF.md":
        "runs/rnn/RNN-06-MAMBA-TRAIN/TRAIN_HANDOFF.md",
    "RNN-06-MAMBA-qualification-train/TRAIN_PROTOCOL.md":
        "runs/rnn/RNN-06-MAMBA-TRAIN/TRAIN_PROTOCOL.md",
    "RNN-06-MAMBA-qualification-train/AMENDMENT_1_chunksize.md":
        "runs/rnn/RNN-06-MAMBA-TRAIN/AMENDMENT_1_chunksize.md",
    "RNN-06-MAMBA-qualification-train/git_evidence/git_evidence.txt":
        "runs/rnn/RNN-06-MAMBA-TRAIN/git_evidence.txt",
    # 06A2
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/PRE_REGISTRATION.md":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/PRE_REGISTRATION.md",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_CONTRACT.md":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_CONTRACT.md",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_CHALLENGES.json":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_CHALLENGES.json",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_RESULTS.json":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_RESULTS.json",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_RESULTS_cs32.json":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_RESULTS_cs32.json",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_MATRIX.csv":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_MATRIX.csv",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_MATRIX_cs32.csv":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_MATRIX_cs32.csv",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_DECISION.md":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_DECISION.md",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/DECISION_ADDENDUM_cs32.md":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/DECISION_ADDENDUM_cs32.md",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/stdout_continuation.log":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/stdout_continuation.log",
    "RNN-06-MAMBA-qualification-train/RNN-06A2-MAMBA-CONTINUATION/stdout_continuation_cs32.log":
        "runs/rnn/RNN-06A2-MAMBA-CONTINUATION/stdout_continuation_cs32.log",
    # 06B
    "RNN-06-MAMBA-qualification-train/RNN-06B-MAMBA-BASE/PRE_REGISTRATION.md":
        "runs/rnn/RNN-06B-MAMBA-BASE/PRE_REGISTRATION.md",
    "RNN-06-MAMBA-qualification-train/RNN-06B-MAMBA-BASE/QUALIFICATION_SPEC.json":
        "runs/rnn/RNN-06B-MAMBA-BASE/QUALIFICATION_SPEC.json",
    "RNN-06-MAMBA-qualification-train/RNN-06B-MAMBA-BASE/STRESS_GRID.json":
        "runs/rnn/RNN-06B-MAMBA-BASE/STRESS_GRID.json",
    "RNN-06-MAMBA-qualification-train/RNN-06B-MAMBA-BASE/BASE_RESULTS.json":
        "runs/rnn/RNN-06B-MAMBA-BASE/BASE_RESULTS.json",
    "RNN-06-MAMBA-qualification-train/RNN-06B-MAMBA-BASE/BASE_CURVES.csv":
        "runs/rnn/RNN-06B-MAMBA-BASE/BASE_CURVES.csv",
    "RNN-06-MAMBA-qualification-train/RNN-06B-MAMBA-BASE/BASE_DECISION.md":
        "runs/rnn/RNN-06B-MAMBA-BASE/BASE_DECISION.md",
    "RNN-06-MAMBA-qualification-train/RNN-06B-MAMBA-BASE/stdout_base.log":
        "runs/rnn/RNN-06B-MAMBA-BASE/stdout_base.log",
    # runners (source of executed identity)
    "RNN-06-MAMBA-qualification-train/ops/rnn_06a2_challenges.py": "ops/rnn_06a2_challenges.py",
    "RNN-06-MAMBA-qualification-train/ops/rnn_06a2_continuation.py": "ops/rnn_06a2_continuation.py",
    "RNN-06-MAMBA-qualification-train/ops/rnn_06b_challenges.py": "ops/rnn_06b_challenges.py",
    "RNN-06-MAMBA-qualification-train/ops/rnn_06b_base.py": "ops/rnn_06b_base.py",
    "RNN-06-MAMBA-qualification-train/ops/rnn_06_train_bundle.py": "ops/rnn_06_train_bundle.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {
        "train": "RNN-06-MAMBA-qualification-train",
        "backlog_1_RNN_06A2": "CONTINUATION_LIFECYCLE = QUALIFIED (cs=256 and cs=32)",
        "backlog_2_RNN_06B": "EXECUTED; FIXED_BACKBONE_GRADED_REGION = BLOCKED (CONFOUNDED_WITH_LENGTH)",
        "pinned_chunk_size": 32, "note_amendment": "AMENDMENT 1 pinned cs=32 (memory)",
        "files": [], "skipped_missing": []}
    blocked_marker = {
        "packet": "RNN-06B-MAMBA-BASE", "FIXED_BACKBONE_GRADED_REGION": "BLOCKED",
        "reason": "CONFOUNDED_WITH_LENGTH",
        "detail": "MP graded curve is competent/monotone/robust but matched LC control degrades "
                  "in lockstep (mean(LC-MP)@{96,128} = -0.0026 < preregistered +0.15); loss is "
                  "generic length/state-saturation, not same-space associative interference.",
        "gate_passed": ["competence", "material_loss", ">=2_mid_band", "monotone", "robust_3of3"],
        "gate_failed": ["confound_controlled"]}

    sha_lines, present = [], []
    for arc, rel in FILES.items():
        ap = os.path.join(REPO, rel)
        if not os.path.isfile(ap):
            manifest["skipped_missing"].append(rel)
            continue
        digest = sha256_file(ap)
        manifest["files"].append({"arcname": arc, "repo_path": rel,
                                  "size": os.path.getsize(ap), "sha256": digest})
        sha_lines.append(f"{digest}  {arc}")
        present.append((ap, arc))

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for ap, arc in present:
            z.write(ap, arc)
        z.writestr("RNN-06-MAMBA-qualification-train/RNN-06B-MAMBA-BASE/BLOCKED.json",
                   json.dumps(blocked_marker, indent=2).encode())
        z.writestr("RNN-06-MAMBA-qualification-train/TRAIN_MANIFEST.json",
                   json.dumps(manifest, indent=2).encode())
        z.writestr("RNN-06-MAMBA-qualification-train/SHA256SUMS.txt",
                   ("\n".join(sha_lines) + "\n").encode())

    print(f"ZIP: {OUT_ZIP}")
    print(f"files_included: {len(present)} (+ BLOCKED.json + TRAIN_MANIFEST.json + SHA256SUMS.txt)")
    print(f"skipped_missing: {manifest['skipped_missing']}")
    print(f"ZIP_SHA256: {sha256_file(OUT_ZIP)}")
    print(f"ZIP_BYTES: {os.path.getsize(OUT_ZIP)}")


if __name__ == "__main__":
    main()
