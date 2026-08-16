#!/usr/bin/env bash
# mtp_tensor_check.sh — Phase 1 gate: does this GGUF carry the MTP draft head?
#
# WHY: MTP spec-decode (`--spec-type draft-mtp`) is our +33-70% decode lever, but community reports
# CONFLICT on whether Unsloth's Qwen3.8-27B GGUF ships the MTP tensors (some quants may strip them).
# A merged/finetuned variant can also drop them. This is a 5-second check; do it before relying on MTP.
#
# PASS: MTP/nextn/mtp tensors listed -> use --spec-type draft-mtp on the single GGUF.
# FAIL: none found -> download a4lg/Qwen3.8-27B-MTP-ONLY-GGUF (Q4_K_M or Q5_K_M; NOT Q8, too big) and
#       pass it as --model-draft <mtp.gguf> --spec-type draft-mtp, OR graft per their README.
#
# Usage: MODEL=/home/augus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf bash ops/qwen38-bringup/mtp_tensor_check.sh
set -u
LLAMA=${LLAMA:-/home/augus/src/llama.cpp-master}
MODEL=${MODEL:-/home/augus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf}

echo "== inspecting: $MODEL =="

# Prefer the python gguf reader (ships with llama.cpp); fall back to the C++ dumper.
if python3 - "$MODEL" <<'PY' 2>/dev/null
import sys
try:
    from gguf import GGUFReader
except Exception:
    sys.exit(2)
r = GGUFReader(sys.argv[1])
names = [t.name for t in r.tensors]
mtp = [n for n in names if any(k in n.lower() for k in ("mtp", "nextn", "next_n", "draft", "eh_proj", "shared_head"))]
# also surface the KV-metadata hint some converters set
meta = {k: str(v) for k, v in r.fields.items() if any(s in k.lower() for s in ("mtp","nextn","predict"))}
print(f"total tensors: {len(names)}")
print(f"MTP-ish tensors ({len(mtp)}):")
for n in mtp[:40]:
    print("   ", n)
if meta:
    print("MTP-ish metadata:", meta)
sys.exit(0 if mtp else 1)
PY
then
  echo ">> PASS: MTP tensors present -> --spec-type draft-mtp on this GGUF."
else
  rc=$?
  if [ "$rc" = 2 ]; then
    echo "(python gguf reader unavailable; falling back to llama-gguf dumper)"
    "$LLAMA/build/bin/llama-gguf" "$MODEL" r 2>/dev/null | grep -iE "mtp|nextn|next_n|draft|eh_proj|shared_head" \
      && echo ">> PASS (via dumper)" \
      || echo ">> FAIL: no MTP tensors found -> use a4lg MTP-ONLY GGUF as --model-draft."
  else
    echo ">> FAIL: no MTP tensors in this GGUF -> download a4lg/Qwen3.8-27B-MTP-ONLY-GGUF"
    echo "         and run with --model-draft <mtp.gguf> --spec-type draft-mtp."
  fi
fi
