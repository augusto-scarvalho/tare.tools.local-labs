#!/usr/bin/env bash
set -euo pipefail

repo=HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF
revision=993a5971fda8f30dd1b7eb2654792ba4415c7460
name=Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
root=/home/augus/models/qwen38-27b/hauhaucs-aggressive-993a5971
path="$root/$name"
expected_bytes=17923393664
expected_sha=ba36dc3c2b2ff5e0aa5d71092a8894546996a6a119ae391803dda07cdc08516d

mkdir -p "$root"
/home/augus/.local/bin/hf download "$repo" \
    HauhauCS-FastMTP-Ed25519-PUBLIC.pem \
    HauhauCS-RELEASE-MANIFEST.json \
    HauhauCS-RELEASE-MANIFEST.json.sig \
    --revision "$revision" --local-dir "$root"

openssl pkeyutl -verify -rawin -pubin \
    -inkey "$root/HauhauCS-FastMTP-Ed25519-PUBLIC.pem" \
    -in "$root/HauhauCS-RELEASE-MANIFEST.json" \
    -sigfile "$root/HauhauCS-RELEASE-MANIFEST.json.sig"

/home/augus/.local/bin/hf download "$repo" "$name" \
    --revision "$revision" --local-dir "$root"

actual_bytes=$(stat -c %s "$path")
test "$actual_bytes" = "$expected_bytes"
printf '%s  %s\n' "$expected_sha" "$path" | sha256sum -c -
printf 'REVISION=%s\nFILE=%s\nBYTES=%s\nSHA256=%s\n' \
    "$revision" "$path" "$actual_bytes" "$expected_sha"
echo "=== DONE $(date -u +%FT%TZ) ==="
