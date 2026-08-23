# Cache harness context compatibility correction

Status: **FROZEN BEFORE CORRECTIVE RERUN**  
Date: 2026-08-22

The stock cache-correctness harness contains a long-context case whose Mistral tokenization produces
9,634 prompt tokens. The preregistered 8,192-context judge profile therefore returned an over-context
HTTP error before that cell could execute. The partial attempt was interrupted and is invalid for the
cache gate.

The corrective run changes only server context allocation to 16,384 by appending a last-wins
`--ctx-size 16384` flag to the same profile. Model, artifact, endpoint, KV format, tools, four cache
cases, nonce, greedy sampling and validators remain unchanged. The 4,096 MiB reserve and 4/4 cache
gate remain binding. This is a qualification profile, not a default change.
