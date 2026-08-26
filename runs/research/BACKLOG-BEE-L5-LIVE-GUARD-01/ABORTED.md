# Execution aborted fail-closed

The frozen implementation expected `/tokenize` to return a top-level `pieces` array. The active server returned token records under `tokens[*].piece`, so all observed piece arrays were empty and the independent pathology precondition failed for all 25 baselines. No gates were scored and no claim is permitted from this packet. The raw baseline responses are preserved in `raw/baseline_abort.json`; a successor must freeze the corrected response parser before rerun.
