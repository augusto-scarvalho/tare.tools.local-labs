# Aborted before execution

The package was stopped after its first targeted unit run. The frozen test SHA-256 `fbe02fcdea497742f85979efe51c6205464edffd71c79c0034c5b52b032b87a2` incorrectly expected the historical sidecar to reject the negative sign in `{"delta":-12}`; actual diagnostic replay accepted it with zero interceptions. No live-model measurement or receipt was produced.

The workspace test was corrected after the packet entered `BLOCKED`, so its current hash intentionally differs from the frozen failed implementation. The successor must freeze a new implementation digest and use the independently reproduced nested-object/array failure instead.
