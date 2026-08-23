# Cache compatibility addendum

Status: **FROZEN BEFORE SECOND CORRECTIVE RERUN**  
Date: 2026-08-22

The 16,384 correction instantiated the first three cache cases, all of which passed, but Mistral's
tokenizer maps the frozen long-context fixture to 30,057 tokens. Its HTTP 400 is again an invalid
over-context cell, not a cache failure.

The final corrective run uses 32,768 context, the smallest power-of-two allocation above the measured
fixture. It proceeds only if at least 4,096 MiB remain free after load. All other factors and the 4/4
gate remain unchanged. The prior 8k/16k over-context attempts remain preserved as harness-compatibility
receipts and are not scored.
