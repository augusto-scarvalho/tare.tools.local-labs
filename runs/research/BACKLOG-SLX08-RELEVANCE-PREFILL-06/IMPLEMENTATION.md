# BACKLOG-SLX08-RELEVANCE-PREFILL-06 implementation

The experimental slop.cpp route remains disabled by default and behind
`SLOP_EXPERIMENTAL_SLX08=1`. R6 adds one request field,
`slx08_selected_block_indices`, plus response telemetry for selection mode and
the exact retained indices. The server validates the fixed half-context budget,
strict ordering, range, uniqueness, and retention of the first/final blocks.

The local-labs runner constructs 64 frozen long-context retrieval fixtures and
compares dense, naive alternating, and explicit relevance-selected arms on
identical token sequences. Fixtures rotate evidence through all 14 middle
positions. Focused tests cover every target position, selector shape failures,
the exact-answer scorer, gate recomputation, request routing and the C++ API
contract. Runtime validation adds four malformed-index requests that must fail.

No CUDA kernel or server-side semantic selector was added. The selector is a
small deterministic client helper; the physical runtime only applies and
reports the selected token blocks.
