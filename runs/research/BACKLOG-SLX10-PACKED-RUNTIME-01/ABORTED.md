# Execution aborted fail-closed

The deployed quantizer refused IQ2_XXS because `blk.0.attn_gate.weight` requires an importance matrix. No imatrix was frozen or available, and no service maintenance or inference occurred. IQ2_XXS remains objectively blocked until an immutable representative imatrix exists. A successor may materialize the supported Q2_K low-bit codec under newly frozen size limits.
