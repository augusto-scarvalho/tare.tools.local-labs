# LAB-OPT-001b — exact live-default confirmation

Frozen before execution on 2026-08-22.

## Reason

LAB-OPT-001 screened six explicit configurations but incorrectly designated `n3/ub2048` as the
incumbent. The live systemd unit omits `--ubatch-size`, and the pinned binary reports a default of
512. This follow-up corrects that control mismatch rather than rewriting the original packet.

## Arms

- Control: MTP draft depth 3 and explicit ubatch 512, otherwise the live service flags.
- Challenger: MTP draft depth 4 and explicit ubatch 1024, the LAB-OPT-001 screen winner.

Both arms allocate the canonical 131,072-token context, q4_0/q4_0 KV, one slot, 32 checkpoints,
batch 2,048, flash attention, all GPU layers, default mmap and 420 W. They run on port 8092 while
the canonical text service is stopped through systemd and the embedding endpoint remains live.

## Protocol and gates

- Three fixed greedy equivalence probes; challenger hashes must equal the control byte-for-byte.
- Three counterbalanced short/long performance repetitions per arm, 128 forced decode tokens.
- Minimum 4,096 MiB free VRAM after load; valid streaming boundaries and telemetry; non-empty runs.
- The challenger confirms only if it improves at least one median throughput axis by at least 5%
  and regresses neither axis by more than 3%.

No deploy file is edited automatically. Even a confirmation is a recommendation with bounded
evidence. Cleanup, SERVE-mode restoration and health verification are mandatory.
