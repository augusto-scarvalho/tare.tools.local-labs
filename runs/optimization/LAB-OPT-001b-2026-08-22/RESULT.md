# LAB-OPT-001b result — blocked by canonical VRAM envelope

Decision: **ABORTED_BY_PRE_REGISTERED_GATE / NO DEFAULT CHANGE**

The exact live-equivalent control (`MTP n3`, explicit binary-default `ubatch=512`, context 131,072)
loaded successfully but left only 2,782 MiB of free VRAM with the embedding service resident. This
is below the frozen 4,096 MiB floor, so the harness stopped before equivalence or performance probes
and did not launch the challenger.

After restoration, the unchanged canonical service reported 2,785 MiB free, independently confirming
that the measurement was not peculiar to the candidate launcher. Both 8080 and 8081 were healthy and
the SERVE/LAB state was coherent after cleanup.

This does not reject `n4/ub1024`; it establishes that a 131k comparison under the existing 4 GiB
reserve is infeasible because the control itself violates the envelope. LAB-OPT-001's 32k screen
remains useful but cannot promote a deploy default. A separate resource-envelope packet must decide
whether to reduce allocated context, revise the reserve policy explicitly, or retain the current
profile knowingly.

Evidence: `trials/confirm-n3-ub512.json` and `logs/confirm-n3-ub512.log`.
