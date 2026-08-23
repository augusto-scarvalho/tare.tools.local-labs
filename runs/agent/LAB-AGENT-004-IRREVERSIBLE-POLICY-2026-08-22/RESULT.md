# LAB-AGENT-004 irreversible recovery policy result

Decision: **PROMOTE APPLICATION POLICY / BOUNDED**  
Date: 2026-08-22

The single system policy fixed the deterministic tool-order failure localized by LAB-AGENT-002.
The reversed-order target passed 5/5 across seeds 0–4. The full eight-case suite then passed 8/8 in
canonical order and 8/8 with the tool list reversed, with zero blind retries and zero endpoint errors.

Promote the following application-level invariant for agent prompts: after an irreversible action has
an unknown outcome, immediately call an available idempotent status/check tool without requesting
permission, and never retry the irreversible action. This is qualified on the frozen local suite, not a
general BFCL claim. Results SHA-256:
`ceac07672eed1fa9354fb89f1de862555bee9114a0a30815f22064accf9746b4`.
