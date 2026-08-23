# LAB-OPS-001 — explicit SERVE/LAB state lock

**Status:** `COMPLETE / QUALIFIED / LIVE SERVE`

`lmctl` now carries a persistent, auditable and fail-closed SERVE/LAB mode lock. It
prevents a lab model or judge from being launched during canonical service, prevents
port 8080 from being launched in LAB, rejects mode transitions that contradict live
processes, and treats the independent embedding server on 8081 as an explicit allowed
auxiliary in both modes.

## Semantics

- `SERVE`: at most one text/judge server, and it must use canonical port 8080.
- `LAB`: canonical port 8080 is forbidden; at most one experimental text/judge server
  may use a non-8080 port.
- Embedding port 8081 is auxiliary and allowed in either state.
- An absent, malformed or schema-invalid state fails closed for launches.
- `mode set` requires an audit reason and supports `--expect` compare-and-set.
- Writes use an exclusive cross-process lock plus atomic `os.replace`. A crashed writer
  leaves a lock receipt and is not silently timed out/stolen.
- All `serve` launches consult the lock before process creation. Stop, inspection,
  telemetry and build commands remain available for recovery.

State is machine-local at
`%LOCALAPPDATA%\tare.tools.local-labs\lmctl-mode.json` (overridable with
`LMCTL_MODE_STATE`); runtime lock state is not committed as repository configuration.

## Qualification

Deterministic suite: **10/10 passed** in `tests/test_lmctl_mode.py`:

- missing and corrupt state fail closed;
- atomic initialization and SERVE→LAB compare-and-set;
- CAS mismatch preserves the previous state;
- empty audit reason is rejected;
- an existing writer lock is not stolen;
- LAB transition refuses active 8080 but permits auxiliary 8081;
- SERVE transition refuses experimental ports;
- 8080 maps to SERVE and other text/judge ports map to LAB;
- runtime drift and multiple primary servers fail closed;
- embedding 8081 coexists coherently in both modes.

Live negative and transition checks:

1. With Qwen3.8 on 8080 and embedding on 8081, SERVE was initialized and reported
   coherent.
2. `serve gemma-judge` on 8091 in SERVE was refused with exit 4 before launch.
3. `mode set lab` while 8080 was active was refused with exit 4; SERVE state remained.
4. After stopping only `llm-inference.service`, SERVE→LAB succeeded while 8081 stayed
   active and coherent.
5. A port-8080 launch in LAB was refused with exit 4 before launch.
6. LAB→SERVE compare-and-set succeeded; the canonical systemd unit was restored.

Final state:

- mode `SERVE`, owner `codex-autonomous-backlog`, reason
  `LAB-OPS-001 qualified; restore canonical service`;
- Qwen3.8 endpoint 8080 healthy, embedding 8081 healthy;
- `llm-inference.service` active/running;
- no experimental server left alive.

## Incidental path repair

Exercising the actual front-door exposed two stale migration references. `lmctl.py`
now imports the package from repository-root `src/` and builds from canonical
`/home/augus/src/slop.cpp`, replacing the obsolete `tools/benchmarks/src` and
`slop.cpp-main` paths.

## Evidence

- Implementation: `tools/benchmarks/lmctl.py`
- Tests: `tests/test_lmctl_mode.py`
- Commands: `python -m unittest -v tests.test_lmctl_mode`,
  `python tools/benchmarks/lmctl.py mode ...`

