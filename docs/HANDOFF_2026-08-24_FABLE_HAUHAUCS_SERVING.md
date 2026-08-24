# Local inference lab handoff — 2026-08-24

This is the authoritative operational handoff after host/WSL tuning, Fable-TC
deployment, HauhauCS Qwen3.8 qualification, ordinary-question comparison, and
PT-BR locale-control closure. It supersedes the live-state and next-work sections
of the 2026-08-23 campaign handoff. Historical results remain authoritative only
for the exact artifact/build/runtime tuple they record.

## 0. Next agent: start here

This handoff is the entry point for the next agent. Do not reconstruct the
campaign from chat history. Read this file, then consult the linked immutable
receipts only for the work being resumed.

Last live verification: **2026-08-24 06:55:03 -03:00**.

- repository: clean `master` at `603a24218fa2043f8e562f8fe22ccf6e5e4b1893`;
- remote: `origin/master` at the same SHA;
- CI: [`local-labs-ci` run 32713908417](https://github.com/augusto-scarvalho/tare.tools.local-labs/actions/runs/32713908417)
  completed successfully for that exact SHA;
- `llm-inference.service`: active/running, PID 205079, `NRestarts=0`;
- `llm-embedding.service`: active/running, PID 203666; its `NRestarts=9` is
  historical and the serving PID was preserved during the final closure;
- `llm-locale-proxy.service`: active/running, PID 209766, `NRestarts=0`;
- 8080 identity: exact Fable-TC path, context 8192, build `b10159-068764d92`;
- 8081 identity: exact Nomic Embed Q8 path, context 2048, build `b9863-5e7f6271c`;
- 8082 health: `{"status":"ok"}`.

There is no half-finished authorized experiment to resume. HauhauCS is a
qualified retained candidate, not the current default. The only remaining work
is the optional, dependency-gated list in section 8; start one of those gates
only after the user selects or authorizes it.

Before touching the serving baseline, capture repository status, unit state,
`/props` identity, health, restart counts, and GPU state. Preserve these
invariants:

1. stop or restart `llm-inference.service` through `systemctl`, never by killing
   its `Restart=always` child;
2. never use a host-wide `pkill -f llama-server`;
3. leave the independent embedding service and port 8081 untouched;
4. use an explicit experimental port when possible and verify exact
   `/props.model_path` before collecting evidence;
5. restore Fable with `ops/qwen38-bringup/restore_fable_service.sh`, then verify
   all three endpoints before declaring the baseline recovered;
6. preserve invalid and superseded receipts with labels; do not pool or silently
   overwrite them.

Read [`runs/serving/CURRENT.md`](../runs/serving/CURRENT.md) for the concise
serving contract and [`CHANGELOG.md`](../CHANGELOG.md) for the published change
boundary.

## 1. Current live state

| Endpoint | State | Identity |
|---|---|---|
| `http://127.0.0.1:8080` | active canonical text service | `fable-tc-l1.0`, 8,192 context, native MTP n4, engine b10159 |
| `http://127.0.0.1:8081` | active independent embedding service | Nomic Embed Text v1.5, 768 dimensions |
| `http://127.0.0.1:8082/v1` | active loopback locale-controlled OpenAI endpoint | injects frozen `qwen38-ptbr-v2` contract, forwards to 8080 |

Systemd units `llm-inference.service`, `llm-embedding.service`, and
`llm-locale-proxy.service` are enabled and active. The locale proxy is
loopback-only; no firewall or LAN exposure was added. The raw port 8080 remains
available for benchmarks and clients that intentionally need unmodified model
behavior.

Operational details and rollback are in [`runs/serving/CURRENT.md`](../runs/serving/CURRENT.md)
and [`FABLE-TC-SERVE-2026-08-23`](../runs/serving/FABLE-TC-SERVE-2026-08-23/RESULT.md).

## 2. Host and WSL tuning closure

The isolated canonical b10165 build passed the matched 131k A/B against the
previous b9863 build: short decode improved 3.99%, long decode improved 8.76%,
and prompt throughput stayed inside the frozen 3% regression boundary. It was
qualified but not promoted because the later explicit serving choice selected
Fable-TC.

Resource-envelope findings:

- 57,344 was the largest tested 4 GiB-reserve context without embeddings;
- 43,008 was the largest tested 4 GiB-reserve context with embeddings resident;
- the exclusive 131,072 profile remained valid but low-reserve;
- n4/ub512 at 53,248 passed the 4 GiB floor and improved decode 5.30% over n3;
- the embedding endpoint was converted from a manual process into the enabled
  `llm-embedding.service`; a CPU-offload arm recovered too little VRAM and was
  rejected, so the exact GPU baseline was restored.

Receipt: [`HOST-WSL-TUNE-2026-08-23`](../runs/optimization/HOST-WSL-TUNE-2026-08-23/RESULT.md).

## 3. Fable-TC serving decision

Fable-TC was deliberately promoted to the persistent 8080 service as the concise,
uncensored Qwen3.6-derived default. Its frozen deploy artifact is
`fable-tc-l1.0-Q4_K_M.gguf`, SHA-256
`052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b`.
The versioned b10159 runtime resolved all engine libraries from its immutable
directory. Health, deterministic chat, MTP activity, embedding coexistence,
binary identity, restart state, and GPU telemetry passed.

The deploy profile intentionally uses 8,192 context. Do not infer the prior
Qwen3.8 131k context policy applies to Fable-TC.

## 4. HauhauCS Qwen3.8 qualification

Candidate identity:

- repository: `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`;
- revision: `993a5971fda8f30dd1b7eb2654792ba4415c7460`;
- artifact: `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf`;
- SHA-256: `ba36dc3c2b2ff5e0aa5d71092a8894546996a6a119ae391803dda07cdc08516d`;
- signed release manifest: verified;
- runtime: b10165, commit `71676e46c`;
- optional third-party FastMTP patch: not installed or used.

Measured gates:

| Gate | HauhauCS result | Reference |
|---|---:|---:|
| HumanEval+ exact 60-task subset | 56/60 | Fable 53/60; vanilla Qwen3.8 57/60 |
| Benign low-refusal panel | 44/44 comply | vanilla Qwen3.8 24 comply, 18 hedge, 2 refuse |
| 131,072 context load | PASS | 1,402 MiB idle VRAM reserve |
| 72,049-token needle | PASS | exact `JADE-7319` |
| Native-MTP decode | 91.37 tok/s median | 94.47% draft acceptance |

The candidate is qualified and retained, but not auto-promoted. It is the stronger
measured coding/low-friction option; Fable remains the better operational default
for ordinary short Portuguese use without a client contract.

Receipt: [`QWEN38-HAUHAUCS-AGGRESSIVE-2026-08-23`](../runs/requalification/QWEN38-HAUHAUCS-AGGRESSIVE-2026-08-23/RESULT.md).

## 5. Ordinary-question comparison

The frozen 48-task Portuguese-first deterministic panel produced:

| Model | No system prompt | First generic PT prompt |
|---|---:|---:|
| Fable-TC | 44/48 | 45/48 |
| Vanilla Qwen3.8 | 43/48 | 44/48 |
| HauhauCS | 38/48 | 43/48 |

All five HauhauCS-only raw failures were semantically correct content emitted in
English. The no-prompt result is therefore a real locale/instruction-adherence
loss, not evidence of five missing facts. Invalid runs caused by argv encoding and
port contamination are retained and labeled; they are excluded from conclusions.

The benchmark harness now refuses occupied ports, verifies `/props.model_path`,
supports UTF-8 system-prompt files, and stops only the exact experimental model
path. The former host-wide `pkill -f llama-server` could kill the independent
embedding process and was removed.

Receipt: [`QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23`](../runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/RESULT.md).

## 6. PT-BR locale-control closure

A stronger generic contract was selected on a new 48-task dev panel, then frozen
before a separate 48-task blind test. The selected v2 contract scored:

- HauhauCS dev: 48/48;
- HauhauCS blind test: 48/48;
- Fable-TC blind test under the same contract: 48/48.

On the original panel, replayed only after the blind result, HauhauCS scored 44/48
and Fable 43/48. No candidate-only language-drift error remained.

The first dev grader produced five false negatives because it required a specific
noun in otherwise valid Portuguese clarification questions. Original task files
and raw generations were preserved; the structural grader correction and
derivative regrade receipt are explicit. The untouched test was not opened before
contract selection.

This result closed the weight-edit branch: no custom ablation, LoRA, new model
download, or training environment was needed. HauhauCS publishes only GGUF
artifacts, and transferring a locale LoRA trained on the official base would add
unearned risk after a perfect blind operational result. Direct Fable/HauhauCS
weight mixing remains invalid because they derive from different Qwen generations.

The selected contract is served through `llm-locale-proxy.service` on 8082. The
proxy preserves existing system messages, supports streaming and non-streaming
OpenAI chat requests, and is covered by five deterministic unit tests.

Receipt: [`QWEN38-HAUHAUCS-LOCALE-CONTROL-2026-08-23`](../runs/requalification/QWEN38-HAUHAUCS-LOCALE-CONTROL-2026-08-23/RESULT.md).

## 7. Repository changes in this closure

- registered revision-bound HauhauCS and vanilla Qwen3.8 model profiles;
- added explicit Qwen3.8 reasoning-template selection to A2 benchmarks;
- corrected refusal classification so quoted refusal phrases are not assistant
  refusals;
- added UTF-8 system-message support to the request collector;
- narrowed experimental cleanup to exact model-path process identity;
- added deterministic normal-QA and locale dev/test tooling and immutable receipts;
- added the loopback locale-contract proxy, systemd unit, tests, and client guide;
- added activation, restoration, download, canary, performance, and 32k/131k
  serving assets for the candidate;
- preserved invalid/superseded attempts instead of pooling them with valid runs.

No model weights, runtime logs, credentials, or private keys are repository
artifacts. The largest new tracked file is below 100 KiB.

## 8. Remaining bounded work

HauhauCS already satisfies the requested coding, low-friction, context, speed, and
locale-control decision. Before replacing Fable as the broad default, remaining
optional gates are:

1. agent/tool-calling regression against vanilla and Fable;
2. a broader math/general-reasoning panel;
3. a reliability soak only after renewed explicit authorization;
4. an isolated native-MTP versus FastMTP comparison only if the third-party patch
   becomes worth its extra trust and maintenance surface.

Do not start a weight edit for locale control unless new held-out evidence defeats
the frozen contract. Do not expose 8082 beyond loopback without a separate network
authorization and firewall review.

## 9. Verification and recovery

Repository CI remains deterministic and CPU-only: compile all Python, run pytest,
then run the benchmark-harness self-test. GPU results are represented by frozen
receipts and are not regenerated in CI.

Operational recovery:

```bash
sudo systemctl restart llm-inference.service
sudo systemctl restart llm-locale-proxy.service
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8082/health
```

To remove only locale enforcement, disable `llm-locale-proxy.service` and keep
8080/8081 untouched. To restore Fable after a candidate canary, run
`ops/qwen38-bringup/restore_fable_service.sh` and verify model identity, context,
engine fingerprint, embedding health, and restart counts.
