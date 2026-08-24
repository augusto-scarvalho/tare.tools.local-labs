# Current serving state

As reverified on 2026-08-24 20:44 -03:00:

- Primary text endpoint `http://127.0.0.1:8080`: `fable-tc-l1.0`, context 8192,
  native MTP n4, engine `b10159-068764d92`, managed by
  `llm-inference.service`.
- Auxiliary embedding endpoint `http://127.0.0.1:8081`: Nomic Embed Text v1.5,
  768 dimensions, managed independently by `llm-embedding.service`.
- Locale-controlled OpenAI endpoint `http://127.0.0.1:8082/v1`: loopback-only
  proxy managed by `llm-locale-proxy.service`, injecting frozen contract
  `qwen38-ptbr-v2` into chat completions.
- Operating lock: coherent `SERVE`.
- Rollback: Qwen3.8 engine b10165 and its 131072-token configuration remain
  preserved under the Fable-specific systemd drop-in.

Canonical transition receipt:
`runs/serving/FABLE-TC-SERVE-2026-08-23/RESULT.md`.

## Qualified alternative retained

`qwen38-hauhaucs-aggressive-q4kp` passed the 2026-08-23 candidate gates at
56/60 HumanEval+, 44/44 comply versus vanilla Qwen3.8's 24/44, 131,072 context,
and 91.37 tok/s median with native MTP. It was not auto-promoted; see
`runs/requalification/QWEN38-HAUHAUCS-AGGRESSIVE-2026-08-23/RESULT.md`.

On the normal-question panel it scored 38/48 versus vanilla's 43/48 and
Fable-TC's 44/48 without a system prompt, driven by correct content emitted in
English on Portuguese tasks. With the same fixed UTF-8 Portuguese language
instruction on all three models, it scored 43/48 versus vanilla's 44/48 and
Fable-TC's 45/48. It met `NO_MEASURABLE_LOSS` against vanilla, but the two-task
reading-category deficit makes it `POSSIBLE_SMALL_LOSS` against Fable-TC. See
`runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/RESULT.md`.

A stronger generic locale contract was then selected on a separate dev panel and
scored 48/48 for both HauhauCS and Fable-TC on a frozen blind test. It also removed
all HauhauCS-only language-drift failures on the original panel. No ablation or
LoRA was justified; see
`runs/requalification/QWEN38-HAUHAUCS-LOCALE-CONTROL-2026-08-23/RESULT.md`.

The 2026-08-24 continuation added two bounded decision results without changing
the serving baseline: HauhauCS passed the agent/tool core 8/8 but scored 191/200
on GSM8K with eight truncations versus Fable's 195/200 and zero truncations.
FastMTP therefore stopped before installation. The newly licensed RWKV7 1.5B
artifact scored 13/48 on the frozen normal-question gate and remains
`HOLD_QUALITY`. See the current queue at
`docs/research/REMAINING_EXPERIMENTS_2026-08-24.md`.
