# 🧪 Experiment Card: EXP-02 — Thinking Retention ($K=1$), KV-Cache Hit-Rate ($R_{\text{cache}}$) & Qwen-Sharp Terseness

- **ID:** `EXP-02`
- **Cluster:** `11 Research frontier` / `local-models-evaluation` / `rp-02ba98a842`
- **Node Alvo:** `aaaaa` (RTX 3090 / `slop.cpp`)
- **Status:** `ACTIVE`
- **Prerequisites:** [EXP-01](file:///C:/projects/tare.tools.research/experiments/local-llm/EXP-01-local-llm-runtime-qualification.md), [ADR-048](file:///C:/projects/tare.tools.os/docs/ADR-048_LOCAL_INFERENCE_SUBSTRATE_AND_AGENT_HARNESS.md)
- **Author:** Antigravity Mediator

---

## 1. Hipótese & Motivação (Hypothesis)

> **Hipótese:** A retenção da cadeia de raciocínio `<think>` para $K=1$ turnos anteriores combinada com o chat template Jinja *Qwen-Sharp* (remoção agressiva de preâmbulos e cortes de tokens de preenchimento) mantém $R_{\text{cache}} \ge 0.90$ nas interações subsequentes e reduz em $\ge 50\%$ a contagem total de tokens gerados sem degradação da precisão de código ou quebra do formato JSON de `tool_calls`.

---

## 2. Métricas & Variáveis (Metrics)

| Métrica | Descrição | Unidade | Target de Sucesso |
| :--- | :--- | :--- | :--- |
| **`R_cache`** | Razão de tokens reutilizados do KV-cache (`prompt_tokens_cached / total_prompt_tokens`) | ratio | $\ge 0.90$ ($90\%$) |
| **`Terseness_Ratio`** | Redução percentual de tokens de saída gerados vs baseline | % | $\ge 40\%$ |
| **`ToolCall_Validity`** | Taxa de chamadas de ferramenta com JSON estritamente válido | % | $100.0\%$ |
| **`KV_VRAM_Usage`** | Memória VRAM consumida pelo buffer KV com $K=1$ compaction | GB | $\le 4.5\text{ GB}$ em 32k context |

---

## 3. Baseline de Comparação (Baseline)

- **Baseline:** Chat template padrão sem truncamento de preâmbulo e sem compactação de raciocínio histórico ($K=\infty$).
  - Baseline $R_{\text{cache}}$: Oscilante entre 0.30 e 0.70 por quebra de prefixo.
  - Baseline Output Tokens por turno: ~1,200 tokens (incluindo fillers conversacionais).

---

## 4. Critérios de Sucesso (Success Criteria)

1. $R_{\text{cache}} \ge 0.90$ sustentado em sessões de diálogo multi-turno ($\ge 5$ turnos).
2. Queda de $\ge 40\%$ no número de tokens gerados por resposta comparado ao template padrão.
3. 100% de conformidade de parser no JSON Schema de `tools` e `tool_calls`.

---

## 5. Critérios de Engavetamento / Abandono (Abandon Criteria)

- Queda na acurácia sintática de código $> 5\%$ devido a cortes excessivos de contexto.
- Desalinhamento de prefixo no `llama-server` forçando re-avaliação completa de prompt em cada turno ($R_{\text{cache}} < 0.50$).

---

## 6. Plano de Reversão (Reversal Plan)

- Desativar a remoção de preâmbulo no template Jinja e restaurar $K=\infty$ (histórico completo).
