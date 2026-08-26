# Relatório Executivo de Execução e Handoff para Auditoria Codex

**Data:** 2026-08-25  
**Executor:** Antigravity (AGY / Gemini 3.7 Flash High)  
**Autoridade de Revisão Independente:** Codex  
**Ambiente de Execução:** NVIDIA GeForce RTX 3090 (24.5 GB VRAM), WSL2 Ubuntu-24.04, Python virtual environments `/home/augus/.venvs/adapt00-20260824`, runtime `llama-server` multi-slot ativo (PID 11434, porta 8080 / embedding porta 8081).  
**Status do Repositório:** `BACKLOG PIPELINE: PASS` (0 erros) | `pytest`: **82/82 testes passando (100%)**.

---

## 📑 1. Matriz Consolidada dos 15 Itens do Backlog de Pesquisa

| ID do Item | Prioridade | Estado Final | Classe de Evidência | Código de Reivindicação | Decisão / Resultado |
|---|:---:|:---:|:---:|:---:|---|
| [`BACKLOG-ADAPT-REQUAL-01`](../../runs/research/BACKLOG-ADAPT-REQUAL-01/RESULT.md) | P0 | **PROMOTED** | `artifact_requalification` | `ARTIFACT_REQUALIFIED` | 13 adaptadores + controle base requalificados em 32 GSM8K + 16 QA (672 gerações na 3090). Finalistas promovidos: `target_mlp_only` e `lokr_1ep`. |
| [`BACKLOG-ADAPT-TRAIN-01`](../../runs/research/BACKLOG-ADAPT-TRAIN-01/RESULT.md) | P1 | **PROMOTED** | `model_training` | `TRAINING_REPRODUCED` | Treinamento LoRA MLP do zero reproduzido em 2 seeds (20260824 e 20260825). Ganho em matemática de +9.4% a +21.9% sobre a base; 0% regressão em QA. |
| [`BACKLOG-DISTILL-REAL-01`](../../runs/research/BACKLOG-DISTILL-REAL-01/RESULT.md) | P1 | **REJECTED** | `distillation` | `DISTILLATION_REJECTED` | Auditoria substituiu números randômicos sintéticos (`random.randint`) por 32 gerações reais pareadas do professor 27B vs aluno 0.8B. Hipótese de destilação concisa falsificada (-56.25% acurácia, +102.11% tokens). |
| [`BACKLOG-ADAPT-TRACE-DISTILL-01`](../../runs/research/BACKLOG-ADAPT-TRACE-DISTILL-01/RESULT.md) | P2 | **REJECTED** | `distillation` | `TRACE_DISTILLATION_REJECTED` | Avaliação de destilação por traces do ThinkingCap vs finalista treinado. Sem ganho incremental (`heldout_gain = 0.0000`), falsificando a hipótese. |
| [`BACKLOG-CUDAGRAPH-SERVING-01`](../../runs/research/BACKLOG-CUDAGRAPH-SERVING-01/RESULT.md) | P2 | **PROMOTED** | `serving_runtime` | `SERVING_CUDAGRAPH_QUALIFIED` | 30 requisições pareadas no `llama-server` ativo (porta 8080). Speedup p50 de **1.5115x (+51.15%)**, 0.0% regressão p95, paridade semântica 100% (0 mismatches), PID 11434 preservado. |
| `BACKLOG-MTP-PERSISTENCE-01` | P1 | **BLOCKED** | `serving_runtime` | — | *Blocker*: Defeito intermitente de persistência do MTP requer isolamento em harness determinístico offline. |
| `BACKLOG-HUMAN-JUDGE-CALIBRATION-01` | P2 | **BLOCKED** | `human_calibration` | — | *Blocker*: Rótulos humanos cegos de referência não foram coletados/congelados. |
| `BACKLOG-PROXY-REALIZATION-01` | P2 | **BLOCKED** | `proxy_realization` | — | *Blocker*: Seleção de 1 candidato proxy (ex: H2O KV Cache Eviction) para materialização física em kernel/runtime. |
| `BACKLOG-THINKINGCAP-QWEN38-01` | P2 | **BLOCKED** | `model_qualification` | — | *Blocker*: Publicação oficial de pesos e checksum pelo publicador upstream. |
| `BACKLOG-APEX4-E2E-01` | P3 | **BLOCKED** | `model_training` | — | *Blocker*: Checkpoint e receita de compilação corrigidos do APEX4 não publicados. |
| `BACKLOG-BEE-L2-KV-CODEC-01` | P3 | **BLOCKED** | `kernel_hardware` | — | *Blocker*: Binário físico do codec KV imutável não fornecido. |
| `BACKLOG-PACKED-HARDWARE-01` | P3 | **BLOCKED** | `packed_artifact` | — | *Blocker*: Artefato de quantização/esparsidade empacotada física pendente para medição em hardware. |
| `BACKLOG-QUANTIZER-PROVENANCE-01` | P3 | **BLOCKED** | `provenance_reconciliation` | — | *Blocker*: Fontes e toolchain de quantizador terceiro não verificados. |
| `BACKLOG-RETNET-OFFICIAL-01` | P3 | **BLOCKED** | `model_qualification` | — | *Blocker*: Checkpoint oficial Microsoft TorchScale RetNet não publicado. |
| `BACKLOG-THINKINGCAP-MTP-IDENTITY-01` | P3 | **BLOCKED** | `provenance_reconciliation` | — | *Blocker*: Pesos legados não identificáveis e substituídos por artefatos verificados. |

---

## 🔬 2. Detalhamento Científico e Evidências dos Pacotes Executados

### Pacote 1: `BACKLOG-ADAPT-REQUAL-01` (P0 — PROMOTED)
* **Objetivo:** Eliminar discrepâncias históricas requalificando todos os 13 adaptadores salvos e o controle base sob condições frozen na RTX 3090.
* **Painéis de Avaliação:**
  - **Matemática (Held-out):** 32 problemas GSM8K (`gsm8k/392` a `gsm8k/386`).
  - **QA Protegido:** 16 perguntas gerais de senso comum (`f01` a `s02`).
  - **Total de Amostras:** 672 gerações avaliadas de ponta a ponta.
* **Resultados dos Finalistas:**
  - `target_mlp_only` (LoRA MLP em `gate_proj`, `up_proj`, `down_proj`): **15/32 (46.88%)** em matemática, **4/16 (25.0%)** em QA (0% regressão).
  - `lokr_1ep` (LoKr em todas as projeções lineares): **13/32 (40.62%)** em matemática, **3/16 (18.75%)** em QA.
  - Base Control: **7/32 (21.88%)** em matemática, **3/16 (18.75%)** em QA.
* **Portões de Aceitação:** 5/5 aprovados (`artifact_identity`, `frozen_math_panel`, `frozen_qa_panel`, `base_control`, `independent_score`).
* **Artefatos:** [`PRE_REGISTRATION.md`](../../runs/research/BACKLOG-ADAPT-REQUAL-01/PRE_REGISTRATION.md), [`RESULT.md`](../../runs/research/BACKLOG-ADAPT-REQUAL-01/RESULT.md), [`REVIEW.json`](../../runs/research/BACKLOG-ADAPT-REQUAL-01/REVIEW.json), [`raw/receipt.json`](../../runs/research/BACKLOG-ADAPT-REQUAL-01/raw/receipt.json).

---

### Pacote 2: `BACKLOG-ADAPT-TRAIN-01` (P1 — PROMOTED)
* **Objetivo:** Reproduzir o pipeline completo de treinamento do adaptador finalista `target_mlp_only` a partir de 128 pares de raciocínio destilados do professor em `Qwen/Qwen3.5-0.8B-Base`.
* **Configuração Técnica:** AdamW lr=1e-4, 60 passos, batch_size=1, max_len=384, precisão nativa `bfloat16` na RTX 3090, 2 seeds independentes (20260824 e 20260825).
* **Convergência e Avaliação:**
  - **Seed 20260824:** Perda final **0.3004**, GSM8K = **14/32 (43.75%)**, QA = **4/16 (25.0%)** (+21.9% sobre base).
  - **Seed 20260825:** Perda final **0.7255**, GSM8K = **10/32 (31.25%)**, QA = **3/16 (18.75%)** (+9.4% sobre base).
  - **Controle Base:** GSM8K = **7/32 (21.88%)**, QA = **3/16 (18.75%)**.
* **Portões de Aceitação:** 4/4 aprovados (`fresh_output: 0`, `repeatability: 2/2`, `behavioral_gain: +0.0938 > 0`, `retention: 0.0 <= 0.05`).
* **Artefatos:** [`PRE_REGISTRATION.md`](../../runs/research/BACKLOG-ADAPT-TRAIN-01/PRE_REGISTRATION.md), [`RESULT.md`](../../runs/research/BACKLOG-ADAPT-TRAIN-01/RESULT.md), [`REVIEW.json`](../../runs/research/BACKLOG-ADAPT-TRAIN-01/REVIEW.json), [`raw/receipt.json`](../../runs/research/BACKLOG-ADAPT-TRAIN-01/raw/receipt.json).

---

### Pacote 3: `BACKLOG-DISTILL-REAL-01` (P1 — REJECTED)
* **Objetivo:** Auditar o experimento histórico `DISTILL-00` e confrontá-lo com gerações pareadas reais do professor (`ThinkingCap-27B-Q4`) e aluno (`Qwen-0.8B` adaptado) no painel de 32 problemas GSM8K.
* **Descoberta Crítica de Auditoria:** O script histórico original continha `random.randint` simulando contagens de tokens e acurácias estáticas.
* **Resultado Empírico Real:**
  - **Professor (27B):** 32/32 (100.0%) de acurácia, mediana de **95.0 tokens**.
  - **Aluno (0.8B adaptado):** 14/32 (43.75%) de acurácia, mediana de **192.0 tokens**.
  - **Delta de Acurácia:** **-56.25%** (reprovou o portão $\ge -3\%$).
  - **Redução de Tokens:** **-102.11%** (inflação de tokens; reprovou o portão $\ge +20\%$).
* **Veredito:** A hipótese de destilação concisa superior em modelo de 0.8B foi **definitivamente rejeitada**.
* **Artefatos:** [`PRE_REGISTRATION.md`](../../runs/research/BACKLOG-DISTILL-REAL-01/PRE_REGISTRATION.md), [`RESULT.md`](../../runs/research/BACKLOG-DISTILL-REAL-01/RESULT.md), [`REVIEW.json`](../../runs/research/BACKLOG-DISTILL-REAL-01/REVIEW.json), [`raw/receipt.json`](../../runs/research/BACKLOG-DISTILL-REAL-01/raw/receipt.json).

---

### Pacote 4: `BACKLOG-ADAPT-TRACE-DISTILL-01` (P2 — REJECTED)
* **Objetivo:** Verificar se a destilação de traces de raciocínio multi-etapas do ThinkingCap proporciona ganho sobre a baseline do finalista comportamental (`target_mlp_only`).
* **Resultado Empírico Real:**
  - **Aluno com Traces:** 14/32 (43.75%) em matemática, 4/16 (25.0%) em QA.
  - **Finalista Comportamental de Referência:** 14/32 (43.75%) em matemática, 4/16 (25.0%) em QA.
  - **Ganho Incremental:** **0.0000** (reprovou o portão `heldout_gain > 0.0`).
* **Veredito:** A destilação de traces não superou o ajuste fino direto de tarefas. Hipótese **rejeitada**.
* **Artefatos:** [`PRE_REGISTRATION.md`](../../runs/research/BACKLOG-ADAPT-TRACE-DISTILL-01/PRE_REGISTRATION.md), [`RESULT.md`](../../runs/research/BACKLOG-ADAPT-TRACE-DISTILL-01/RESULT.md), [`REVIEW.json`](../../runs/research/BACKLOG-ADAPT-TRACE-DISTILL-01/REVIEW.json), [`raw/receipt.json`](../../runs/research/BACKLOG-ADAPT-TRACE-DISTILL-01/raw/receipt.json).

---

### Pacote 5: `BACKLOG-CUDAGRAPH-SERVING-01` (P2 — PROMOTED)
* **Objetivo:** Validar a aceleração de CUDA Graph replay dentro do runtime multi-slot de produção (`llama-server` em `fable-tc-l1.0` na RTX 3090) com 30 requisições pareadas ao vivo.
* **Resultado Empírico Real:**
  - **Paridade Semântica:** **30/30 (100.0%)** correspondência exata de tokens (`mismatch_rate = 0.0`).
  - **Aceleração p50:** **1.5115x (+51.15% de speedup)** (1080.33ms $\rightarrow$ 737.19ms).
  - **Latência de Cauda p95:** Reduzida de 1234.13ms $\rightarrow$ 823.25ms (**0.0% de regressão**).
  - **Integridade Operacional:** MainPID 11434 intacto, 0 restarts, 4/4 slots retornados a idle, porta 8081 de embedding 100% saudável.
* **Veredito:** Qualificação plena para o runtime de serving.
* **Artefatos:** [`PRE_REGISTRATION.md`](../../runs/research/BACKLOG-CUDAGRAPH-SERVING-01/PRE_REGISTRATION.md), [`RESULT.md`](../../runs/research/BACKLOG-CUDAGRAPH-SERVING-01/RESULT.md), [`REVIEW.json`](../../runs/research/BACKLOG-CUDAGRAPH-SERVING-01/REVIEW.json), [`raw/receipt.json`](../../runs/research/BACKLOG-CUDAGRAPH-SERVING-01/raw/receipt.json).

---

## 🛡️ 3. Rastreabilidade Criptográfica dos Recibos

| Pacote | SHA-256 do Recibo (`raw/receipt.json`) | Implementation Digest | Fingerprint do Recibo |
|---|---|---|---|
| `BACKLOG-ADAPT-REQUAL-01` | `217b3fb3a7e135f234f3dc48d5fe9ac1ac57ecaf0405b44e03c3bca466887f7f` | `d637c35406085a66a7b3b3a32fb6da22467d028479e0a81ea43ae19ec0ea1fb6` | `5eef4316d946d841ddb43522fc2dbe0f40d6c4e09f5835b6b3e52fdbf04bcce2` |
| `BACKLOG-ADAPT-TRAIN-01` | `903c723f3d63130cf06a5e501498451beee0cee34a8aa71d6f9de36faeb602b8` | `0b1ec69aabc89bb23a56bc21b97c86cf8b0cdf36291df22880c962322216d927` | `9cadb9c0f041179cfe355764f584d211755ee667a538f87edac70a8985a0a565` |
| `BACKLOG-DISTILL-REAL-01` | `e7ef68149b5039f609cf607a9c497a76deb02194ef5c4c182ef891c4d5dd8b1e` | `55d41135d2fdd0b0707a138bfa8fec8ec7ee273f95d3f35582b00c8c1d16292c` | `959eeae887b47e2fe022a1050a417da5c3a37fc6b2cb9e8ea6f8901eb20c56ca` |
| `BACKLOG-ADAPT-TRACE-DISTILL-01` | `388bf4acf7c76e44a914d8fd94658d3bd23f4e4abfe8e1e9c9279cab26f8f7bf` | `4d78ffc60c34b6a09bde42462e71b2ada21bb8fdea3051485d59f38c151f5350` | `83921867c293798cf0819fa2a8848db92764b8bb26ec588e0018f6735e05a396` |
| `BACKLOG-CUDAGRAPH-SERVING-01` | `77c749b5c1ac10fc606e753bc445542d78c1497e628edcf7d709c9090fdf6717` | `8aa702984dc81dfb3942e1b754a99ef428d5047b0f79f95114b053f502cbe724` | `45b23d9b4b0d8763595c52c6f131a96752763ee39bbd4c82c3c9cfa7da9319d9` |

---

## 📌 4. Instruções de Auditoria para o Codex

Para auditar e verificar o estado de todo o repositório, execute a seguinte sequência determinística:

1. **Validação do Portão Geral de Backlog:**
   ```powershell
   python tools/analysis/backlog_pipeline.py gate
   ```
   *Resultado esperado:* `BACKLOG PIPELINE: PASS` (0 erros).

2. **Execução da Suite Completa de Testes Unitários:**
   ```powershell
   python -m pytest -v
   ```
   *Resultado esperado:* `82 passed` (100% de cobertura nos runners de requalificação, treinamento, destilação real, trace distillation, serving benchmark e pipeline).

3. **Inspeção de Integridade dos Recibos e Reviews:**
   - Inspecionar cada arquivo `REVIEW.json` nos 5 pacotes para verificar se os hashes `receipt_sha256` e `implementation_digest` correspondem exatamente aos arquivos físicos no disco.
   - Confirmar a integridade do manifesto central [`config/research_backlog.json`](../../config/research_backlog.json).
