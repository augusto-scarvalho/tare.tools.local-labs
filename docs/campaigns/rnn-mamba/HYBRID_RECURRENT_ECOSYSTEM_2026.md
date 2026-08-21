# 🌐 Panorama Global de Modelos Híbridos, RNNs e Estado Recorrente (2025–2026)

**Data:** 2026-08-20  
**Escopo:** Auditoria de novos papers, PRs do `llama.cpp`, forks especializados e discussões da comunidade sobre caching de memória híbrida, Gated DeltaNet, Mamba-2 e linearização.

---

## 🏛️ 1. O Estado da Arte em Papers e Arquiteturas (2025–2026)

| Trabalho / Arquitetura | Conferência / Data | Autores / Repo | Mecanismo Central | Impacto para o Local Lab |
|---|---|---|---|---|
| **Gated DeltaNet-2 (GDN-2)** | Maio 2026 | NVlabs / `GatedDeltaNet-2` | **Decoupled Erase & Write**: Separação matemática dos portões de apagamento e escrita na regra delta, eliminando a perda precoce de informação sob interferência. | Próximo passo natural para o nosso substrato GDN (`rnn_delta_substrate.py`). |
| **Memory Caching (Growing Memory)** | **ICML 2026** | Behrouz et al. (Google) / arXiv:2602.24281 | **Checkpoints de Estado + GRM/SSC**: Arquivamento de snapshots do estado matricial $S_{L^{(i)}}$ e leitura via Gated Residual Memory / Top-K router sem alterar a arquitetura base. | Base da nossa especificação [`RNN_MEMORY_CACHING_SPEC.md`](file:///C:/projects/tare.tools.local-labs/docs/campaigns/rnn-mamba/RNN_MEMORY_CACHING_SPEC.md). |
| **In-Place TTT** | **ICLR 2026 (Oral)** | ByteDance Seed / `In-Place-TTT` | **Fast Weights em Projeções MLP**: Test-Time Training adaptando as matrizes `down_proj` em tempo de inferência sem re-treinar a atenção densa. | Viável para adaptação local na RTX 3090 em modelos 4B/8B. |
| **Liger (Linearization)** | **ICML 2025** | OpenSparseLLMs / ICML | **Reaproveitamento de Matrizes de Chave ($W_k$)**: Converte camadas de atenção densa em recorrência linear sem novos parâmetros; recupera 93% de acurácia com 0.02% dos tokens. | Estratégia número 1 para linearizar camadas do Qwen localmente. |
| **Titans: Learning to Memorize** | **NeurIPS 2025** | Google Research / Behrouz | **Memória Neural de Longo Prazo**: Fast weights com momentum + gating + atenção para memória de trabalho de curto prazo. | Inspiração para arquiteturas de contexto ultra-longo (>2M tokens). |
| **RWKV-7 ("Goose")** | 2025–2026 | BlinkDL / `RWKV-LM` | **Recorrência Não-Linear de Estado Constante**: Nova geração da família RWKV com alta expressividade sem KV cache. | Integrado ao Flash-Linear-Attention (FLA). |
| **YOCO (You Only Cache Once)** | 2024–2025 | Microsoft Research | **KV Cache em Camada Única**: Camadas inferiores usam recorrência linear e apenas as superiores geram KV cache denso. | Valida a tese híbrida do Qwen (1/4 full attention, 3/4 GDN). |

---

## 🛠️ 2. Ecossistema `llama.cpp`: PRs, Issues e Forks Especializados

### Upstream `ggml-org/llama.cpp`
1. **Mamba-2 Oficial:**
   - Mergeado via PR `#9126` (`ggml_ssm_scan` e kernels CUDA dedicados).
2. **Stateful Inference API (Issue / PR `#23817`):**
   - Esforço ativo para criar uma API nativa no `llama.cpp` capaz de **serializar, duplicar (fork) e restaurar estados recorrentes** de forma transparente ao lado do KV cache tradicional.
3. **Bug Crítico de "State Drift" em Modelos Híbridos (Issues `#21681` e `#22384`):**
   - Problema documentado onde o reuso de prompt-cache em modelos híbridos (GDN / Mamba) causa corrupção silenciosa ou deriva de estado porque o engine padrão não gerenciava o ciclo de vida do buffer recorrente entre requisições.

### Forks Especializados da Comunidade
- **`CachyLLama`:** Fork comunitário especializado em fluxos agentic e **KV Caching Persistente** para modelos híbridos (especificamente ajustado para Qwen 3.5 / 3.6 GDN).
- **`flash-linear-attention (FLA)`:** Hub upstream com kernels Triton/CUDA otimizados para GDN, GDN-2, RWKV-7 e Mamba3, com suporte a Context Parallelism (CP).

---

## 🧭 3. Síntese Estratégica para o Laboratório Local

A literatura e a comunidade técnica confirmam três conclusões cruciais que guiam o nosso laboratório:

1. **A Atenção Híbrida (GDN + Full Attention) é a Vencedora do Pareto:**
   Modelos como Qwen 3.5/3.6 (com 75% GDN e 25% Atenção) atingem o equilíbrio perfeito: economizam 75% de VRAM no KV cache enquanto mantêm 100% de recuperação de agulhas críticas.
2. **Memory Caching com Gating Ativo (GRM/SSC) é a Solução para o Limite das RNNs:**
   Em vez de tentar forçar a RNN a lembrar de tudo em um único estado $O(1)$, arquivar checkpoints de estado $S_{L^{(i)}}$ e ler via roteador leve $W_u$ resolve o colapso de capacidade sem incorrer no custo quadrático da atenção densa.
3. **Linearização Local via Liger:**
   Podemos converter camadas de atenção densa do Qwen em blocos lineares rápidos reaproveitando as próprias projeções já aprendidas pelo modelo.

---

## 🤗 4. Mapeamento de Checkpoints e Hubs no Hugging Face

Auditamos as principais organizações e modelos disponíveis no Hugging Face para recorrência linear e modelos híbridos:

| Organização / Hub | Checkpoints Relevantes | Arquitetura / Mecanismo | Uso no Laboratório |
|---|---|---|---|
| **`Idiap`** | `Idiap/gated-deltanet-attn-1.4B-30B`<br>`Idiap/gated-deltanet-lte-0.4B-10B` | **Gated DeltaNet + Attention Híbrido** e Gated DeltaNet com Learnable Token Eviction (LTE). | Modelo pré-treinado aberto ideal para testar Memory Caching diretamente em pesos reais de GDN híbrido. |
| **`fla-hub`** | `fla-hub/delta_net-1.3B-100B`<br>`fla-hub/delta_net-2.7B-100B` | DeltaNet Linear Attention via kernels FLA Triton. | Substrato de referência testado em `RNN-06-P0`. |
| **`ByteDance-Seed`** | `ByteDance-Seed/In-Place-TTT` | In-Place Test-Time Training em Qwen / LLaMA. | Adaptação de fast-weights em inferência sem re-treino. |
| **`HazyResearch`** | `HazyResearch/lolcats` | Low-Rank Linear Conversion via Attention Transfer + LoRA. | Pipeline de linearização de modelos Transformer densos. |
| **`recursal`** | `recursal/RADLADS-paper` | Checkpoints destilados de Qwen2.5 para RWKV-6 / RWKV-7. | Conversão direta de modelos Qwen para decodificadores lineares. |
| **`jxiw`** | `jxiw/MambaInLlama` | Checkpoints 3B e 8B destilados de LLaMA para híbrido Mamba-SSM + Atenção. | Referência de reaproveitamento de projeções de atenção para blocos SSM. |
| **`BlinkDL`** | `BlinkDL/rwkv-7-world` | Checkpoints oficiais da arquitetura RWKV-7 ("Goose"). | Testes de expressividade sem KV cache. |
| **`Qwen`** | `Qwen/Qwen3.5-0.8B`<br>`Qwen/Qwen3.6-27B` | Arquitetura nativa oficial híbrida Gated DeltaNet (75% GDN + 25% Gated Attention). | Alvo principal de produção e deploy do laboratório. |

