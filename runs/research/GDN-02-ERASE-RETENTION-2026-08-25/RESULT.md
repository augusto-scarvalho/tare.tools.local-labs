# GDN-02 Gated DeltaNet-2 Erase & Retention - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A porta de esquecimento seletivo do DeltaNet-2 suprimiu com sucesso o fato obsoleto (**vazamento de apenas 2.84%**), mas a capacidade finita do estado matricial ($64 \times 64$) limitou a fidelidade de retenção colateral após 50 fatos contínuos a **65.31%**, demonstrando a necessidade de camadas de atenção plena híbridas para retenção associativa perpétua.

---

## 🎯 1. Resumo Executivo

O experimento testou a capacidade de esquecimento e atualização cirúrgica de fatos associativos em memórias matriciais recorrentes ($64 \times 64$) comparando três mecanismos através de [`tools/probes/gdn02_erase_retention_lab.py`](../../tools/probes/gdn02_erase_retention_lab.py).

A hipótese de preservação perfeita em alta carga associativa foi **FALSIFICADA**:
- O **Gated DeltaNet-2** provou ser altamente eficaz na eliminação do fato antigo alvo, reduzindo o resíduo para **2.84%** (contra 45.0% no decaimento estático e 13.5% no DeltaNet clássico).
- No entanto, a sobreposição linear após 50 fatos consecutivos provocou interferência cumulativa no espaço vetorial $\mathbb{R}^{64}$, limitando a retenção das outras 49 chaves a **65.31%** e a fidelidade de atualização a **73.32%**.

---

## 📊 2. Tabela de Comparação de Mecanismos Recorrentes (50 Fatos)

| Mecanismo Recorrente | Vazamento do Fato Antigo | Fidelidade da Atualização | Retenção Colateral (49 Fatos) | Cosseno Médio | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **`STATIC_DECAY_SSM`** | 45.00% (Alto Vazamento) | 61.40% | 61.22% | 0.7150 | `REJECTED` |
| **`CLASSIC_DELTANET`** | 13.50% | 70.20% | 71.43% | 0.7741 | `REJECTED` |
| **`QUERY_GATED_DELTANET2`**| **2.84% (PASS VAZAMENTO)** | **73.32% (FAIL GATE)** | **65.31% (FAIL GATE)** | **0.7620** | `REJECTED` |

---

## 🔬 3. Diretriz de Arquitetura

1. **Associação Híbrida Obrigatória**:
   - Estados recorrentes $O(1)$ são excelentes para fluxo e raciocínio sequencial local, mas não devem ser encarregados de armazenar dezenas de fatos associativos densos sem o suporte das camadas de atenção plena do transformador (`SLX-11`).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/GDN-02-ERASE-RETENTION-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/gdn02_erase_retention_lab.py`](../../tools/probes/gdn02_erase_retention_lab.py)
- **Agente Executor**: Antigravity
