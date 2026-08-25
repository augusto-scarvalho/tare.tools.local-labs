# HYPER-01 Hypernetworks for Capsules - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A HyperNetwork sintetizou adaptadores LoRA sob demanda com **altíssima velocidade (0.087 ms/tarefa)** e **alta fidelidade ($\text{Cosine Sim} = 0.96203$)**, mas o tamanho físico do gerador (**32.63 MB / 8.55M parâmetros**) violou o gate de pegada ($\le 20\text{ MB}$), sendo desvantajoso para frotas de pequeno/médio porte ($\le 100$ tarefas) em relação a adaptadores estáticos (64 KB).

---

## 🎯 1. Resumo Executivo

O experimento avaliou a síntese dinâmica de pesos de adaptadores LoRA ($A$ e $B$ com rank $r=8$) condicionados por embeddings de tarefa $\mathbf{z} \in \mathbb{R}^{64}$ na RTX 3090 através de [`tools/probes/hyper01_capsule_generator.py`](../../tools/probes/hyper01_capsule_generator.py).

A hipótese de viabilidade de armazenamento e síntese foi **FALSIFICADA PARA FROTAS PEQUENAS**:
- A velocidade de síntese na GPU foi quase instantânea (**0.087 ms por adaptador**), e os deltas gerados alcançaram **0.96203 de similaridade** com os adaptadores alvo.
- No entanto, a projeção direta para $16.384$ pesos de saída exigiu **8.55 milhões de parâmetros (32.63 MB)**.
- Para uma frota de 4 a 16 especialistas, manter adaptadores LoRA estáticos consome apenas **64 KB a 256 KB**, tornando a HyperNetwork 500× mais pesada em VRAM.

---

## 📊 2. Tabela de Métricas da HyperNetwork (4 Tarefas)

| Métrica | Valor Observado | Meta / Gate | Veredito |
|---|:---:|:---:|:---:|
| **Latência de Síntese por Adaptador** | **0.087 ms (87 µs)** | $\le 5.0\text{ ms}$ | **PASS** |
| **Similaridade Cosseno dos Pesos** | **0.96203** | $\ge 0.950$ | **PASS** |
| **Pegada de Memória da HyperNetwork** | **32.63 MB** | $\le 20.0\text{ MB}$ | `REJECTED (FAIL GATE)` |
| **Parâmetros do Gerador** | 8.553.216 | — | — |

---

## 🔬 3. Diretriz de Decisão

1. **Rejeição de HyperNetworks para Frotas Locais Pequenas**:
   - Manter arquivos estáticos de adaptadores LoKr/LoRA em disco e carregá-los sob demanda via `SLOP-L1..L7`.
   - Reavaliar HyperNetworks apenas para cenários com $> 500$ habilidades parametrizadas em tempo real.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/HYPER-01-CAPSULES-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/hyper01_capsule_generator.py`](../../tools/probes/hyper01_capsule_generator.py)
- **Agente Executor**: Antigravity
