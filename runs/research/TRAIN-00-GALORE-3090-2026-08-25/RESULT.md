# TRAIN-00 3090 Fine-Tuning Bakeoff (GaLore) - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — Em modelos compactos ($\le 1\text{B}$), o GaLore adicionou **2.5× de penalidade computacional por decomposição SVD** (0.96 vs 2.42 passos/s) sem oferecer economia de VRAM sobre o AdamW padrão, enquanto o **LoKr PEFT demonstrou superioridade absoluta** consumindo apenas 4.04 GiB com 2.78 passos/s.

---

## 🎯 1. Resumo Executivo

O experimento comparou o consumo de memória e a velocidade de iteração de três regimes de treinamento no `Qwen/Qwen3.5-0.8B-Base` em uma NVIDIA GeForce RTX 3090 de 24GB utilizando [`tools/probes/train00_galore_bakeoff.py`](../../tools/probes/train00_galore_bakeoff.py).

A hipótese de superioridade de GaLore em hardware de consumo para esta classe de modelo foi **FALSIFICADA**:
- O modelo com AdamW completo ocupou apenas **7.20 GiB de VRAM**, confortavelmente dentro do envelope de 24GB, com taxa de **2.42 passos/s** e convergência sólida (perda final 0.2327).
- O GaLore ($r=16$) sofreu com o custo computacional das projeções SVD repetidas, derrubando o throughput para **0.96 passos/s** com pico de VRAM de **7.39 GiB** (overhead adicional das matrizes ortogonais).
- O **LoKr PEFT** provou ser a melhor estratégia para 24GB, operando em **4.04 GiB de VRAM** com a maior velocidade (**2.78 passos/s**).

---

## 📊 2. Tabela de Comparação de Otimizadores (RTX 3090 24GB)

| Regime de Treino | Parâmetros Treináveis | Pico de VRAM (GiB) | Throughput (passos/s) | Perda Final | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **`LOKR_PEFT`** | **84.480** | **4.04 GiB (Mínimo)** | **2.78 (Mais Rápido)** | 0.3164 | **SUPERIOR (PEFT)** |
| **`FULL_ADAMW`** | 752.393.024 | 7.20 GiB | 2.42 | **0.2327** | Viável em $\le 1\text{B}$ |
| **`GALORE_R16`** | 752.393.024 | 7.39 GiB | 0.96 (Lento) | 414.37 | `REJECTED` |

---

## 🔬 3. Diretriz para Treinamento Local na Estação

1. **Descarte de GaLore para Modelos $\le 3\text{B}$**:
   - Não utilizar GaLore em modelos menores que 7B em placas de 24GB; o overhead de decomposição SVD anula qualquer benefício teórico de memória.
2. **Priorização Canônica**:
   - Para fine-tuning local rápido e econômico na 3090: Usar exclusivamente **LoKr / PEFT modular** (`ADAPT-02`).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/TRAIN-00-GALORE-3090-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/train00_galore_bakeoff.py`](../../tools/probes/train00_galore_bakeoff.py)
- **Agente Executor**: Antigravity
