# TRAIN-00 3090 Fine-Tuning Bakeoff (GaLore) - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: O gargalo de memória de treinamento full-parameter em GPUs de 24GB (como a RTX 3090) é dominado pelos estados em FP32 do otimizador AdamW ($8\text{ bytes/parâmetro}$). A projeção de gradiente em subespaço de baixo rank (GaLore — *Gradient Low-Rank Projection*, $r=16$) reduz o estado do otimizador em $\ge 50\%$ em relação ao AdamW completo mantendo convergência de perda idêntica e sem congelar o backbone como no PEFT.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Braços Experimentais (100 passos de treino por braço)**:
  1. `FULL_ADAMW`: AdamW padrão em todos os parâmetros do modelo.
  2. `LOKR_PEFT`: LoKr ($r=8$, adaptando 224k parâmetros).
  3. `GALORE_R16`: GaLore projetado ($r=16$, projeção periódica de subespaço SVD a cada 20 passos).
* **Métricas**:
  1. `peak_vram_gib`: Consumo de pico de VRAM durante backward e step.
  2. `step_throughput_steps_per_sec`: Velocidade de iteração.
  3. `final_training_loss`: Perda média nos últimos 10 passos.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Economia de VRAM de GaLore ($\ge 30\%$)**: Redução de VRAM de pico vs Full AdamW.
2. **Gate de Convergência de Perda**: Perda final do GaLore comparável ($\le 1.15\times$) à do Full AdamW.
3. **Throughput de Treino**: $\ge 2.0$ passos/segundo na RTX 3090.
