# HYPER-01 Hypernetworks for Capsules - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Uma HyperNetwork compacta (MLP de 2 camadas com $\le 1.5\text{M}$ parâmetros) condicionada em vetores de embedding de tarefa ($\mathbf{z} \in \mathbb{R}^{64}$) consegue sintetizar os pesos de adaptadores LoRA de rank $r=8$ sob demanda em $< 2.0\text{ ms}$ na GPU, alcançando **$\ge 90\%$ de fidelidade de aproximação dos pesos estáticos** e viabilizando a geração contínua de milhares de habilidades sem sobrecarga de I/O em disco.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Arquitetura da HyperNetwork**:
  - Entrada: Vetor de metadados da tarefa $\mathbf{z} \in \mathbb{R}^{64}$.
  - Backbone: MLP de 2 camadas ($64 \rightarrow 256 \rightarrow 1024 \times 8$).
  - Saída: Fatores LoRA $A \in \mathbb{R}^{1024 \times 8}$ e $B \in \mathbb{R}^{8 \times 1024}$.
* **Domínios Avaliados**: 4 tarefas sintéticas (Matemática, Código, Formatação JSON, QA Factual).
* **Métricas**:
  1. `synthesis_latency_ms`: Tempo de inferência da HyperNetwork para gerar um adaptador completo.
  2. `adapter_weight_cosine_sim`: Similaridade de cosseno entre os pesos gerados e os pesos alvo estáticos.
  3. `hypernetwork_vram_overhead_mb`: Pegada de memória da HyperNetwork.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Latência de Síntese ($\le 5.0\text{ ms}$)**: Geração de adaptador em tempo de voo.
2. **Gate de Fidelidade de Peso ($\ge 0.950$)**: $\text{Cosine Sim} \ge 0.950$ contra os adaptadores alvo.
3. **Pegada de Memória ($\le 20\text{ MB}$)**: Tamanho da HyperNetwork em VRAM.
