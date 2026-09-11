# 🧪 TARE.TOOLS LOCAL LABS — BACKLOG DE EXPERIMENTOS DE HARDWARE & MODELOS

> **Governança:** [ADR-048: Substrato de Inferência Local, Harness Agêntico & Aceleração de Hardware](file:///C:/projects/tare.tools.os/docs/ADR-048_LOCAL_INFERENCE_SUBSTRATE_AND_AGENT_HARNESS.md)  
> **Nó Alvo de Execução:** Workstation `aaaaa` (NVIDIA GeForce RTX 3090 24GB VRAM / WSL2 Ubuntu 24.04 CUDA)  
> **Endpoint Tailscale:** `http://100.107.245.30:8080/v1`

---

## 📋 1. Catálogo dos 5 Experimentos Empíricos Priorizados

| Card ID | Título do Experimento | Escopo & Hipótese de Validação | Métrica de Sucesso Principal | Status |
| :---: | :--- | :--- | :--- | :---: |
| [`EXP-01`](file:///C:/projects/tare.tools.research/experiments/local-llm/EXP-01-local-llm-runtime-qualification.md) | **Runtime Qualification & CUDA Levers** | Validação das 4 alavancas CUDA do `slop.cpp` (`[B2b]` DMA pinning, prefetch skip-staging, MoE hot-cache e MTP spec draft $n=3$). | $\ge 600\text{ t/s}$ prefill, $\ge 30\text{ t/s}$ decode, zero OOM em 64k. | **Ativo** |
| [`EXP-02`](file:///C:/projects/tare.tools.research/experiments/local-llm/EXP-02-thinking-retention-kv-cache.md) | **Thinking Retention (K=1) & Qwen-Sharp** | Retenção do bloco `<think>` no turno imediato para preservar KV-cache hit multi-turno com template anti-preâmbulo. | $R_{\text{cache}} \ge 0.90$, redução de $\ge 50\%$ em tokens de padding. | **Ativo** |
| [`EXP-03`](file:///C:/projects/tare.tools.research/experiments/local-llm/EXP-03-locality-aware-placement.md) | **Locality-Aware Placement & Zero-Cost** | Roteamento de tarefas para a 3090 com failover determinístico para a nuvem ($T_1$) após $N_{\text{total}} \le 5$. | $\ge 70\%$ de tarefas resolvidas a custo zero (\$0/token). | **Ativo** |
| [`EXP-04`](file:///C:/projects/tare.tools.research/experiments/local-llm/EXP-04-sandboxed-execution-isolation.md) | **3-Process Sandboxed Test Harness** | Topologia de isolamento do AiderDriver no host com suíte de testes envelopada em sandbox Bubblewrap (`--unshare-net`). | $100\%$ de contenção de egress e de segredos do host. | **Ativo** |
| [`EXP-05`](file:///C:/projects/tare.tools.research/experiments/local-llm/EXP-05-scheduling-admission-backpressure.md) | **Scheduling Admission & Backpressure** | Telemetria contínua de headroom de VRAM e controle de concorrência com fila assíncrona. | Zero travamentos por concorrência e cushion de $\ge 3\text{ GB}$ VRAM. | **Ativo** |

---

## 🚀 2. Como Executar os Experimentos a partir do Notebook `acer`

```powershell
# 1. Verificar se o nó aaaaa está online na rede Tailscale:
python C:\projects\tare.tools.os\scripts\run_local_lab_experiments.py --status

# 2. Executar a suíte completa de experimentos contra o slop.cpp no aaaaa:
python C:\projects\tare.tools.os\scripts\run_local_lab_experiments.py --all

# 3. Executar um experimento isolado:
python C:\projects\tare.tools.os\scripts\run_local_lab_experiments.py --exp EXP-01

# 4. Validar os contratos em modo mock hermético offline:
python C:\projects\tare.tools.os\scripts\run_local_lab_experiments.py --mock
```

---

## 🛡️ 3. Regra de Não-Desvio

- Este backlog trata exclusivamente de **aferição física de modelos, hardware e telemetria**.
- A execução desses experimentos **não cria trains de software**, não gera pacotes em monorepo e não altera arquivos de produção dos repositórios satélites.
