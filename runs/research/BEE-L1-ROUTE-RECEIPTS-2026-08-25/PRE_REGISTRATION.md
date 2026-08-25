# BEE-L1 Effective Route Receipts - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em runtimes heterogêneos de inferência (como `slop.cpp`, `llama.cpp` e PyTorch CUDA), a discrepância entre a configuração solicitada na CLI/serviço e a rota física executada na GPU é a principal causa de *falsos verdes* (benchmarks que passam sem erro, mas rodam em fallback silencioso de CPU ou com buffers desabilitados). Implementar um verificador determinístico com contrato formal dos 4 níveis de ciclo de vida (`requested` $\rightarrow$ `resolved` $\rightarrow$ `realized` $\rightarrow$ `exercised`) detecta regressões silenciosas de rota e emite recibos de auditoria imutáveis com checksum criptográfico.

---

## 🎯 1. Contrato dos 4 Níveis de Ciclo de Vida

1. **`requested`**: Flags, argumentos e políticas passados pelo usuário/orquestrador (ex: `--ctk q4_0`, `--flash-attn`, `n_gpu_layers 99`).
2. **`resolved`**: Parâmetros resolvidos pelo grafo de inicialização após inspeção de hardware e capacidades (ex: backend CUDA selecionado, split de camadas offloaded).
3. **`realized`**: Alocação física verificada em VRAM/Host (ex: buffers de KV criados no device com formato de precisão conferido via props do runtime).
4. **`exercised`**: Confirmação empírica de execução de kernels no hardware (ex: contagem não-nula de tokens decodificados pela rota sem ativação de fallback).

---

## 🛑 2. Critérios de Promoção e Testes

1. **Detecção de Fallback Silencioso**: Identificar e falhar imediatamente quando `requested != resolved` ou `resolved != realized`.
2. **Verificação de Runtime Ativo**: Auditar as rotas dos serviços de produção locais (`llm-inference.service` na 8080 e `llm-embedding.service` na 8081).
3. **Passagem na Suite de Testes**: 100% de cobertura nos cenários canônicos (rota válida, fallback de KV, VRAM overflow simulado, mismatch de backend).
