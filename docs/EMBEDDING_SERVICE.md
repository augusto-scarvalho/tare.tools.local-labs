# Embedding service on node `aaaaa` (2026-09-07)

Qwen3-Embedding-4B serves the ecosystem's dense vectors. It replaced Nomic Embed v1.5,
which is English-centric and was never evaluated on Portuguese; on MTEB-PT (2026-07) the
Qwen3 family has the best retrieval among the models evaluated, and the 4B scores 69.45
on MTEB multilingual against 64.33 for the 0.6B. The 8B (70.58) costs twice the time, RAM
and index size for one point; both GGUFs are kept under `/home/augus/models/embedding/`.

> Integration note (2026-09-08): this is the recorded 2026-09-07 deployment report.
> This merge does not re-run its measurements, deploy services, or establish a new
> cross-quantization retrieval qualification. Select `LOCAL_EMBED_ENDPOINT` explicitly
> for the remote node; consult the current Library configuration for its default.

## Two servers, one model family, one vector namespace

| Server | Port | Quant | Mode | Why |
|---|---|---|---|---|
| `llm-embedding.service` (24/7) | 8081 | Q4_K_M | hybrid: weights in RAM, GPU compute per batch (`-ngl 0`) | queries and incremental indexing; holds ~0.9 GB VRAM with `ubatch` 512 |
| `~/ops/embedding_reindex_server.sh` (on demand) | 8082 | Q8_0 | `full` / `hybrid` / `cpu`, chosen by free VRAM | bulk (re)indexing at the fastest mode the GPU can spare |

Index and queries share the model family, so the Q8 index and Q4 queries are comparable:
measured cosine between Q8 and Q4 vectors of the same text is 0.98 on average, and the top-1
changed only on near ties (4/5 agreement on a 10-document probe). The vector namespace in
`library_vectors.db` is `qwen3-embedding-4b`; a different model family needs a new namespace.

Measured on the i7-13700K + RTX 3090 (`llama-bench`, pp512 unless noted):

| Mode | Q8 4B | notes |
|---|---|---|
| pure CPU, 8 threads (P-cores) | ~110 tok/s | 24 threads are slower: E-cores hurt; batching does not help |
| hybrid `-ngl 0`, ubatch 512 | ~1,100 tok/s | ~1.9 GB VRAM at ubatch 2048, 0.9 GB at 512 |
| hybrid `-ngl 0`, ubatch 4096, pp4096 | ~4,500 tok/s | batching pays: the per-batch weight copy is amortized |
| full GPU | ~10,000 tok/s | needs the LLM gateway stopped (`systemctl stop llm-inference.service`) |

Queries carry the Qwen3 instruction prefix (`Instruct: ... / Query: ...`); documents do not.

## Network

WSL2 runs in `mirrored` networking, so `0.0.0.0` listeners in WSL appear on the Tailscale
address `100.107.245.30`. Each port needs a Hyper-V firewall rule scoped to the WSL VM
creator id (`WSL-8080`, `WSL-8081`, `WSL-8082`) plus the ordinary inbound rule. The old
`netsh portproxy` for 8081 (`100.107.245.30:8081 -> 127.0.0.1:8081`) was removed: with
mirrored networking it shadowed the WSL listener and the port answered nothing from the
tailnet. Clients select `LOCAL_EMBED_ENDPOINT=http://100.107.245.30:8081` explicitly.

## Operations

```bash
# resident service (WSL, Ubuntu-24.04)
systemctl status llm-embedding.service          # enabled since 2026-09-07; it was stopped and disabled since 09-05
# bulk reindex from a client (e.g. Acer), fastest mode available:
~/ops/embedding_reindex_server.sh start [full|hybrid|cpu]   # on aaaaa
LOCAL_EMBED_ENDPOINT=http://100.107.245.30:8082 python tools/indexer/embed_corpus.py --root . --federated --force-local
~/ops/embedding_reindex_server.sh stop
# full mode needs the gateway down; bring it back afterwards:
sudo systemctl start llm-inference.service      # drop-in preloads qwen38
```

The server rejects inputs above its physical batch (`ubatch`); the Library chunker caps
paragraphs at 1,200 characters and the client never disguises a rejected chunk as a vector.

Source of the ops script: `ops/qualified-model-fleet/embedding_reindex_server.sh` in this
repository (installed at `/home/augus/ops/` on the node).

The wrapper stores its PID and log under `${XDG_RUNTIME_DIR:-/tmp}/llm-embedding-reindex-$UID`,
serializes control operations, and checks the configured server/model in `/proc` before
stopping a recorded PID. A stale PID never authorizes stopping an unrelated process.
`REINDEX_MODEL`, `REINDEX_SERVER`, and `REINDEX_RUNTIME_DIR` allow explicit installation
paths. Missing GPU telemetry selects CPU mode; startup failure removes the PID record.
Offline tests cover these boundaries without starting an embedding service.
