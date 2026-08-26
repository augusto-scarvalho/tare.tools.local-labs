# BACKLOG-NEGATIVE-KV-REAL-SCREEN-01 aborted before measurement

The GPU worker loaded the frozen Qwen weights but stopped before any forward pass or decisive metric. In Transformers 5.15.1, `AutoModelForCausalLM` returned a wrapper whose `model.model` is already `Qwen3_5TextModel`; the worker incorrectly attempted `model.model.language_model` and raised `AttributeError`.

No score, threshold, prompt, layer, seed or aggregation rule was observed or changed. The exact worker stdout/stderr and input ledgers remain under `raw/`. The persistent inference service was restored through the host `finally` block with the same executable/arguments, healthy ports 8080/8081 and `NRestarts=0`.

A successor may change only the model-object binding from `model.model.language_model` to `model.model`, freeze this failed packet and rerun the identical preregistered experiment in a fresh output directory.
