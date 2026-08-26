# Execution aborted fail-closed

Both frozen historical module-targeting artifacts declare `peft_type=LOKR`. The active llama.cpp converter requires LoRA fields and aborted on missing `lora_alpha` before the inference service was stopped. No runtime observations or gates were produced. A successor must explicitly freeze actual LoRA checkpoints.
