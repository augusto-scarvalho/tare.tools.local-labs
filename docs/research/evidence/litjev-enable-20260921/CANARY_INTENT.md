# Activation canary

Two bounded calls through the deployed worker: one resident-Qwen call using the
unchanged effort-cost-0 synthetic request; one CPU fallback call with the same
request while a diagnostic image lease is held and the text backend unloaded.
No actual image/training is submitted or interrupted. Both must return a valid
Choice, retain truthful GPU=1/CPU=0 output usage, select the expected cheaper
adequate pair and preserve lease semantics. Timeout55 seconds per worker;
no answer tuning/retries. Check queue/lease before admission. Initial empty text
residency must be restored. No model promotion from this transport canary.
