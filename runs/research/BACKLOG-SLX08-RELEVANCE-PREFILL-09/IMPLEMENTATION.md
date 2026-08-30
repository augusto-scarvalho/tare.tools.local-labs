# BACKLOG-SLX08-RELEVANCE-PREFILL-09 implementation

R8 stopped before measurement or service maintenance because one manually
transcribed source digest was missing a hexadecimal `e`. R9 delegates the same
implementation with the exact digest copied from `Get-FileHash`. No scientific
or runtime factor changed. Focused tests recompute every R9 source digest before
the packet is marked implemented.
