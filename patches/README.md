# Archival engine patch receipts

Files in this directory preserve experiment provenance and historical diffs.
They are not the canonical implementation and should not be applied as the
normal way to build the runtime.

The maintained source, runtime flags, build instructions, and qualification
harness live in [`slop.cpp`](https://github.com/augusto-scarvalho/slop.cpp).
When citing an archival patch, also record the exact `slop.cpp` source commit and
the local experiment receipt that used it.
