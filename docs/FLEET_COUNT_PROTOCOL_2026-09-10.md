# Public text counting on the qualified model gateway

Owner: Local Labs. Implemented and exercised on aaaaa on 2026-09-10.
The gateway remains the owner of selecting/loading one resident backend;
it does not own an agent loop, context compaction, retries or Work admission.

## Public contract

`POST /v1/fleet/profile` accepts exactly `{"model":"qwen38"}`. Selection is
explicit and may replace the resident model. It reads the selected server's
`/props` and `/slots`, checks the live alias/artifact path against the registry,
and returns the minimum per-slot context window, all slot windows, template
SHA-256, build identifier, registered artifact SHA-256, runtime configuration
SHA-256 and a fingerprint. The GGUF is not freshly hashed: provenance is
`qualified_registry_and_live_model_path_not_fresh_gguf_hash`.

`POST /v1/fleet/count` accepts exactly `{"model":"qwen38","request":{...}}`.
The inner request must have the same explicit model and the complete Chat
payload, including tools and template parameters. Messages have string/null
content; structured multimodal content is refused. Envelope size is limited
to 1 MiB. Under the existing request lock, the gateway selects the model,
reads its profile, calls `/apply-template` with the whole request, then calls
`/tokenize` with `add_special=true, parse_special=true`. It returns the token
count and a `tare.tools/fleet-count/1` binding over the requested alias,
canonical model, original request digest and effective fingerprint.

For `POST /v1/chat/completions`, the client may attach that binding
as `tare_fleet_binding`. The gateway removes the envelope field, reselects
and rechecks the live profile/request under the same lock, and refuses drift
with HTTP 409 before generation. The successful finite response adds
`tare_fleet_observation` with the checked binding/profile. Legacy requests
without the field retain their existing path. Bound response JSON is limited to
8 MiB. The streaming extension emits a separate SSE data event with empty
`choices` and `tare_fleet_observation` before forwarding backend events. This
receipt attests only the represented route check; completion still requires the
provider terminal event and usage remains the provider's own counters. The entire
streamed request, including `stream` and `stream_options`, is part of the binding.
Forwarding uses available chunks instead of waiting for a 64 KiB block. The
existing request lock remains held through forwarding.

The binding is a stateless drift check, not a lease, signature or security
gate. A switch away and back is allowed if the effective profile is unchanged;
it gives no guarantee of KV cache retention. It does not detect an in-place
artifact replacement that preserves all observed identity fields.

## Cost and failure visibility

Count receipts record four internal operations: props, slots, template and
tokenizer. Bound generation records two verification operations: props and
slots, in addition to forwarding generation. Health/loading polls are outside
these counters. Client HTTP receipts cover the client-to-gateway boundary;
they must not be advertised as all physical HTTP operations of the system.
No token usage is invented for calls without returned provider usage.

The protocol follows the selected slop.cpp `tools/server/README.md` contracts
for `/apply-template`, `/tokenize`, `/props` and `/slots`. Exact equality was
observed on the bounded text/tool canaries below, not established universally
for every template, multimodal input or time-sensitive template behavior.

## Qualification and operational change

Synthetic gateway HTTP tests cover aliases, switch-away/back, request/template/
slot/runtime/identity drift, invalid envelopes and legacy generation. Combined
with Kernel/OS regressions, 123 Windows tests passed. Real native-agent CLI
sessions on Qwen3.8, HauhauCS, Fable-TC and Qwen3.6 MoE each read a random marker
from a file and delivered it in two generations. All eight input counts equal
the corresponding provider usage; all four mutated requests were refused409.
The task does not qualify general coding ability or context-selection savings.

See the OS owner for [consumer profiles, complete results and retained evidence](../../tare.tools.os/docs/operations/NATIVE_FLEET_COUNT_2026-09-10.md).

The two serving files were updated on aaaaa and the existing system service
`llm-inference.service` restarted. Registry, default route and unit were
unchanged. Qwen3.8 was restored and observed idle with the service active.
No remote commit or push was performed. The old gateway bytes were verified
and moved into the central local evidence lot; its remote scratch directory
was deleted after verification.

Rollback, if needed for this exact deployment: recover `gateway.before.py`
from the evidence archive, verify SHA-256
`b4afb885e1b23ed0bc227620f4140038eeb6bcfc2b066d645666ce07e10a3ccb`,
restore `tools/serving/qualified_model_gateway.py` in the aaaaa owner and restart
the same system service. Native fleet-bound consumers then require selecting
a previously qualified realization or restoring the new gateway; they must
not silently continue without counting. No session journal translation is
part of rollback. The additive `fleet_count.py` is unused by the old gateway.
