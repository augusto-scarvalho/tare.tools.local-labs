# Client integration

The raw llama-server remains on `http://127.0.0.1:8080`. A local
OpenAI-compatible proxy on `http://127.0.0.1:8082` injects the selected locale
contract into `/v1/chat/completions` while passing all other paths through.

Use `http://127.0.0.1:8082/v1` as the OpenAI base URL for clients that should
enforce Portuguese locale behavior. Existing system messages are preserved after
the locale contract. The proxy does not alter user, assistant, tool or JSON
messages and does not rewrite model output.

The listener is loopback-only. Remote/LAN clients must inject `contract_v2.txt`
as their first system instruction or receive a separately authorized firewall and
binding change; this experiment does not broaden network exposure.

Direct port 8080 remains available for benchmarks and clients that deliberately
want the model's unmodified behavior.
