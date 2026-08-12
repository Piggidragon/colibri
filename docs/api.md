# V4 local API and web UI

`coli serve` starts the dependency-free local OpenAI-compatible gateway around
one persistent DeepSeek V4 engine. `coli web` starts the same service and opens
the bundled dashboard.

```bash
cd c
./coli serve --model ~/Services/models/colibri/deepseek-v4-flash \
  --ram 24 --ctx 32768 --host 127.0.0.1 --port 8000 \
  --model-id deepseek-v4-flash-local
```

The default bind address is loopback. Set an API key before binding beyond the
local machine:

```bash
COLI_API_KEY=local-secret ./coli serve \
  --model ~/Services/models/colibri/deepseek-v4-flash --ram 24 --ctx 32768
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer local-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-local",
    "messages": [{"role": "user", "content": "Explain cache locality."}],
    "max_tokens": 64,
    "stream": true
  }'
```

Implemented endpoints are `GET /health`, `GET /v1/models`,
`GET /v1/models/{model}`, `POST /v1/chat/completions`, and the legacy
`POST /v1/completions`. JSON and SSE streaming responses include token usage.

## V4 limits

- Generation is greedy. `temperature` and `top_p` are accepted by the protocol
  but ignored by the V4 target engine.
- One V4 context is active at a time. Concurrent requests queue in the gateway;
  `--max-queue` (default 8) and `--queue-timeout` (default 300 seconds) bound
  that queue.
- Tools/function calling are not wired up for V4 and return a clear 400 error.
- Requests must fit `CTX` together with their requested completion. The server
  returns `context_length_exceeded` instead of silently truncating a prompt.

`--kv-slots` is kept for the shared launcher surface, but the V4 engine admits
only slot 0. Keep it at its default of 1.

`coli web --no-browser` is useful on the headless V4 host. The dashboard and
API share the selected port.
