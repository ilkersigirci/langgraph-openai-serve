# LGOS Chainlit UI

Standalone Chainlit client for an OpenAI-compatible LGOS endpoint. It
intentionally does not install or import the `langgraph-openai-serve` Python
package, demonstrating that UI logic needs only the OpenAI wire protocol.

The small set of LGOS-specific discovery fields, metadata keys, and event
schemas used by this client is declared locally in
[`lgos_protocol.py`](src/lgos_chainlit/lgos_protocol.py). That file links every
declaration to its authoritative source in the main LGOS repository.

```bash
cp .env.example .env
uv run --env-file .env lgos-chainlit-setup
uv run --env-file .env lgos-chainlit
```

Application settings use the `DEMO_CHAINLIT_` prefix. Chainlit's native
`DATABASE_URL` and `CHAINLIT_AUTH_SECRET` variables remain unprefixed.
