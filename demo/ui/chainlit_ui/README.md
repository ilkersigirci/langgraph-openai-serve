# LGOS Chainlit UI

Standalone Chainlit client for an OpenAI-compatible LGOS endpoint. It
intentionally does not install or import the `langgraph-openai-serve` Python
package, demonstrating that UI logic needs only the OpenAI wire protocol.

Before starting, replace the example signing secret and configure the required
S3-compatible bucket and credentials in `.env`.

The small set of LGOS-specific model-detail fields, metadata keys, and event
schemas used by this client is declared locally in
[`lgos_protocol.py`](src/lgos_chainlit/lgos_protocol.py). That file links every
declaration to its authoritative source in the main LGOS repository.

```bash
cp .env.example .env
uv run --env-file .env lgos-chainlit-setup
uv run --env-file .env lgos-chainlit
```

Application settings use the `DEMO_CHAINLIT_` prefix. Reusable helper settings
use `CHAINLIT_UTILS_`; Chainlit's native `DATABASE_URL` and
`CHAINLIT_AUTH_SECRET` variables remain unprefixed. Plotly elements use
Chainlit's native `BUCKET_NAME`, `APP_AWS_*`, and `DEV_AWS_ENDPOINT` S3
settings so charts survive thread resume.
