# LGOS Chainlit UI

Standalone Chainlit client for an OpenAI-compatible LGOS endpoint. It
intentionally does not install or import the `langgraph-openai-serve` Python
package, demonstrating that UI logic needs only the OpenAI wire protocol.

The client uses Responses exclusively and never connects directly to an LGOS
or Files container. `OPENAI_GATEWAY_TYPE=litellm|bifrost` selects the gateway.
LiteLLM uses managed Responses routing; Bifrost uses its native Responses
route. Both use their normal Files route. A separate catalog client uses only
the gateway's model-detail pass-through when necessary to preserve LGOS
descriptions, features, and client-setting schemas.

Before starting, replace the example signing secret and configure the required
S3-compatible bucket and credentials in `.env`.

The small set of LGOS-specific model-detail fields and metadata keys used by
this client is declared locally in
[`lgos_protocol.py`](src/lgos_chainlit/lgos_protocol.py). That file links every
declaration to its authoritative source in the main LGOS repository.

```bash
cp .env.example .env
uv run --locked --env-file .env lgos-chainlit-setup
uv run --locked --env-file .env lgos-chainlit
```

Application settings use the `DEMO_CHAINLIT_` prefix. Reusable helper settings
use `CHAINLIT_UTILS_`; Chainlit's native `DATABASE_URL` and
`CHAINLIT_AUTH_SECRET` variables remain unprefixed. Native Chainlit elements
use `BUCKET_NAME`, `APP_AWS_*`, and `DEV_AWS_ENDPOINT` S3 settings so generated
files survive thread resume.

User attachments are uploaded separately through the selected gateway's normal
OpenAI Files API. LiteLLM assigns those requests to `litellm_proxy`; Bifrost
assigns them to `lgos-files`. The returned `file_id` reaches the graph as a
Responses `input_file` content part.
