# Docker

## Demo Compose

=== "API only"

    ```bash
    docker compose up -d lgos-demo-api
    ```

    OpenAI base URL: `http://localhost:8000/v1`

=== "Bifrost proxy"

    ```bash
    docker compose up --wait lgos-bifrost
    ```

    - Bifrost inference base URL: `http://localhost:8081/v1`
    - Bifrost discovery base URL:
      `http://localhost:8081/openai_passthrough/v1`

    Run `make test-bifrost` to verify detailed model metadata through the proxy.
    See [Configure an OpenAI Proxy](openai-proxy.md) for client settings and the
    Bifrost routing contract.

=== "Chainlit UI"

    ```bash
    docker compose up -d lgos-chainlit
    ```

    - OpenAI base URL: `http://localhost:8000/v1`
    - Chainlit: `http://localhost:5000`

    Prepare `.env` and its Chainlit signing secret as described in
    [Getting Started](../tutorials/getting-started.md#run-the-chainlit-ui).
    The default mock login does not require an OAuth client. The selected simple
    graph exposes its history control through Chainlit Chat Settings discovered
    from `GET /v1/models/{model}`.

=== "Open WebUI"

    ```bash
    docker compose up -d open-webui
    ```

    - OpenAI base URL: `http://localhost:8000/v1`
    - Open WebUI: `http://localhost:8080`

Both UI choices start PostgreSQL and `lgos-demo-api` as dependencies. PostgreSQL
stores the interrupt demo's LangGraph checkpoints and Chainlit conversation
history under `./docker/volumes/lgos-db`. Compose `pre_start` lifecycle hooks
apply the relevant schema migrations before starting each service.

Import `demo/ui/openwebui/openwebui_pipe.py` in `Admin Panel -> Functions` and
enable it. Open WebUI's native
[manifold Pipe](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/#creating-multiple-models-with-pipes)
hook fetches `/v1/models` and adds every registered graph to the model selector.
Configure the API base URL and key with the Function valves; there is no
per-model valve. The included Pipe does not retrieve detailed runtime settings
descriptors or send `metadata.langgraph_runtime_settings`, so those graphs use
their registered server defaults. See the
[client support matrix](../explanation/openai-compatibility.md#client-and-integration-support).

The Pipe bridges streaming text, streaming citation annotations, and interrupt
approval; it does not own graph or transport behavior. See
[Citation ownership](../explanation/openai-compatibility.md#citation-ownership-and-ui-rendering)
and the [interrupt protocol](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for those boundaries. Select `interruptible-approval` to try confirmation and
`lgos-rag` to stream a linked documentation answer.

After changing the local Function file, update or re-import it in Open WebUI;
Open WebUI stores its own copy of imported Function code.

## Custom App Image

Start with the FastAPI application from
[Custom Graphs](../tutorials/custom-graphs.md#register-and-bind). The files below
containerize that application as `app:app`.

Minimal Dockerfile:

```dockerfile title="Dockerfile"
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Minimal `requirements.txt`:

```text title="requirements.txt"
langgraph-openai-serve
uvicorn
```

Add any packages required by your graphs.

Minimal compose:

```yaml title="compose.yaml"
services:
  langgraph-api:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Production Notes

!!! warning "The example is a starting point"

    The example omits the production controls listed below.

- Add bearer-token authentication before exposing the API.
- Terminate HTTPS at a reverse proxy or platform load balancer.
- Use a production ASGI setup appropriate for your platform.
- Set memory and CPU limits.
- Use durable LangGraph checkpointers for interruptible graphs. The demo uses
  `AsyncPostgresSaver`; set `DEMO_POSTGRES_URI` when running the API outside Compose.
- Run `uv run --module demo.api.setup_checkpointer` once as a deployment task
  before starting or replacing API workers.
- Run `uv run --module demo.ui.chainlit_ui.setup_database` once before starting
  or replacing Chainlit workers.
- Generate a unique `CHAINLIT_AUTH_SECRET`, replace the local demo PostgreSQL
  credentials, and do not deploy with `DEMO_CHAINLIT_LOGIN_TYPE=mock`.
- For PocketID OAuth, store its client secret outside source control, terminate
  TLS at the ingress, set `CHAINLIT_URL` to the external HTTPS origin, and
  register its `/auth/oauth/PocketID/callback` URL in PocketID.
- Restrict Chainlit `allow_origins` to the deployed HTTPS origin. Chainlit uses
  WebSockets; configure session affinity when multiple UI workers sit behind a
  load balancer.
- Configure supported object storage before enabling Chainlit file uploads.
- Configure logging and monitoring.
