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

    - Standard inference, without client events:
      `http://localhost:8081/v1` with `openai/`-prefixed models
    - Detailed discovery or event-enabled inference:
      `http://localhost:8081/openai_passthrough/v1` with unprefixed models

    Run `make test-bifrost` to verify detailed model metadata through the proxy.
    See [OpenAI-Compatible Proxies](../integrations/openai-proxies.md) for the
    Bifrost routing contract.

=== "Chainlit UI"

    ```bash
    docker compose up -d lgos-chainlit
    ```

    - OpenAI base URL: `http://localhost:8000/v1`
    - Chainlit: `http://localhost:5000`

    Prepare `.env` and its signing secret as described in the
    [Chainlit integration](../integrations/chainlit.md).

=== "Open WebUI"

    ```bash
    docker compose up -d open-webui
    ```

    - OpenAI base URL: `http://localhost:8000/v1`
    - Open WebUI: `http://localhost:8080`

    Synchronize and configure the included Pipes as described in the
    [Open WebUI integration](../integrations/open-webui.md).

Both UI choices start PostgreSQL and `lgos-demo-api` as dependencies. PostgreSQL
stores LangGraph checkpoints and Chainlit data under
`./docker/volumes/lgos-db`. Compose `pre_start` hooks apply the required schema
migrations before each service starts.

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
- Configure logging and monitoring.
