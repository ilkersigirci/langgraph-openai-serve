# Docker

## Demo Compose

=== "API only"

    ```bash
    docker compose up -d lgos-demo-api
    ```

    OpenAI base URL: `http://localhost:8000/v1`

=== "Full stack"

    ```bash
    docker compose up -d open-webui
    ```

    - OpenAI base URL: `http://localhost:8000/v1`
    - Open WebUI: `http://localhost:8080`

The full stack starts PostgreSQL, `lgos-demo-api`, and `open-webui`. PostgreSQL
stores the interrupt demo's LangGraph checkpoints under `./docker/volumes/lgos-db`.
Before starting the API, its Compose `pre_start` lifecycle hook initializes or
migrates the checkpoint schema.

Import `demo/ui/openwebui/openwebui_pipe.py` in `Admin Panel -> Functions` and
enable it. Open WebUI's native
[manifold Pipe](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/#creating-multiple-models-with-pipes)
hook fetches `/v1/models` and adds every registered graph to the model selector.
Configure the API base URL and key with the Function valves; there is no
per-model valve.

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
- Configure logging and monitoring.
