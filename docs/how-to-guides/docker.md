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

The Pipe yields OpenAI content deltas through Open WebUI's native generator
streaming contract and leaves Markdown rendering to the UI. For streaming
requests, it also forwards OpenAI citation annotations unchanged when LGOS
returns them; it does not create or translate citations. Select
`interruptible-approval` to try confirmation and `lgos-rag` to stream a linked
documentation answer.
After changing the local Function file, update or re-import it in Open WebUI;
Open WebUI stores its own copy of imported Function code.

Ownership:

- `langgraph-openai-serve` owns OpenAI-compatible transport and LangGraph
  interrupt/resume behavior.
- The Open WebUI Function owns only the UI bridge: yield streamed text and
  annotation chunks, detect the `langgraph_interrupt` tool call, show the
  confirmation modal, and send the resume tool message.
- Keep graph logic, HTTP routes, and custom response shapes out of the Open
  WebUI Function.

## Custom App Image

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

Minimal app:

```python title="app.py"
from fastapi import FastAPI
from langgraph_openai_serve import GraphConfig, GraphRegistry, LanggraphOpenaiServe
from my_graphs import chat_graph

app = FastAPI()
graphs = GraphRegistry(
    registry={"chat": GraphConfig(graph=chat_graph, streamable_node_names=["generate"])}
)
LanggraphOpenaiServe(app=app, graphs=graphs).bind_openai_api()
```

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

    Add authentication, HTTPS termination, resource limits, durable state, and
    operational monitoring before exposing the service.

- Add bearer-token authentication before exposing the API.
- Terminate HTTPS at a reverse proxy or platform load balancer.
- Use a production ASGI setup appropriate for your platform.
- Set memory and CPU limits.
- Use durable LangGraph checkpointers for interruptible graphs. The demo uses
  `AsyncPostgresSaver`; set `DEMO_POSTGRES_URI` when running the API outside Compose.
- Run `uv run --module demo.api.setup_checkpointer` once as a deployment task
  before starting or replacing API workers.
- Configure logging and monitoring.
