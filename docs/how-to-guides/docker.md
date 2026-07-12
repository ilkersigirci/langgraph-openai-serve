# Docker

## Demo Compose

Run the API container:

```bash
docker compose up -d lgos-demo-api
```

Run the full demo stack:

```bash
docker compose up -d open-webui
```

Use:

- OpenAI base URL: `http://localhost:8000/v1`
- Open WebUI: `http://localhost:8080`

Compose starts PostgreSQL, `lgos-demo-api`, and `open-webui`. PostgreSQL stores
the interrupt demo's LangGraph checkpoints under `./docker/volumes/lgos-db`.
Before starting the API, its Compose `pre_start` lifecycle hook initializes or
migrates the checkpoint schema.

Import `demo/ui/openwebui/hitl_function.py` in
`Workspace -> Functions`, enable it, then select
`LangGraph OpenAI Pipe`. Send a request; the confirmation dialog resumes
the LangGraph interrupt with approve or reject.

The Pipe also converts OpenAI `url_citation` annotations to Open WebUI's native
`source` events. To try it, set the Function's `MODEL` valve to
`citation-events`; cited sources then appear in Open WebUI's source UI.

Ownership:

- `langgraph-openai-serve` owns OpenAI-compatible transport and LangGraph
  interrupt/resume behavior.
- The Open WebUI Function owns only the UI bridge: translate OpenAI citation
  annotations to native Open WebUI sources, detect the `langgraph_interrupt`
  tool call, show the confirmation modal, and send the resume tool message.
- Keep graph logic, HTTP routes, and custom response shapes out of the Open
  WebUI Function.

## Custom App Image

Minimal Dockerfile:

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Minimal `requirements.txt`:

```text
langgraph-openai-serve
uvicorn
```

Add any packages required by your graphs.

Minimal app:

```python
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

```yaml
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

- Add bearer-token authentication before exposing the API.
- Terminate HTTPS at a reverse proxy or platform load balancer.
- Use a production ASGI setup appropriate for your platform.
- Set memory and CPU limits.
- Use durable LangGraph checkpointers for interruptible graphs. The demo uses
  `AsyncPostgresSaver`; set `DEMO_POSTGRES_URI` when running the API outside Compose.
- Run `uv run --module demo.api.setup_checkpointer` once as a deployment task
  before starting or replacing API workers.
- Configure logging and monitoring.
