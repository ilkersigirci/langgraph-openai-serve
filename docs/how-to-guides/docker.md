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

Compose starts `lgos-demo-api` and `open-webui`.

Import `demo/ui/openwebui/hitl_function.py` in
`Workspace -> Functions`, enable it, then select
`LangGraph HITL Approval Modal`. Send a request; the confirmation dialog resumes
the LangGraph interrupt with approve or reject.

Ownership:

- `langgraph-openai-serve` owns OpenAI-compatible transport and LangGraph
  interrupt/resume behavior.
- The Open WebUI Function owns only the UI bridge: detect the
  `langgraph_interrupt` tool call, show the confirmation modal, and send the
  resume tool message.
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
from langgraph_openai_serve import GraphConfig, GraphRegistry, LangchainOpenaiApiServe
from my_graphs import chat_graph

app = FastAPI()
graphs = GraphRegistry(
    registry={"chat": GraphConfig(graph=chat_graph, streamable_node_names=["generate"])}
)
LangchainOpenaiApiServe(app=app, graphs=graphs, configure_cors=True).bind_openai_chat_completion()
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
- Use durable LangGraph checkpointers for interruptible graphs.
- Configure logging and monitoring.
