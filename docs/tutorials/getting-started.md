# Getting Started

This tutorial uses the runnable demo app in this repository. It exposes several
LangGraph workflows through the OpenAI-compatible `/v1` interface, so you can
inspect the code and call the graphs through an OpenAI client immediately.

## Prerequisites

- Python 3.11 or newer
- `uv`
- An OpenAI-compatible backend configured for the simple LLM graph, if you call
  `simple-graph-with-history` or `simple-graph-no-history`

The adapter and mock MCP demo graphs do not need real API keys.

## Run The Demo API

From the repository root:

```bash
make run-demo-api
```

The OpenAI-compatible API is available at `http://localhost:8000/v1`.

List the registered graphs:

```bash
curl http://localhost:8000/v1/models
```

The demo app registers these model names:

- `simple-graph-with-history`
- `simple-graph-no-history`
- `custom-input-output-context`
- `advanced-mcp-tools`

## Call It With The OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="DUMMY",
)

response = client.chat.completions.create(
    model="custom-input-output-context",
    messages=[{"role": "user", "content": "Show me the custom adapter."}],
    user="demo-user",
)

print(response.choices[0].message.content)
```

Try the async mock MCP graph:

```python
response = client.chat.completions.create(
    model="advanced-mcp-tools",
    messages=[{"role": "user", "content": "What is the weather in Istanbul?"}],
)

print(response.choices[0].message.content)
```

## Where To Look

- `demo/api/app.py` registers graph names as OpenAI model names.
- `demo/api/graphs/simple.py` exposes the default message-based graph.
- `demo/api/graphs/custom_io.py` shows `request_to_input`,
  `context_factory`, and `output_to_text`.
- `demo/api/graphs/advanced_mcp.py` shows an async graph factory that loads
  mock MCP-style tools before creating a ReAct graph.

## Existing FastAPI Apps

Mount the OpenAI-compatible routes on your own FastAPI app:

```python
from fastapi import FastAPI
from langgraph_openai_serve import GraphConfig, GraphRegistry, LangchainOpenaiApiServe
from your_graphs import my_graph

app = FastAPI(title="My LangGraph API")

graphs = GraphRegistry(
    registry={
        "my-graph": GraphConfig(
            graph=my_graph,
            streamable_node_names=["generate"],
        )
    }
)

server = LangchainOpenaiApiServe(app=app, graphs=graphs)
server.bind_openai_chat_completion()
```

Routes are mounted at `/v1` by default. Set `LGOS_OPENAI_API_PREFIX` to change
that default, or pass `prefix="/custom/path"` when binding one server instance.
FastAPI docs for the mounted OpenAI API are disabled by default; set
`LGOS_OPENAI_API_DOCS_ENABLED=true` to inspect `/v1/docs` locally.

## Next Steps

- [Create custom graphs](custom-graphs.md)
- [Connect OpenAI clients](openai-clients.md)
- [Run with Docker](../how-to-guides/docker.md)
