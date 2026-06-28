# Getting Started

This tutorial uses the repository demo app. It serves several LangGraph graphs
through the OpenAI-compatible `/v1` interface.

## Prerequisites

- Python 3.11 or newer
- `uv`
- An OpenAI-compatible upstream model only if you call the simple LLM graphs

The custom adapter and mock MCP demo graphs do not require real API keys.

## Run The Demo API

```bash
make run-demo-api
```

The base URL is `http://localhost:8000/v1`.

Inspect registered graphs:

```bash
curl http://localhost:8000/v1/models
```

The demo model names are listed in [Reference](../reference.md#demo-models).

## Call A Graph

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

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
```

## Demo Files

- `demo/api/app.py`: registers graph names as OpenAI model names.
- `demo/api/graphs/simple.py`: default `{"messages": messages}` graph shape.
- `demo/api/graphs/custom_io.py`: input, output, and context adapters.
- `demo/api/graphs/advanced_mcp.py`: async factory with mock MCP-style tools.
- `demo/api/graphs/interruptible.py`: interrupt and resume graph persisted in
  `checkpoints.sqlite` by the demo application.
- `demo/ui/chainlit_ui/hitl.py`: Chainlit interrupt approval demo.
- `demo/ui/openwebui/hitl_function.py`: Open WebUI Pipe Function approval modal
  demo.

## Human In The Loop Demo

The `interruptible-approval` model showcases LangGraph `interrupt()` and resume.

Chainlit:

```bash
make run-demo-ui-chainlit-hitl
```

Open WebUI:

```bash
docker compose up -d open-webui
```

See [Docker](../how-to-guides/docker.md) for the Open WebUI Function import and
[OpenAI compatibility](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for the interrupt tool-call protocol.

## Existing FastAPI Apps

```python
from fastapi import FastAPI
from langgraph_openai_serve import GraphConfig, GraphRegistry, LangchainOpenaiApiServe
from your_graphs import my_graph

app = FastAPI()
graphs = GraphRegistry(
    registry={
        "my-graph": GraphConfig(
            graph=my_graph,
            streamable_node_names=["generate"],
        )
    }
)

LangchainOpenaiApiServe(app=app, graphs=graphs).bind_openai_chat_completion()
```

See [Reference](../reference.md) for prefixes, docs URLs, settings, and public
classes.

## Next Steps

- [Create custom graphs](custom-graphs.md)
- [Connect OpenAI clients](openai-clients.md)
- [Run with Docker](../how-to-guides/docker.md)
