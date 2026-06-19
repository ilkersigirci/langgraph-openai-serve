# LangGraph OpenAI Serve

A package that provides an OpenAI-compatible API for LangGraph instances.

## What It Does

- Expose your LangGraph instances through an OpenAI-compatible API
- Register multiple graphs as OpenAI model names
- Mount the API on an existing FastAPI app
- Use standard OpenAI clients, Open WebUI, Chainlit, or any compatible client
- Support streaming graph nodes when the underlying graph emits streamed messages

## Installation

```bash
# Using uv
uv add langgraph-openai-serve

# Using pip
pip install langgraph-openai-serve
```

## Run The Demo

The `demo` folder shows the package in action with a FastAPI backend and optional UIs.

```bash
make run-demo-api
```

The API runs at `http://localhost:8000/v1`.

List the registered graphs:

```bash
curl http://localhost:8000/v1/models
```

Call a graph with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="DUMMY",
)

response = client.chat.completions.create(
    model="simple-graph-no-history",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

Streaming works the same way for graphs that stream from configured nodes:

```python
stream = client.chat.completions.create(
    model="simple-graph-with-history",
    messages=[{"role": "user", "content": "Write a short poem about graphs."}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")
```

## Demo Graphs

- `demo/api/graphs/simple.py` shows the default `{"messages": messages}` graph shape.
- `demo/api/graphs/custom_io.py` shows custom input, output, and runtime context adapters.
- `demo/api/graphs/advanced_mcp.py` shows an async graph factory that loads mock MCP tools before building a ReAct graph.
- `demo/api/app.py` shows how graph names are registered as OpenAI model names.

The demo registers:

- `simple-graph-with-history`
- `simple-graph-no-history`
- `custom-input-output-context`
- `advanced-mcp-tools`

Try the custom adapter graph with a normal, non-streaming request:

```python
response = client.chat.completions.create(
    model="custom-input-output-context",
    messages=[{"role": "user", "content": "Show me custom schemas."}],
    user="demo-user",
)

print(response.choices[0].message.content)
```

## Use Your Own FastAPI App

```python
from fastapi import FastAPI
from langgraph_openai_serve import GraphConfig, GraphRegistry, LangchainOpenaiApiServe
from your_graphs import my_graph

app = FastAPI(title="LangGraph OpenAI API")

graphs = GraphRegistry(
    registry={
        "my-graph": GraphConfig(
            graph=my_graph,
            streamable_node_names=["generate"],
        )
    }
)

server = LangchainOpenaiApiServe(app=app, graphs=graphs)
server.bind_openai_chat_completion(prefix="/v1")
```

`GraphConfig` also accepts `request_to_input`, `context_factory`, and
`output_to_text` adapters when your graph uses custom LangGraph schemas. See
`demo/api/graphs/custom_io.py` for the complete example.

## More

- Docker and Open WebUI demo: `docker-compose.yml`
- Chainlit demo: `make run-demo-ui-chainlit`
- Full docs: `docs/`
