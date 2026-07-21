# Get Started With The Package

Build a small application that registers one LangGraph graph as an OpenAI
`model` using the installed `langgraph-openai-serve` package.

!!! tip "Looking for a ready-made stack?"

    The [self-contained demo](demo/index.md) provides example graphs,
    PostgreSQL, Docker Compose, Chainlit, Open WebUI, and Bifrost.

## Install

Create or enter a Python 3.11 or newer project, then add LGOS:

=== "uv (recommended)"

    ```bash
    uv add langgraph-openai-serve
    ```

=== "pip"

    ```bash
    pip install langgraph-openai-serve
    ```

LGOS installs its FastAPI, LangGraph, OpenAI SDK, and server dependencies. Add
the model providers, tools, and persistence packages required by your graphs.

## Create A Graph And Application

Create `app.py`:

```python title="app.py"
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
)


def respond(state: MessagesState) -> dict[str, list[AIMessage]]:
    text = str(state["messages"][-1].content)
    return {"messages": [AIMessage(content=f"LGOS received: {text}")]}


graph = (
    StateGraph(MessagesState)
    .add_node("respond", respond)
    .add_edge(START, "respond")
    .add_edge("respond", END)
    .compile()
)

registry = GraphRegistry(
    registry={"echo": GraphConfig(graph=graph)},
)

app = FastAPI()
LanggraphOpenaiServe(app=app, graphs=registry).bind_openai_api()
```

The registry key `echo` is the OpenAI model name. This deterministic graph is
deliberately provider-free, so the first request needs no upstream API key.

## Run The Server

```bash
uvicorn app:app --reload
```

The OpenAI-compatible base URL is `http://localhost:8000/v1`.

## Call The Graph

Use the ordinary OpenAI Python client installed with LGOS:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

response = client.chat.completions.create(
    model="echo",
    messages=[{"role": "user", "content": "Hello from an OpenAI client"}],
)

print(response.choices[0].message.content)
```

The result is `LGOS received: Hello from an OpenAI client` in a standard Chat
Completions response. The dummy key satisfies the SDK; LGOS does not enforce
authentication unless the host application adds it.

## Next Steps

- [Register adapters, streaming nodes, settings, and interrupts](tutorials/custom-graphs.md)
- [Connect Python and JavaScript OpenAI clients](tutorials/openai-clients.md)
- [Add bearer-token authentication](how-to-guides/authentication.md)
