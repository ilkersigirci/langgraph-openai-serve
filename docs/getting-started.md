# Get Started With The Package

Build a small application that registers one LangGraph graph as an OpenAI
`model` using the installed `langgraph-openai-serve` package.

!!! tip "Looking for a ready-made stack?"

    The [self-contained demo](demo/index.md) provides example graphs,
    PostgreSQL, Docker Compose, Chainlit, Open WebUI, and Bifrost.

## Choose Responses Or Chat Completions

Use the Responses API for new clients. It is LGOS's primary client contract and
the only route that exposes workflow features such as phase-tagged status and
human-in-the-loop interrupts. Use Chat Completions when an existing client only
supports that API and the graph needs the simpler compatibility surface.

| Need | Responses API (`/v1/responses`) | Chat Completions (`/v1/chat/completions`) |
| --- | --- | --- |
| New LGOS integration | **Recommended** | Compatibility for existing Chat-only clients |
| Final assistant text | Message with `phase="final_answer"`; typed SSE events when streaming | Assistant message; `delta.content` when streaming |
| Graph status from `status_event()` | Streaming message with `phase="commentary"` when the graph declares `client_events` | Ignored |
| Human review with LangGraph `interrupt()` | `langgraph_interrupt` function calls resumed with `previous_response_id` and matching outputs | Unsupported; interrupt-enabled models return HTTP 400 |
| Client-executed function tools | `function_call` and `function_call_output` items | `tool_calls` and tool messages |
| File input by opaque Files API ID | `input_file` content part | Native Chat file content part |
| Citation annotations | Response output-text annotations | Assistant-message or final-stream annotations |
| Conversation history | Client resends ordinary input; `previous_response_id` is reserved for interrupt resume | Client resends message history |

The `langgraph_openai_serve.features` model extension advertises
`client_events`, `file_inputs`, and `interrupts` so a capability-aware UI can
enable only supported controls. See the
[complete compatibility contract](explanation/openai-compatibility.md) for the
accepted fields, item shapes, streaming events, errors, and retention rules.

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
    registry={
        "echo": GraphConfig(
            graph=graph,
            description="Echo the latest user message.",
        )
    },
)

app = FastAPI()
LanggraphOpenaiServe(app=app, graphs=registry).bind_openai_api()
```

The registry key `echo` is the OpenAI model name. Its required description is
advertised by the LGOS model extension for clients that render model catalogs.
This deterministic graph is deliberately provider-free, so the first request
needs no upstream API key.

## Run The Server

```bash
uvicorn app:app --reload
```

The OpenAI-compatible base URL is `http://localhost:8000/v1`.

## Call The Graph

Use the ordinary OpenAI Python client installed with LGOS:

=== "Responses (recommended)"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

    response = client.responses.create(
        model="echo",
        input="Hello from an OpenAI client",
        store=False,
    )

    print(response.output_text)
    ```

=== "Chat Completions"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

    completion = client.chat.completions.create(
        model="echo",
        messages=[
            {"role": "user", "content": "Hello from an OpenAI client"}
        ],
    )

    print(completion.choices[0].message.content)
    ```

Both calls print `LGOS received: Hello from an OpenAI client`. The dummy key
satisfies the SDK; LGOS does not enforce authentication unless the host
application adds it. Responses remains the primary client contract; the Chat
example is the concise compatibility path for simple graphs.

## Next Steps

- [Understand the request path and state ownership](explanation/architecture.md)
- [Register adapters, streaming nodes, settings, and interrupts](tutorials/custom-graphs.md)
- [Connect Python and JavaScript OpenAI clients](tutorials/openai-clients.md)
- [Add bearer-token authentication](how-to-guides/authentication.md)
