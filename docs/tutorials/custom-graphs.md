# Custom Graphs

Each `GraphRegistry` key becomes an OpenAI `model` name.

## Default Message Graph

```python title="Default message graph"
GraphConfig(graph=my_graph, streamable_node_names=["generate"])
```

Without adapters, graph input is `{"messages": langchain_messages}` and output
text is read from `result["messages"][-1].content`.

## Custom Schemas

Use adapters when your graph has native LangGraph input, output, or context
schemas:

```python title="Custom graph adapters"
GraphConfig(
    graph=custom_io_graph,  # (1)!
    request_to_input=request_to_input,  # (2)!
    context_factory=context_factory,  # (3)!
    output_to_text=output_to_text,  # (4)!
)
```

1.  Keep the graph's native LangGraph schema.
2.  Build graph input from the validated OpenAI request and converted messages.
3.  Build optional LangGraph runtime context from the request and public settings.
4.  Render the graph's native output as OpenAI assistant text.

See `demo/api/graphs/custom_io.py` for the runnable version.

## Runtime Context

Use LangGraph runtime context for immutable, per-invocation application values
that nodes need but that do not belong in graph state. Define a context schema,
declare it on `StateGraph`, and read it from the injected `Runtime` object:

```python title="Typed runtime context"
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


@dataclass(frozen=True)
class AppContext:
    user_id: str


class State(TypedDict, total=False):
    question: str
    answer: str


async def generate(state: State, runtime: Runtime[AppContext]) -> dict[str, str]:
    return {
        "answer": f"{runtime.context.user_id} asked: {state['question']}"
    }


custom_graph = (
    StateGraph(State, context_schema=AppContext)
    .add_node("generate", generate)
    .add_edge(START, "generate")
    .add_edge("generate", END)
    .compile()
)
```

Build that context from the validated OpenAI request at the adapter boundary:

```python title="Request to runtime context"
from langchain_core.messages import BaseMessage

from langgraph_openai_serve import ClientSettings, GraphConfig
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest


def request_to_input(
    request: ChatCompletionRequest,
    messages: list[BaseMessage],
) -> State:
    return {"question": str(messages[-1].content or "")}


def context_factory(
    request: ChatCompletionRequest,
    _client_settings: ClientSettings | None,
) -> AppContext:
    return AppContext(user_id=request.user or "anonymous")


def output_to_text(output: State) -> str:
    return output["answer"]


custom_graph_config = GraphConfig(
    graph=custom_graph,
    request_to_input=request_to_input,
    context_factory=context_factory,
    output_to_text=output_to_text,
)
```

LGOS passes the returned value through LangGraph's `context` argument. Do not
put application values such as `user_id`, model selection, or prompt options in
`config["configurable"]`.

!!! note "Keep context and config separate"

    See [Context versus config](../explanation/langgraph-integration.md#context-versus-config)
    for execution settings, checkpoint identity, and async config propagation.
    For a checkpointed workflow, continue to [Interrupts](#interrupts).

## User-Configurable Runtime Context

Expose only a safe, explicit subset of runtime context when an ordinary OpenAI
client should configure a graph. Define that public subset as a `ClientSettings`
model. All fields automatically share one JSON metadata envelope:

```python title="Public client configuration"
from pydantic import Field

from langgraph_openai_serve import ClientSettings, GraphConfig


class PublicRuntimeConfig(ClientSettings):
    use_history: bool = Field(
        default=True,
        title="Use conversation history",
    )

configurable_graph = GraphConfig(
    graph=my_graph,
    client_config=PublicRuntimeConfig,
    streamable_node_names=["generate"],
)
```

LGOS validates the request directly from JSON and passes the resulting
`PublicRuntimeConfig` instance as LangGraph runtime context. `ClientSettings`
provides strict, immutable, extra-forbid, default-validating model configuration.
`GET /v1/models` remains a standard lightweight list;
`GET /v1/models/{model}` returns the generated JSON Schema, validated defaults,
and schema version for the selected graph.

Small typed options use one compact JSON object, for example
`metadata={"langgraph_config": "{\"use_history\":false}"}`. Clients omit
values equal to the advertised defaults. Native Chat Completions fields keep
their normal API meaning and are not reused as graph context.

System instructions remain ordinary `system` messages in graph input. They are
independent of `ClientSettings` and do not appear in its discovery descriptor.

!!! warning "Do not publish internal context automatically"

    Keep user IDs, tenant identity, authorization state, database clients,
    secrets, and resource handles server-derived. Combine `client_config` with
    `context_factory(request, settings)` when the final runtime context also
    needs server-owned values; do not expose those values as public configuration.

See the runnable configuration in `demo/api/graphs/simple.py` and the exact
wire format in [OpenAI compatibility](../explanation/openai-compatibility.md#public-client-configuration).

## Async Factories

`GraphConfig.graph` may be a compiled graph, sync factory, or async factory:

```python title="Async graph factory"
async def advanced_graph():
    tools = await mcp_client.get_tools()
    return create_agent(model=model, tools=tools)

GraphConfig(graph=advanced_graph)
```

See `demo/api/graphs/advanced_mcp.py` for a mock MCP-style example.

## Register And Bind

```python title="Application registration"
from langgraph_openai_serve import GraphConfig, GraphRegistry, LanggraphOpenaiServe

graphs = GraphRegistry(
    registry={
        "my-graph": GraphConfig(graph=my_graph, streamable_node_names=["generate"]),
        "advanced-mcp-tools": GraphConfig(graph=advanced_graph),
    }
)

LanggraphOpenaiServe(graphs=graphs).bind_openai_api()
```

## Streaming

When an OpenAI request sets `stream=True`, LGOS forwards only streamed
`AIMessageChunk` values from `streamable_node_names`. Deterministic graphs that
return a final dictionary should be called without `stream=True`.

!!! tip "Choose streamable nodes deliberately"

    List only nodes whose model chunks should reach the client. This prevents
    internal graph work from appearing as assistant output.

## Interrupts

Enable the interrupt feature for checkpointed human-in-the-loop graphs:

```python
from langgraph_openai_serve import GraphConfig, GraphFeature

GraphConfig(
    graph=interruptible_graph,
    features={GraphFeature.INTERRUPTS},
)
```

Clients must pass `metadata.langgraph_thread_id` so follow-up tool messages
resume the same LangGraph thread. The interrupt is represented as an OpenAI tool
call named `langgraph_interrupt`.

!!! warning "A checkpointer is required"

    Interrupt-enabled graphs must be compiled with a checkpointer. Production
    deployments should use durable storage so a pending thread can resume after
    a process restart.

## Next Steps

- [OpenAI clients](openai-clients.md)
- [LangGraph integration](../explanation/langgraph-integration.md)
- [Reference](../reference.md)
