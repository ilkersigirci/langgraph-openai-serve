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
3.  Build optional LangGraph runtime context from the request.
4.  Render the graph's native output as OpenAI assistant text.

See `demo/api/graphs/custom_io.py` for the runnable version.

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

Streaming only forwards streamed `AIMessageChunk` values from
`streamable_node_names`. Deterministic graphs that return a final dictionary
should be called without `stream=True`.

!!! tip "Choose streamable nodes deliberately"

    List only nodes whose model chunks should reach the client. This prevents
    internal graph work from appearing as assistant output.

## Interrupts

Set `interrupts_enabled=True` for checkpointed human-in-the-loop graphs. Clients
must pass `metadata.langgraph_thread_id` so follow-up tool messages resume the
same LangGraph thread. The interrupt is represented as an OpenAI tool call named
`langgraph_interrupt`.

!!! warning "A checkpointer is required"

    Interrupt-enabled graphs must be compiled with a checkpointer. Production
    deployments should use durable storage so a pending thread can resume after
    a process restart.

## Next Steps

- [OpenAI clients](openai-clients.md)
- [LangGraph integration](../explanation/langgraph-integration.md)
- [Reference](../reference.md)
