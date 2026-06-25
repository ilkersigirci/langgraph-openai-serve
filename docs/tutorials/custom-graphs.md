# Custom Graphs

LangGraph OpenAI Serve maps OpenAI `model` names to LangGraph configurations.
The easiest way to learn the supported graph shapes is to inspect the demo
graphs in `demo/api/graphs`.

## Default Message Graph

Most chat graphs can use the default shape:

```python
GraphConfig(
    graph=my_graph,
    streamable_node_names=["generate"],
)
```

With no adapters, the runner sends this input to the graph:

```python
{"messages": langchain_messages}
```

The response text is read from:

```python
result["messages"][-1].content
```

See `demo/api/graphs/simple.py` and the underlying package graph for a complete
message-based example.

## Custom Input, Output, And Context

Use adapters when your graph declares native LangGraph input, output, or context
schemas:

```python
GraphConfig(
    graph=custom_io_graph,
    request_to_input=request_to_input,
    context_factory=context_factory,
    output_to_text=output_to_text,
)
```

The adapter hooks are:

- `request_to_input(request, messages)` converts the OpenAI request into the
  graph input schema.
- `context_factory(request)` builds the LangGraph runtime context.
- `output_to_text(output)` converts the graph output schema into the assistant
  message text.

See `demo/api/graphs/custom_io.py` for the complete runnable example.

## Async Graph Factories And MCP-Style Tools

`GraphConfig.graph` can be a compiled graph or a callable that returns one. If
the callable is async, the server awaits it before execution. That is useful for
loading tools from MCP or another async source:

```python
async def advanced_graph():
    tools = await mcp_client.get_tools()
    return create_agent(model=model, tools=tools)

GraphConfig(graph=advanced_graph)
```

See `demo/api/graphs/advanced_mcp.py` for a mock MCP client that loads tools
asynchronously and builds a ReAct graph without requiring real MCP servers or API
keys.

## Register A Graph

Register graphs in a `GraphRegistry`; each key becomes an OpenAI model name:

```python
from langgraph_openai_serve import GraphConfig, GraphRegistry, LangchainOpenaiApiServe

graphs = GraphRegistry(
    registry={
        "my-graph": GraphConfig(
            graph=my_graph,
            streamable_node_names=["generate"],
        ),
        "advanced-mcp-tools": GraphConfig(graph=advanced_graph),
    }
)

server = LangchainOpenaiApiServe(graphs=graphs)
server.bind_openai_chat_completion()
```

The default OpenAI-compatible prefix is `/v1`. Set `LGOS_OPENAI_API_PREFIX` or
pass `prefix=` to mount the API elsewhere.
FastAPI docs for the mounted OpenAI API are disabled by default; set
`LGOS_OPENAI_API_DOCS_ENABLED=true` to inspect `/v1/docs` locally.

The demo registration lives in `demo/api/app.py`.

## Streaming

Streaming only emits chunks from nodes listed in `streamable_node_names`, and
the graph must actually produce streamed `AIMessageChunk` values. A deterministic
graph that returns one final dictionary should be called without `stream=True`.

## Next Steps

- [Connect OpenAI clients](openai-clients.md)
- [Run with Docker](../how-to-guides/docker.md)
- [LangGraph integration details](../explanation/langgraph-integration.md)
