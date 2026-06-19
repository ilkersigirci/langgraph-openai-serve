# LangGraph Integration

LangGraph OpenAI Serve bridges OpenAI-compatible chat requests to LangGraph
workflows. A registered graph name becomes the `model` value clients use in
`/v1/chat/completions`.

## Registration

Graphs are registered with `GraphRegistry`:

```python
from langgraph_openai_serve import GraphConfig, GraphRegistry

graphs = GraphRegistry(
    registry={
        "chat": GraphConfig(
            graph=chat_graph,
            streamable_node_names=["generate"],
        ),
        "advanced-mcp-tools": GraphConfig(graph=advanced_graph),
    }
)
```

`GraphConfig.graph` can be either:

- a compiled LangGraph graph
- a sync factory that returns a compiled graph
- an async factory that returns a compiled graph

The async factory form is useful when graph construction needs async setup, such
as loading tools from MCP. See `demo/api/graphs/advanced_mcp.py`.

## Request Adaptation

Incoming OpenAI messages are converted to LangChain messages before execution.
By default, the graph input is:

```python
{"messages": langchain_messages}
```

By default, response text is read from:

```python
result["messages"][-1].content
```

Graphs with custom LangGraph schemas can override this with `GraphConfig`
adapters:

```python
GraphConfig(
    graph=graph,
    request_to_input=request_to_input,
    context_factory=context_factory,
    output_to_text=output_to_text,
)
```

Those adapters let a graph use native `input_schema`, `output_schema`, and
`context_schema` while still serving OpenAI-compatible requests. See
`demo/api/graphs/custom_io.py`.

## Non-Streaming Execution

For normal requests, the runner resolves the graph, builds input and context,
then calls LangGraph with `ainvoke`:

```python
graph = await graph_config.resolve_graph()
inputs = await graph_config.build_input(request, langchain_messages)
context = await graph_config.build_context(request)

result = await graph.ainvoke(inputs, config=runnable_config, context=context)
response_text = await graph_config.render_output(result)
```

## Streaming Execution

For streaming requests, the runner uses LangGraph message streaming:

```python
async for event in graph.astream(
    inputs,
    config=runnable_config,
    context=context,
    stream_mode=["messages"],
    subgraphs=True,
    version="v2",
):
    ...
```

Only `AIMessageChunk` values from configured `streamable_node_names` are sent to
the client. Hidden-tagged chunks are ignored.

This means streaming is opt-in at registration time and depends on the graph
actually emitting streamed message chunks. The custom input/output/context demo
returns a final deterministic value, so it is intended for non-streaming calls.

## Demo Coverage

The demo app covers the important integration paths:

- `simple-graph-with-history` and `simple-graph-no-history` use the default
  message graph contract.
- `custom-input-output-context` uses custom input, output, and runtime context
  adapters.
- `advanced-mcp-tools` uses an async graph factory that loads mock MCP-style
  tools before building a ReAct graph.

## Next Steps

- [Custom graphs tutorial](../tutorials/custom-graphs.md)
- [OpenAI compatibility](openai-compatibility.md)
- [API reference](../reference.md)
