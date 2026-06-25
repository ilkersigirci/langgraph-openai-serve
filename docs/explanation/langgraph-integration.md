# LangGraph Integration

A registered graph name becomes the OpenAI `model` value clients pass to
`/v1/chat/completions`.

## Registration

`GraphRegistry` stores model names and `GraphConfig` values:

```python
GraphRegistry(
    registry={
        "chat": GraphConfig(graph=chat_graph, streamable_node_names=["generate"]),
        "advanced-mcp-tools": GraphConfig(graph=advanced_graph),
    }
)
```

`GraphConfig.graph` can be a compiled graph, sync factory, or async factory.
Async factories support setup such as MCP-style tool loading before creating a
ReAct graph.

## Adaptation

The default graph contract is:

- input: `{"messages": langchain_messages}`
- output text: `result["messages"][-1].content`

Use `request_to_input`, `context_factory`, and `output_to_text` when the graph
has custom input, output, or context schemas. Those adapters keep the public HTTP
surface OpenAI-compatible while letting the graph stay idiomatic LangGraph.

## Runner Behavior

For non-streaming requests, the runner resolves the graph, builds input and
context, then calls `ainvoke`.

For streaming requests, the runner calls `astream` with message and update
stream modes. It emits text from configured streamable nodes and converts
interrupt updates into OpenAI tool-call chunks.

See [Custom Graphs](../tutorials/custom-graphs.md) for runnable examples.
