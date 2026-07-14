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

The OpenAI response mode and the LangGraph execution interface are separate.
Both paths call `graph.astream(..., version="v2")` so LGOS can process custom
events and interrupts during execution. Consequently, `stream=false` does not
mean LGOS calls `graph.ainvoke()`; it means LGOS consumes the internal events
before returning one HTTP response.

=== "Complete response"

    When `stream` is omitted or `false`, the route awaits `invoke_run()`. The
    runner consumes `values` and `custom` events and collects custom events from
    all namespaces. If interrupt support is enabled and an interrupt appears,
    it returns that interrupt immediately; otherwise it renders the latest
    root-namespace value after the graph stream ends. The chat service returns
    one OpenAI chat completion, so no graph events reach the HTTP client
    incrementally.

=== "SSE response"

    When `stream=true`, the route returns an SSE response backed by
    `stream_run()`. The runner consumes `messages` and `custom` events, plus
    `updates` for interrupt-enabled graphs. Only `AIMessageChunk` values from
    configured streamable nodes become text chunks. The chat service uses
    recognized citation events for final annotations and renders interrupts as
    tool-call chunks.

See [Custom Graphs](../tutorials/custom-graphs.md) for runnable examples.
