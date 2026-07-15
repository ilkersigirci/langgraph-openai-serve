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

### Context Versus Config

LGOS keeps LangGraph's invocation channels separate:

```python
graph.astream(
    graph_input,
    context=runtime_context,
    config=runnable_config,
)
```

- `graph_input` contains mutable workflow state, including converted messages.
- `runtime_context` is built by `GraphConfig.context_factory`, validated against
  the graph's `context_schema`, and available to nodes as `runtime.context` on
  an injected `Runtime[Context]`.
- `runnable_config` contains callbacks and execution identity. When an OpenAI
  request supplies `metadata.langgraph_thread_id`, LGOS maps it to
  `config["configurable"]["thread_id"]` for the LangGraph checkpointer.

Application settings that nodes consume belong in typed runtime context, not in
the configurable section of `RunnableConfig`. The thread ID is different: the
checkpointer needs it to restore state before node execution, so it remains
execution configuration.

Because LGOS supports Python 3.11 and newer, callback/config context propagates
automatically to nested async runnable calls. Node functions only need an
injected `RunnableConfig` when they inspect or modify execution configuration;
they do not need one solely to pass `config` to a nested model's `ainvoke()`.

See [Custom Graphs](../tutorials/custom-graphs.md#runtime-context) for a complete
typed context example and LangGraph's official
[runtime](https://reference.langchain.com/python/langgraph/runtime/Runtime),
[streaming](https://docs.langchain.com/oss/python/langgraph/streaming), and
[persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
documentation for the underlying conventions.

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
