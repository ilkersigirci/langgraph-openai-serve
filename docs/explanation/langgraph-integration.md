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
- `runtime_context` starts with optional validated `GraphConfig.client_settings`
  settings and can be composed with server-owned values by
  `GraphConfig.context_factory(request, settings)`. When the graph declares a
  `context_schema`, LGOS validates the final value against it. Nodes receive the
  result as `runtime.context` on an injected `Runtime[Context]`.
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

See [Custom Graphs](../tutorials/custom-graphs.md#runtime-context) for a typed
server-owned context example and
[Configure LangGraph Runtime Settings](../how-to-guides/langgraph-runtime-settings.md)
for public settings, discovery, and request handling. LangGraph's official
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

## Request Cancellation

For streaming chat completions (`stream=true`), LGOS ties graph iteration to the
HTTP response lifetime. The route returns Starlette's ordinary
`StreamingResponse`, while a request-scoped FastAPI `yield` dependency owns one
`asyncio` producer task and the AnyIO memory channel feeding that response.
Closing the client stream—including clicking **Stop** in the Chainlit
demo—is detected by `StreamingResponse`, which ends the response. Dependency
teardown then cancels and awaits the producer and closes the nested graph
iterator. This uses the normal OpenAI streaming connection; it adds no custom
cancellation route, header, or SSE event.

!!! info "Why cancellation raises the dependency minimums"

    LGOS requires `fastapi[standard]>=0.121.0` because stream ownership depends
    on a request-scoped `yield` dependency remaining alive until after
    `StreamingResponse` finishes. FastAPI restored post-response cleanup for
    streaming dependencies in 0.118.0 and added explicit
    `Depends(scope="request")` support in 0.121.0, making that the functional
    compatibility floor.

    FastAPI 0.139 is important because it introduced native SSE support built
    around a request-scoped producer, an AnyIO memory channel, and an ordinary
    `StreamingResponse`. That implementation inspired LGOS's architecture and
    confirms this lifecycle as a framework-supported pattern. LGOS cannot use it
    directly because `/v1/chat/completions` must dynamically return either JSON
    or pre-framed OpenAI SSE from the same route, so 0.139 is not the functional
    minimum. See FastAPI's [SSE documentation](https://fastapi.tiangolo.com/tutorial/server-sent-events/)
    and
    [dependency lifecycle notes](https://fastapi.tiangolo.com/advanced/advanced-dependencies/#dependencies-with-yield-and-streamingresponse-technical-details).

    LGOS also depends directly on `anyio>=4,<5` for its
    [memory object stream](https://anyio.readthedocs.io/en/stable/streams.html#memory-object-streams)
    and shielded
    [cancellation scope](https://anyio.readthedocs.io/en/stable/cancellation.html#shielding).
    The lower bound selects the AnyIO major version against which teardown is
    implemented and tested; the upper bound prevents an unreviewed future major
    release from changing cancellation or stream behavior underneath this
    lifecycle. FastAPI continues to select its compatible Starlette version.

The Chainlit demo closes its OpenAI stream when the message handler is cancelled.
Any partial assistant text remains visible but is excluded from later model
context because it is an incomplete response.

!!! warning "Cancellation is cooperative"

    Asynchronous graph and model work stops at cancellation points. Synchronous,
    CPU-bound, blocking, or cancellation-swallowing code may continue, and
    blocked cleanup can delay request teardown. A proxy must propagate the
    downstream disconnect to LGOS; a proxy that continues consuming the upstream
    response also keeps the graph request alive. An upstream provider decides
    whether closing its own connection stops remote generation or billing. This
    request-scoped path does not cover `stream=false`.

!!! note "Not durable run cancellation"

    LGOS does not create an addressable run record or expose cancellation by run
    ID. Disconnect cancellation also does not create a resumable LangGraph
    interrupt. By contrast, [LangSmith Agent Server](https://docs.langchain.com/langsmith/agent-server)
    persists queued and running work and provides an explicit
    [run cancellation API](https://docs.langchain.com/langsmith/cancel-run) that
    can preserve or roll back checkpoints. Use that runtime when cancellation
    must remain available after the original HTTP request is gone.
