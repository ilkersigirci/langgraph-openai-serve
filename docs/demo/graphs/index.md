# Example Graphs

The demo API registers the following graphs. They demonstrate LGOS features;
none is installed as a built-in model by the `langgraph-openai-serve` package.
Each registration also supplies a short `GraphConfig.description` used by the
demo model catalogs.

| Model | Demonstrates | Graph feature | Extra runtime requirement |
| --- | --- | --- | --- |
| `custom-input-output-context` | Request, output, and typed runtime-context adapters | None | None |
| [`citation-events`](events-and-citations.md) | Structured OpenAI URL citations with portable Markdown content | None | None |
| `advanced-mcp-tools` | Async graph factories and a mock MCP-style tool | None | None |
| [`complex-subgraphs`](complex-subgraphs.md) | Router-selected subgraphs and nested streamed output | None | None |
| [`status-events`](events-and-citations.md) | Portable status updates for native client UI | `client_events` | None |
| [`custom-event-showcase`](events-and-citations.md) | Public progress and artifact events interleaved with text | `client_events` | None |
| [`persistent-plot`](persistent-plot.md) | An editable thread-scoped chart | `client_events` | PostgreSQL store |
| [`interruptible-approval`](interruptible-approval.md) | One checkpointed batch from parallel nested approval subgraphs | `interrupts` | PostgreSQL checkpointer and run coordinator |
| `simple-graph` | Streamed model output and discoverable runtime settings | None | Upstream chat model |
| [`lgos-rag`](lgos-rag.md) | Agentic retrieval over the packaged demo corpus | None | Upstream chat and embedding models |

The demo API opens its PostgreSQL runtime during application startup, so
PostgreSQL must be available even when you call a provider-free graph. Start it
with the [demo API instructions](../api.md#start-postgresql-and-the-api).

`persistent-plot` stores application data with a LangGraph Store.
`interruptible-approval` checkpoints graph execution. Neither mechanism makes
LGOS the owner of UI conversation history.

!!! tip "Start without provider credentials"

    Use `custom-input-output-context`, `citation-events`,
    `advanced-mcp-tools`, `complex-subgraphs`, `status-events`,
    `custom-event-showcase`, `persistent-plot`, or `interruptible-approval` to
    explore the transport without a real model API key.

## Source Map

All graph code is owned by the independent `demo/api` project:

- `demo/api/src/lgos_demo_api/app.py` registers graph names as OpenAI model
  names.
- `demo/api/src/lgos_demo_api/graphs/simple.py` publishes safe runtime settings
  for conversation history and intended audience.
- `demo/api/src/lgos_demo_api/graphs/lgos_rag.py` implements agentic retrieval,
  relevance grading, bounded rewriting, and grounded streamed answers.
- `demo/api/src/lgos_demo_api/corpus/` contains the Markdown embedded in source
  installs, wheels, and API images.
- `demo/api/src/lgos_demo_api/graphs/custom_io.py` contains input, output, and
  context adapters.
- `demo/api/src/lgos_demo_api/graphs/advanced_mcp.py` constructs an agent from
  an async factory and mock tool.
- `demo/api/src/lgos_demo_api/graphs/complex_subgraphs.py` and
  `graphs/subgraphs/` implement router-selected specialists.
- `demo/api/src/lgos_demo_api/graphs/status_events.py` emits portable status
  updates.
- `demo/api/src/lgos_demo_api/graphs/custom_events.py` emits explicitly public
  progress and artifact events.
- `demo/api/src/lgos_demo_api/graphs/persistent_plot.py` edits a thread-scoped
  chart document and emits the current chart on every call.
- `demo/api/src/lgos_demo_api/graphs/interruptible.py` runs reusable nested
  approval steps as one checkpointed batch.
- `demo/api/src/lgos_demo_api/graphs/citations.py` emits citation events that
  LGOS maps to OpenAI annotations.

Continue with [Run the Demo API](../api.md#call-a-graph) for request examples or
[Chainlit](../chainlit.md) and [Open WebUI](../open-webui.md) for UI behavior.
