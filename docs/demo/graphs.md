# Example Graphs

The demo API registers the following graphs. They demonstrate LGOS features;
none is installed as a built-in model by the `langgraph-openai-serve` package.

| Model | Demonstrates | Extra runtime requirement |
| --- | --- | --- |
| `custom-input-output-context` | Request, output, and typed runtime-context adapters | None |
| `citation-events` | Structured OpenAI URL citations with portable Markdown content | None |
| `advanced-mcp-tools` | Async graph factories and a mock MCP-style tool | None |
| `complex-subgraphs` | Router-selected subgraphs and nested streamed output | None |
| `status-events` | Portable status updates for native client UI | None |
| `custom-event-showcase` | Public progress and artifact events interleaved with text | None |
| `interruptible-approval` | Checkpointed human approval represented as an OpenAI tool call | PostgreSQL |
| `simple-graph` | Streamed model output and discoverable runtime settings | Upstream chat model |
| `lgos-rag` | Agentic retrieval over the packaged demo corpus | Upstream chat and embedding models |

The demo API opens its PostgreSQL checkpointer during application startup, so
PostgreSQL must be available even when you call a provider-free graph. Start it
with the [demo API instructions](api.md#start-postgresql-and-the-api).

!!! tip "Start without provider credentials"

    Use `custom-input-output-context`, `citation-events`,
    `advanced-mcp-tools`, `complex-subgraphs`, `status-events`,
    `custom-event-showcase`, or `interruptible-approval` to explore the
    transport without a real model API key.

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
- `demo/api/src/lgos_demo_api/graphs/interruptible.py` pauses and resumes a
  checkpointed approval flow.
- `demo/api/src/lgos_demo_api/graphs/citations.py` emits citation events that
  LGOS maps to OpenAI annotations.

Continue with [Run the Demo API](api.md#call-a-graph) for request examples or
[Chainlit](chainlit.md) and [Open WebUI](open-webui.md) for UI behavior.
