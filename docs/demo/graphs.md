# Example Graphs

The demo API registers the following graphs. They demonstrate LGOS features;
none is installed as a built-in model by the `langgraph-openai-serve` package.
Each registration also supplies a short `GraphConfig.description` used by the
demo model catalogs.

| Model | Demonstrates | Graph feature | Extra runtime requirement |
| --- | --- | --- | --- |
| `custom-input-output-context` | Request, output, and typed runtime-context adapters | None | None |
| `citation-events` | Structured OpenAI URL citations with portable Markdown content | None | None |
| `advanced-mcp-tools` | Async graph factories and a mock MCP-style tool | None | None |
| `complex-subgraphs` | Router-selected subgraphs and nested streamed output | None | None |
| `status-events` | Portable status updates for native client UI | `client_events` | None |
| `custom-event-showcase` | Public progress and artifact events interleaved with text | `client_events` | None |
| `interruptible-approval` | Checkpointed human approval represented as an OpenAI tool call | `interrupts` | PostgreSQL checkpointer and run coordinator |
| `simple-graph` | Streamed model output and discoverable runtime settings | None | Upstream chat model |
| `lgos-rag` | Agentic retrieval over the packaged demo corpus | None | Upstream chat and embedding models |

The demo API opens its PostgreSQL checkpointer during application startup, so
PostgreSQL must be available even when you call a provider-free graph. Start it
with the [demo API instructions](api.md#start-postgresql-and-the-api).

## Interrupt Runtime

`interruptible-approval` is the only demo graph that persists API execution
state; ordinary chat history remains client-owned. The demo uses LGOS's default
shared checkpoint scope, so multi-tenant applications must instead derive that
scope from authenticated server state. Operation identity, canonical replay,
and retention rules are defined in
[OpenAI Compatibility](../explanation/openai-compatibility.md#tool-calls-and-interrupts).

Each demo API process opens one PostgreSQL connection pool in its FastAPI
lifespan, waits for the pool to become ready before serving, and closes it at
shutdown. The checkpointer and same-run coordinator share that pool. The latter
holds a
session-level [PostgreSQL advisory lock](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS)
only while validating or advancing a request, never while awaiting human input.
One of the pool's five connections is reserved for checkpoint I/O; exhausting
the other four coordination slots returns HTTP 409.

The demo environment enables `LANGGRAPH_STRICT_MSGPACK=true`. This selects
LangGraph's strict allowlist policy for checkpoint deserialization; it does not
replace database access controls or integrity monitoring. See the upstream
[LangGraph security advisory](https://github.com/langchain-ai/langgraph/security/advisories/GHSA-g48c-2wqr-h844).
The demo has no expiry worker; production deployments must reap abandoned
pending runs and follow LangGraph's
[interrupt idempotency rules](https://docs.langchain.com/oss/python/langgraph/interrupts#rules-of-interrupts).

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
