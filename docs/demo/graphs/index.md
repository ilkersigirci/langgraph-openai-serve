# Example Graphs

The demo API registers the following graphs. They demonstrate LGOS features;
none is installed as a built-in model by the `langgraph-openai-serve` package.
Each registration also supplies a short `GraphConfig.description` used by the
demo model catalogs.

| Model | Demonstrates | Graph feature | Graph-specific dependency |
| --- | --- | --- | --- |
| [`custom-input-output-context`](core-patterns.md#custom-input-output-context) | Request, output, and typed runtime-context adapters | None | None |
| [`citation-events`](events-and-citations.md) | Structured OpenAI URL citations with portable Markdown content | None | None |
| [`file-input`](file-input.md) | Central Files API IDs resolved into OpenAI Responses file inputs | `file_inputs` | Central Files API and upstream Responses model |
| [`advanced-mcp-tools`](core-patterns.md#advanced-mcp-tools) | Async graph factories and a mock MCP-style tool | None | None |
| [`complex-subgraphs`](complex-subgraphs.md) | Router-selected subgraphs, status, and nested streamed output | `client_events` | None |
| [`multi-node-streaming`](core-patterns.md#multi-node-streaming) | Two sequential fake-model nodes contributing ordered text to one assistant message | None | None |
| [`status-events`](events-and-citations.md) | Portable status updates for native client UI | `client_events` | None |
| [`custom-event-showcase`](events-and-citations.md) | Public progress and artifact events interleaved with text | `client_events` | None |
| [`persistent-plot-agent`](persistent-plot-agent.md) | A tool-calling agent with an editable thread-scoped chart | `client_events` | Upstream chat model and PostgreSQL store |
| [`interruptible-approval`](interruptible-approval.md) | Durable choice-or-text human review before protected actions | `interrupts` | PostgreSQL checkpointer and run coordinator |
| [`simple-graph`](core-patterns.md#simple-graph) | Streamed model output and discoverable runtime settings | None | Upstream chat model |
| [`simple-graph-external-tools`](core-patterns.md#simple-graph-external-tools) | Client-provided function tools returned as model tool calls | None | Upstream chat model |
| [`lgos-rag`](lgos-rag.md) | Agentic retrieval over the packaged demo corpus | None | Upstream chat and embedding models |

The demo API opens its PostgreSQL runtime during application startup, so
PostgreSQL must be available even when you call a provider-free graph. Start it
with the [demo API instructions](../api.md#start-postgresql-and-the-api).

`persistent-plot-agent` stores application data with a LangGraph Store.
`interruptible-approval` checkpoints graph execution. Neither mechanism makes
LGOS the owner of UI conversation history.

## Source Map

All graph code is owned by the independent `demo/api` project:

- `demo/api/src/lgos_demo_api/app.py` registers graph names as OpenAI model
  names.
- `demo/api/src/lgos_demo_api/graphs/` contains every graph and adapter listed
  above.
- `demo/api/src/lgos_demo_api/corpus/` contains the Markdown packaged with the
  `lgos-rag` example.

Continue with [Run the Demo API](../api.md#call-a-graph) for request examples or
[Chainlit](../chainlit.md) and [Open WebUI](../open-webui.md) for UI behavior.
