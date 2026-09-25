# Example Graphs

The demo API registers the following graphs. They demonstrate LGOS features;
none is installed as a built-in model by the `langgraph-openai-serve` package.
Each registration also supplies a short `GraphConfig.description` used by the
demo model catalogs.

| Model | Demonstrates | Graph feature | Graph-specific dependency |
| --- | --- | --- | --- |
| [`advanced-graph`](advanced-graph.md) | General chat, gateway tools, uploaded-file Q&A, routed cited research, and approval before saving a searchable note, in the foreground or background | `background`, `client_events`, `file_inputs`, `interrupts`, `mcp_tools` | Responses model, selected gateway, OpenAI-compatible vector service, and PostgreSQL |
| [`background-mock`](background-mock.md) | Deterministic background execution in an independently deployed worker, with no model call | `background` | Hatchet |
| [`custom-input-output-context`](core-patterns.md#custom-input-output-context) | Request, output, and typed runtime-context adapters | None | None |
| [`citation-events`](events-and-citations.md) | Structured OpenAI URL citations with portable Markdown content | None | None |
| [`file-input`](file-input.md) | Central Files API IDs resolved into OpenAI Responses file inputs | `file_inputs` | Central Files API and upstream Responses model |
| [`mcp-mock`](core-patterns.md#mcp-mock) | Async MCP-style tool discovery and an agent tool loop | None | None |
| [`mcp-postgres`](mcp-postgres.md) | Read-only database questions with MCP discovery and execution owned by the native UI client | `mcp_tools` | Upstream model, selected gateway, DBHub, and PostgreSQL |
| [`complex-subgraphs`](complex-subgraphs.md) | Router-selected subgraphs, status, and nested streamed output | `client_events` | None |
| [`custom-event-showcase`](events-and-citations.md) | Filtering internal progress and artifact events at the API boundary | `client_events` | None |
| [`multi-node-streaming`](core-patterns.md#multi-node-streaming) | Two sequential fake-model nodes contributing ordered text to one assistant message | None | None |
| [`response-outcomes`](core-patterns.md#response-outcomes) | Native refusal content and incomplete terminal responses | None | None |
| [`status-events`](events-and-citations.md) | Portable status updates for native client UI | `client_events` | None |
| [`persistent-plot-agent`](persistent-plot-agent.md) | A tool-calling agent with an editable thread-scoped chart | None | Upstream model, Files API, and PostgreSQL store |
| [`interruptible-approval`](interruptible-approval.md) | Durable choice-or-text human review before protected actions | `interrupts` | PostgreSQL checkpointer and run coordinator |
| [`simple-graph`](core-patterns.md#simple-graph) | Streamed model output and discoverable runtime settings | None | Upstream chat model |
| [`simple-graph-external-tools`](core-patterns.md#simple-graph-external-tools) | Client-provided function tools returned as model tool calls | None | Upstream chat model |
| [`server-tool`](server-tool.md) | Installed package versions and OpenAI-compatible web search selected by the client | None | Upstream model plus SearXNG, Degoog, or upstream OpenAI search |
| [`lgos-rag`](lgos-rag.md) | Agentic retrieval with structured URL citations over the packaged demo corpus | `client_events` | Upstream chat and embedding models |

The demo API opens its PostgreSQL runtime during application startup, so
PostgreSQL must be available even when you call a provider-free graph. Start it
with the [demo API instructions](../api.md#start-postgresql-and-the-api).

`persistent-plot-agent` stores application data with a LangGraph Store.
`interruptible-approval` checkpoints graph execution. `advanced-graph` uses both:
checkpoints for human review and Store receipts for vector-service uploads. None
makes LGOS the owner of UI conversation history.

Background Responses from `advanced-graph` and `background-mock` are
polling-only and stored by Hatchet. They still do not persist a UI conversation
or expose event replay.

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
