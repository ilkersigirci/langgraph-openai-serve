# Persistent Plot Agent

`persistent-plot-agent` demonstrates durable, thread-scoped application data behind
the stateless LGOS `/v1` API. The UI owns the chat transcript and supplies each
request's model context; the graph stores only the canonical quarterly revenue
values in LangGraph's
[`AsyncPostgresStore`](https://reference.langchain.com/python/langgraph.store.postgres/aio/AsyncPostgresStore).

It is a real LangChain [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents)
agent with two typed tools:

- `show_quarterly_revenue` reads and displays the current chart.
- `update_quarterly_revenue` applies all absolute-value edits from one turn and
  displays the result.

The tools receive the Store, request context, and stream writer through native
[`ToolRuntime`](https://docs.langchain.com/oss/python/langchain/runtime). The
model chooses a tool, but only the tools read or change stored values.

## LangGraph Topology

```mermaid
graph TD;
	__start__ --> model;
	model -.-> __end__;
	model -.-> tools;
	tools -.-> model;
```

## Request Flow

Both Chainlit and Open WebUI send their current model-context messages, a
UI-provided user identifier as OpenAI `user`, and their stable thread or chat
identifier as `metadata.session_id`.

```mermaid
sequenceDiagram
  participant UI as Chainlit / Open WebUI
  participant LGOS as LGOS /v1
  participant Agent as Agent and tools
  participant Store as AsyncPostgresStore

  UI->>LGOS: messages + user + session_id
  LGOS->>LGOS: validate settings and build request context
  LGOS->>Agent: messages + request context
  Agent->>Store: aget chart document
  alt update requested and values changed
    Agent->>Store: aput complete document once
  end
  Agent-->>LGOS: semantic chart event + assistant text
  LGOS-->>UI: OpenAI response or stream
```

For each request:

1. LGOS validates `user`, `metadata.session_id`, and the request-scoped chart
   settings.
2. The agent selects the read or update tool from the user's natural-language
   request.
3. The tool loads the chart with `store.aget()`. A missing document uses the
   schema defaults without writing them.
4. An update batches every requested assignment, validates the complete
   document, and calls `store.aput()` once only when a value changed.
5. The tool publishes the resulting chart and returns the current values to the
   model, which produces the ordinary assistant answer.

For example, `Which quarter is highest?` reads without writing. `Set Q1 to 200
and Q3 to 250` reads once and writes one updated document. The current tool
contract supports absolute assignments; relative edits such as `increase Q3 by
10%` are outside this demo's contract.

## Store Scope

A LangGraph Store addresses a JSON-like value by `namespace` and `key`. The
demo uses:

```text
namespace = ("demo", "persistent-plot-agent", "threads", sha256(user + "\0" + session_id))
key       = "quarterly-revenue"
value     = {"schema_version": 1, "q1": 120, "q2": 180, "q3": 150, "q4": 230}
```

The hash keeps raw identifiers out of the namespace. API processes sharing the
demo database select the same document for the same user and session; changing
either value selects an independent document. The API setup command creates the
Store schema, and each process uses its lifespan-managed PostgreSQL pool for
Store operations.

`AsyncPostgresStore` replaces the document with one atomic PostgreSQL
[`INSERT ... ON CONFLICT DO UPDATE`](https://www.postgresql.org/docs/current/sql-insert.html#SQL-ON-CONFLICT).
The tool's complete `aget()` → merge → `aput()` sequence is not a conditional
update, so concurrent edits to the same chart are last-write-wins. This demo
keeps that policy explicit instead of holding a database connection throughout
an agent run. An application that requires collaborative editing should add a
domain-specific revision check at its persistence boundary.

This document is long-term application data, not conversation state or a
[LangGraph checkpoint](interruptible-approval.md). See LangGraph's
[Store concepts](https://docs.langchain.com/oss/python/langgraph/persistence#memory-store)
for the native namespace, key, and value model.

## UI Rendering

Both tools emit a small, versioned `kind=chart` client event containing labels,
series, titles, and presentation settings. The graph does not emit a serialized
Plotly figure:

- Chainlit converts the event to its native
  [`Plotly`](https://docs.chainlit.io/api-reference/elements/plotly) element;
  see [Chainlit streaming and events](../chainlit.md#streaming-events-and-citations).
- Open WebUI converts it to Plotly.js and sends its native persistent
  [`embeds` event](https://docs.openwebui.com/features/extensibility/plugin/development/events/#embeds-or-chatmessageembeds);
  see [Open WebUI streaming and events](../open-webui.md#streaming-status-and-citations).

Those UI-native records are rendered copies, not replacements for the
canonical Store document.

Rich chart rendering requires a streaming request that opts into
`metadata.langgraph_stream_events="v1"`. The assistant answer remains standard
streamed OpenAI text. A non-streaming request still reads or updates the Store,
but returns only the assistant answer.

This boundary generalizes beyond charts:

- Keep canonical application data in the graph's Store.
- Inline a small, versioned semantic view when it is cheap to transport.
- For a large or binary artifact, emit an opaque ID or authorized expiring URL
  instead of the body.
- Let each UI persist its own rendered message or element.

The UIs should not read LangGraph's PostgreSQL tables directly. Direct reads
couple them to LangGraph's storage schema, bypass the API's authorization
boundary, and create a second data-access contract.

## Ownership Boundaries

| State | Owner | Lifetime |
| --- | --- | --- |
| Transcript and rendered chart | Chainlit or Open WebUI | UI-defined |
| Quarterly revenue values | `AsyncPostgresStore` | Across requests and API restarts |
| Chart type, currency label, and legend visibility | UI request settings | One request; the UI resends them |
| Graph execution | LGOS | One request |

See [Demo Architecture](../architecture.md#state-ownership) for the physical
services and database layout.

!!! warning "Correlation is not authorization"

    `user` and `session_id` are request correlation values. A production
    application must derive them from authenticated server state and define a
    retention policy for stored chart documents.

## Try It

In one Chainlit thread or Open WebUI chat, send:

1. `Show the chart.`
2. `Set Q1 to 200 and Q3 to 250.`
3. `Which quarter is highest?`

The third request reads the values written by the second. Start another thread
or chat to select an independent Store document.
