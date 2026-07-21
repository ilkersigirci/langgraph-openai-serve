# Reference

## OpenAI-Compatible API

Default prefix: `/v1`. Change it with `LGOS_OPENAI_API_PREFIX` or
`bind_openai_api(prefix=...)`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | List standard summaries for registered graph models. |
| `GET` | `/v1/models/{model}` | Retrieve one model with optional LGOS discovery metadata. |
| `POST` | `/v1/chat/completions` | Run a graph through OpenAI chat completions. |
| `GET` | `/v1/health` | Health check. |

FastAPI docs for the mounted OpenAI app are disabled by default. Set
`LGOS_OPENAI_API_DOCS_ENABLED=true` to expose `{prefix}/docs`, `{prefix}/redoc`,
and `{prefix}/openapi.json`.

## Settings

Package settings:

| Setting | Default | Notes |
| --- | --- | --- |
| `LGOS_OPENAI_API_PREFIX` | `/v1` | Must start with `/`; trailing slash is normalized. |
| `LGOS_OPENAI_API_DOCS_ENABLED` | `false` | Enables docs only for the mounted OpenAI app. |

Settings prefixed with `DEMO_` belong to the independent example applications
and are documented under [Demo Settings and Commands](demo/reference.md).

## Public API

Use `LanggraphOpenaiServe` to bind OpenAI-compatible routes to a FastAPI app.
Use `GraphRegistry` to map OpenAI `model` names to `GraphConfig` values.
The registry must contain at least one graph. Pydantic rejects empty registries
and model IDs that cannot be addressed as one URL path segment. Registry keys
are read-only after validation; use `registry.register(model_id, config)` to add
or replace a graph.

`GraphConfig` accepts:

- `graph`: compiled graph, sync factory, or async factory.
- `streamable_node_names`: node names whose streamed `AIMessageChunk` values are
  forwarded to clients.
- `features`: `GraphFeature` values that enable optional server behavior.
- `client_settings`: explicit public `ClientSettings` model class advertised by
  model retrieval.
- `runtime_callbacks`: callbacks included in the LangGraph `RunnableConfig`.
- `request_to_input(request, messages)`: custom OpenAI request to graph input.
- `context_factory(request, client_settings)`: compose the final typed LangGraph
  runtime context from server-owned values and optional validated public settings.
- `output_to_text(output)`: custom graph output to assistant text.

When both are configured, LGOS validates the public settings first and passes
them to `context_factory`. Without a factory, the validated settings instance is
the runtime context, so the graph must use that settings model as its
`context_schema`. A factory may return `None`; every non-null result requires a
graph context schema. LGOS passes server-owned factory results to LangGraph
without rebuilding them. LangGraph's native
[runtime-context handling](https://docs.langchain.com/oss/python/langgraph/graph-api#runtime-context)
constructs mapping values through dataclass and Pydantic context schemas and
trusts existing instances. The factory owns the validity of instances it
creates. Graphs should access context from an injected `Runtime[Context]`.
Runtime context is separate from `RunnableConfig`:

| Value | LGOS/LangGraph path | Intended use |
| --- | --- | --- |
| Graph input | `graph.astream(input, ...)` | Messages and mutable workflow state. |
| Runtime context | public settings → optional `context_factory` → `context=` → `Runtime.context` | Immutable per-run application values and dependencies. |
| Runnable config | `config=` | Callbacks, tags, tracing, and other execution controls. |
| Checkpoint thread | `metadata.langgraph_thread_id` → `config["configurable"]["thread_id"]` | Load, save, interrupt, and resume checkpoint state. |

LGOS assembles runnable config from `runtime_callbacks` and, when present, the
checkpoint thread ID. There is intentionally no adapter for placing arbitrary
OpenAI request fields into `config["configurable"]`; use typed runtime context
for values consumed by nodes.

The same `features` set drives runtime behavior and the versioned
`langgraph_openai_serve.features` extension returned by
`GET /v1/models/{model}`. `GraphFeature.INTERRUPTS` enables and advertises the
interrupt/resume flow.

### Runtime Settings

Subclass `ClientSettings` to publish only fields deliberately selected by the
server author. LGOS never inspects or publishes the LangGraph context schema:

```python title="Public settings model"
from pydantic import Field

from langgraph_openai_serve import ClientSettings


class PublicSettings(ClientSettings):
    use_history: bool = Field(default=True, title="Use conversation history")
```

Pass this model as `GraphConfig.client_settings` and use it as the graph's context
schema when it is the complete runtime context. Every public field must have a
default. Registration rejects subclasses that change the inherited strict,
frozen, extra-forbid, or default-validation behavior, as well as fields excluded
from Pydantic serialization.

All public fields travel together as compact JSON text in the
`metadata.langgraph_runtime_settings` string. Clients omit values equal to the advertised
defaults. System instructions remain ordinary OpenAI messages and are
independent of `ClientSettings`; native Chat Completions fields keep their
standard request semantics.

LGOS validates defaults and generates the discovery JSON Schema when the graph
is registered, then validates settings on every request. Without
`context_factory`, the settings become `Runtime.context`. A factory can instead
combine them with server-derived identity, authorization, database clients, and
other dependencies.

The serialized descriptor appears only on model retrieval as
`langgraph_openai_serve.client_settings`, with independent `schema_version`,
`json_schema`, and `defaults` fields. All client settings use the fixed
`metadata.langgraph_runtime_settings` envelope. Clients use the descriptor's
validated `defaults` object as the baseline; `default` keywords within the
generated JSON Schema are annotations, not the runtime baseline.

See [Configure LangGraph Runtime Settings](how-to-guides/langgraph-runtime-settings.md)
for the runtime settings flow, and
[Runtime Settings](explanation/openai-compatibility.md#runtime-settings) for the
request lifecycle.

Interrupt-enabled graphs must be compiled with a LangGraph checkpointer and
requests must include `metadata={"langgraph_thread_id": "<client-chat-id>"}`.
Use a durable checkpointer in production.

## Client Stream Events

Inside a graph node or tool, mark a passive client notification as explicitly
public with `client_event()`:

```python
from langgraph.config import get_stream_writer
from langgraph_openai_serve import client_event

get_stream_writer()(
    client_event(
        "progress",
        {
            "stage": "retrieval",
            "completed": 2,
            "total": 5,
            "message": "Searching documents",
        },
        namespace=("research",),
    )
)
```

The v1 vocabulary is `status`, `progress`, and `artifact`. Event data must be
JSON-safe, and every namespace segment must be a string. Keep payloads small and
represent large artifacts by an ID or URL. The namespace is a stable,
author-defined path; LGOS does not expose LangGraph's dynamic execution
namespace.

Events are streaming-only and opt-in. Clients request them with
`metadata={"langgraph_stream_events": "v1"}` and receive a versioned
`langgraph_openai_serve` property on an otherwise standard Chat Completions
chunk. Missing and unsupported versions produce the ordinary strict stream.
Unknown custom events remain available only to direct runner consumers.

See [Client stream events](explanation/openai-compatibility.md#client-stream-events)
for the wire contract and [OpenAI clients](tutorials/openai-clients.md#client-stream-events)
for consumption.

## Citation Events

Inside a graph node or tool, emit a citation with LangGraph's stream writer:

```python
from langgraph.config import get_stream_writer
from langgraph_openai_serve import citation_event

get_stream_writer()(
    citation_event(
        url="https://example.com/source",
        title="Example source",
        span=(10, 14),
    )
)
```

`span` uses Python's half-open convention, so `text[10:14]` returns the cited
text. LGOS converts it to OpenAI's inclusive `end_index` at the event boundary.
Use `citation_slice(annotation, text)` to validate received indices and convert
them back to a Python slice. Citation events must refer to the final rendered
assistant text.

See [Citation ownership](explanation/openai-compatibility.md#citation-ownership)
for transport and client behavior.

The graph runner preserves LangGraph's native `CustomStreamPart` values,
including their execution namespace. Other event types remain available to
direct runner consumers through `langgraph_openai_serve.graph.runner`.

::: langgraph_openai_serve
