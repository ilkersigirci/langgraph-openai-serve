# Reference

## OpenAI-Compatible API

Default prefix: `/v1`. Change it with `LGOS_OPENAI_API_PREFIX` or
`bind_openai_api(prefix=...)`. Generic access logs are emitted by the
deployment's ASGI server or ingress proxy.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | List registered graph models with LGOS descriptions. |
| `GET` | `/v1/models/{model}` | Retrieve one model with the required LGOS metadata extension. |
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
| `LGOS_ENABLE_LANGFUSE` | `false` | Lazily adds the package Langfuse callback to every graph run. |

Settings prefixed with `DEMO_` belong to the independent example applications
and are documented under [Demo Settings and Commands](demo/reference.md).

## Public API

Use `LanggraphOpenaiServe` to bind OpenAI-compatible routes to a FastAPI app.
After binding, `server.openai_app` exposes the mounted FastAPI application for
host integrations such as manual middleware or telemetry instrumentation.
Use `GraphRegistry` to map OpenAI `model` names to `GraphConfig` values.
The registry must contain at least one graph. Pydantic rejects empty registries
and model IDs that cannot be addressed as one URL path segment. Registry keys
are read-only after validation; use `registry.register(model_id, config)` to add
or replace a graph.

`LanggraphOpenaiServe(..., checkpoint_scope=resolver)` accepts an optional sync
or async callable from FastAPI `Request` to a non-empty, server-trusted string.
Interrupt checkpoint keys include this scope before model and run identity. Use
an authenticated tenant or principal identifier when caller-chosen run UUIDs
must be isolated between security domains; do not derive the scope from
untrusted Chat Completions metadata or the Chat Completions `user` field. The
default `"default"` scope is suitable only for a single-tenant or shared-trust
deployment. The resolver must return the same scope for the initial request and
its resume; changing tenant identity makes the other scope's checkpoint
deliberately unreachable.

`GraphConfig` accepts:

- `graph`: compiled graph, sync factory, or async factory.
- `description`: required human-readable model description advertised by model
  listing and retrieval.
- `streamable_node_names`: node names whose streamed `AIMessageChunk` values are
  forwarded to clients.
- `features`: `GraphFeature` values that enable optional server behavior.
- `client_settings`: explicit public `ClientSettings` model class advertised by
  model retrieval.
- `runtime_callbacks`: callbacks included in the LangGraph `RunnableConfig`.
  When Langfuse tracing is enabled, LGOS adds its callback without mutating this
  collection or manager.
- `run_coordinator`: asynchronous single-flight coordination for interrupt
  runs. It receives LGOS's internal run key, rejects an occupied key instead of
  queueing it, and returns an async context manager.
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
| Interrupt run | server scope + model + optional `metadata.langgraph_run_id` UUID → internal checkpoint key | Isolate, retry, interrupt, and resume one operation. |

LGOS assembles runnable config from `runtime_callbacks` and, for an
interrupt-enabled run, a fixed-length SHA-256 checkpoint key derived from the
server-trusted scope, registered model, and operation UUID. This is deliberately
not a UI chat or thread ID. There is intentionally no adapter for placing
arbitrary OpenAI request fields into `config["configurable"]`; use typed runtime
context for values consumed by nodes.

Langfuse is a first-class optional integration. Install it and enable the
default callback through process environment settings:

```bash
uv add "langgraph-openai-serve[tracing]"
export LGOS_ENABLE_LANGFUSE=true
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

`LANGFUSE_BASE_URL` is optional; Langfuse Cloud is the default. Set it only for
a different cloud region or a self-hosted instance. Langfuse's
`CallbackHandler` owns its standard SDK configuration and error behavior. LGOS
constructs it on the first graph run that needs runnable configuration, then
reuses that process-wide handler. When enabled, the deployment-level toggle is
authoritative: LGOS adds Langfuse alongside empty, list, or manager callbacks
without altering the registered `GraphConfig` or caller-owned collection. To
provide a custom Langfuse handler, leave the toggle off and pass that handler
through `runtime_callbacks`.

For explicit construction, import
`langgraph_openai_serve.integrations.langfuse.get_langfuse_callback` or pass an
application-created vendor handler through `runtime_callbacks`.

When a callback is present, LGOS gives the graph run the stable name
`lgos.chat_completion` and adds `RunnableConfig.metadata` fields for the
request ID, registered graph model, (for interrupt runs) operation ID, and (when
the request supplies `metadata.session_id`) the Langfuse-recognized
`langfuse_session_id`. LangGraph also propagates primitive configurable values
during execution, so callbacks on interrupt runs receive the derived checkpoint
`thread_id`. LGOS does not set LangChain's native tracer `run_id` or force a
custom Langfuse trace ID. See [Production Logging and Request
Correlation](how-to-guides/production-logging.md#langfuse-correlation).

The same `features` set drives runtime behavior and the versioned
`langgraph_openai_serve.features` extension returned by
`GET /v1/models/{model}`. `GraphFeature.CLIENT_EVENTS` enables and advertises
public client-event chunks. `GraphFeature.INTERRUPTS` enables and advertises
the interrupt/resume flow.

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

Interrupt-enabled graphs have additional registration requirements:

- compile the graph with an asynchronous checkpointer that supports
  `aget_tuple()`, `alist()`, `aput()`, `aput_writes()`, and `adelete_thread()`;
- configure an asynchronous `run_coordinator`; and
- use a durable checkpointer and cross-process coordinator in production.

The initial request does not require metadata. LGOS generates a UUID operation
ID and returns it in every interrupt tool call. A caller that needs deterministic
initial-request retries can instead supply a non-nil UUID in
`metadata.langgraph_run_id`. `InMemoryRunCoordinator` is suitable only for
tests and a single-process development server; it cannot serialize requests
across workers or hosts.

Pending checkpoints exist only to resume an interrupt batch returned to the
client. LGOS deletes isolated checkpoint state after terminal completion or
when execution fails or is cancelled before producing that batch. Operators
must separately define an expiry policy for runs abandoned after a batch is
returned.

### PostgreSQL Coordination

Install `langgraph-openai-serve[postgres]` to use the public
`langgraph_openai_serve.integrations.postgres.PostgresRunCoordinator`. Use
LangGraph's official
[`AsyncPostgresSaver`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver)
for checkpoint storage; the LGOS adapter only supplies the cross-worker run
lease that the saver does not own. Run the saver's `setup()` once before API
workers start. A shared pool must follow the saver's upstream connection
requirements: `autocommit=True`, `prepare_threshold=0`, and mapping rows.

`PostgresRunCoordinator(pool, max_concurrent_leases=...)` accepts an existing
`psycopg_pool.AsyncConnectionPool` configured with mapping rows and the default
`close_returns=False`; physical session closure is the safety fallback for an
indeterminate lock operation. When the saver shares that pool, set the lease
limit below the pool maximum so at least one connection remains available for
checkpoint writes. Create one coordinator per process-owned pool so that this
capacity limit is not accidentally multiplied. Session advisory locks require
direct PostgreSQL connections or session-mode pooling; transaction-mode poolers
cannot preserve the lease. Lock contention itself fails immediately through
PostgreSQL's `pg_try_advisory_lock`; connection checkout still follows the
pool's configured timeout. The
[demo deployment](demo/docker.md#demo-services) uses one pool for both
components and a separate one-shot schema setup process.

## Client Stream Events

Declare the feature on every graph that publishes client events:

```python
from langgraph_openai_serve import GraphConfig, GraphFeature

config = GraphConfig(
    graph=graph,
    description="Graph that reports media-generation status.",
    features={GraphFeature.CLIENT_EVENTS},
)
```

Inside a long-running graph node or tool, publish user-facing status with
`status_event()`:

```python
from langgraph.config import get_stream_writer
from langgraph_openai_serve import status_event

writer = get_stream_writer()
writer(status_event("Generating audio", namespace=("media",)))

# Perform the long-running work.

writer(
    status_event(
        "Audio ready",
        done=True,
        namespace=("media",),
    )
)
```

The portable status data matches native UI status concepts:

```json
{
  "type": "status",
  "namespace": ["media"],
  "data": {
    "description": "Generating audio",
    "done": false,
    "hidden": false
  }
}
```

`done=False` displays ongoing work; always finish a visible status sequence with
`done=True`. Set `hidden=True` on the final update when clients should remove the
status after completion. Status text is deliberately authored by the graph:
LGOS does not infer it from internal node names or state.

For other passive notifications, use `client_event()`:

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

Events are streaming-only and require both the graph feature and client opt-in.
Clients request them with
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
