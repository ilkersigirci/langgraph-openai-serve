# Reference

## OpenAI-Compatible API

Default prefix: `/v1`. Change it with `LGOS_OPENAI_API_PREFIX` or
`bind_openai_api(prefix=...)`. Generic access logs are emitted by the
deployment's ASGI server or ingress proxy.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | List registered graph models with LGOS descriptions and features. |
| `GET` | `/v1/models/{model}` | Retrieve one model with the required LGOS metadata extension. |
| `POST` | `/v1/responses` | Run a graph now or accept an opted-in polling-only background Response. |
| `GET` | `/v1/responses/{response_id}` | Retrieve an authorized, unexpired background Response snapshot. |
| `POST` | `/v1/responses/{response_id}/cancel` | Atomically request cancellation or return the terminal result that already won. |
| `POST` | `/v1/chat/completions` | Run a graph through OpenAI chat completions. |
| `GET` | `/v1/health` | Health check. |

FastAPI docs for the mounted OpenAI app are disabled by default. Set
`LGOS_OPENAI_API_DOCS_ENABLED=true` to expose `{prefix}/docs`, `{prefix}/redoc`,
and `{prefix}/openapi.json`.

### Responses Request

The route accepts string or ordered message input, instructions, plain
`input_text`, `input_file.file_id`, string-valued metadata, `user`, flat client
function tools, registered name-only custom-tool selectors, standard
`web_search`, their choices,
`parallel_tool_calls`, plain text output, and streaming. Replayed
assistant output messages preserve `phase`; complete
`function_call` items and matching string-valued `function_call_output` items
support ordinary client-tool continuation. Interrupt continuation sends only
matching `function_call_output` items with `previous_response_id`.
Server execution returns native `custom_tool_call` and
`custom_tool_call_output` pairs or a
`web_search_call` in the same response; complete output items can also be
replayed as history.
LGOS executes registered custom tools inside that response. This intentionally
differs from OpenAI's ordinary custom-tool flow, where caller code executes the
tool and supplies its output to a later model request; the wire items remain
standard Responses types.
The public `web_search` shape does not prescribe the graph's search backend;
the bundled demo chooses an HTTP or upstream provider backend.

Foreground LGOS Responses are not persisted for retrieval or deletion. Omitted,
null, and false `store` values are accepted, and the returned foreground
Response reports `store=false`; foreground `store=true` is rejected.
For a model with `GraphConfig.background`, `background=true` selects the
polling-only durable path and permits either `store=false` or `store=true`.
The latter selects the configured longer bounded result retention; it is not a
conversation store. Background streaming and cursor/event replay are rejected.
`conversation` is unsupported in both modes.
`previous_response_id` is supported for interruptible graphs to resume execution
(and rejected for non-interruptible and background graphs); new `instructions`
are rejected on those resumes. The route also rejects unregistered custom tools, client-supplied
custom descriptions or formats, other built-in tools,
structured output, image/audio input, URL or inline file input, function
result-content lists, reasoning and generation controls, `include`, stream options,
service tiers, reusable prompts,
prompt-cache controls, and truncation. Unknown fields are not silently ignored.
See the [supported Responses subset](explanation/openai-compatibility.md#supported-responses-subset)
for the complete behavior and continuation rules.

### Chat Completions Request

Chat messages accept string content, explicit `text` content parts, and native
`file` parts containing only `file.file_id`. The route supports modern function
`tools`, `tool_choice`, assistant `tool_calls`, matching `tool` messages,
streaming, and `stream_options.include_usage`. Image and audio parts, inline
file data or filenames, prompt-cache fields, deprecated function fields,
generation controls, and other unknown fields are rejected.

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
The registry copies its initial mapping and must contain at least one graph. It
rejects empty model IDs, `.`, `..`, and IDs containing `/`. The public
`registry.registry` mapping is an insertion-ordered, read-only view; use
`registry.register(model_id, config)` to add or replace a graph. Replacing an
existing ID preserves its position.

`LanggraphOpenaiServe(..., checkpoint_scope=resolver, background=execution)`
accepts an optional sync
or async callable from FastAPI `Request` to a non-empty, server-trusted string.
Interrupt checkpoint keys and background Response authorization include this
scope before model and run identity. Use
an authenticated tenant or principal identifier when caller-chosen run UUIDs
must be isolated between security domains; do not derive the scope from
untrusted OpenAI metadata or the OpenAI `user` field. The
default `"default"` scope is suitable only for a single-tenant or shared-trust
deployment. The resolver must return the same scope for the initial request and
its resume; changing tenant identity makes the other scope's checkpoint
deliberately unreachable.

Responses `input_file.file_id` content and native Chat file parts normalize to
the same LangChain file block, so graphs receive native `file_id` values and
decide whether to download, parse, or forward them. File upload and storage
belong to an external OpenAI Files API, not the LGOS package. See
[Accept And Display Files](how-to-guides/file-inputs.md).

`GraphConfig` accepts:

- `graph`: compiled graph, sync factory, or async factory.
- `description`: required human-readable model description advertised by model
  listing and retrieval.
- `features`: `GraphFeature` values that enable optional server behavior or
  advertise graph input and client-tool capabilities.
- `client_settings`: explicit public `ClientSettings` model class advertised by
  model retrieval.
- `server_tools`: internal allowlist of server-executed tool names. A
  registered custom tool is selected with the Responses `custom` type and name;
  `web_search` uses its built-in type. The graph owns tool definitions and
  execution. Model retrieval does not advertise tools. See
  [Server Tools](explanation/openai-compatibility.md#server-tools).
- `runtime_callbacks`: callbacks included in the LangGraph `RunnableConfig`.
  When Langfuse tracing is enabled, LGOS adds its callback without mutating this
  collection or manager.
- `run_coordinator`: asynchronous single-flight coordination for interrupt
  or background runs. It rejects an occupied LGOS checkpoint key instead of
  queueing it and returns an async context manager.
- `background`: optional immutable `BackgroundPolicy` that opts the model into
  polling-only execution and records its graph version. Native retries and
  timeouts are configured on the Hatchet backend.
- `request_to_input(request, messages)`: custom normalized request and LangChain
  messages to graph input.
- `context_factory(request, client_settings)`: compose the final typed LangGraph
  runtime context from normalized request values, server-owned values, and optional
  validated public settings.
- `output_to_message(output)`: custom graph output to a durable `AIMessage`.

`GraphConfig` is immutable after construction. Pydantic snapshots `features`
and `server_tools` as frozen sets, so later mutations of the input collections
cannot change a registered model. To change a declaration, construct a
replacement and pass it to `registry.register()`. Freezing the declaration does
not make a caller-owned callback handler or callback manager internally
immutable.

Streaming forwards non-empty text from every `AIMessageChunk` emitted by the
graph's `messages` stream. Configure private `ChatOpenAI` calls with
[`disable_streaming=True`](https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/disable_streaming);
LangChain then uses the complete invocation path and does not emit model stream
chunks for that call.

A directly supplied compiled graph is reused. A sync or async graph factory is
called for every request and is never cached; LGOS validates each resolved value
as a compiled state graph and rechecks its context schema and interrupt
checkpointer capabilities before execution. Static configuration relationships,
including the requirement that `run_coordinator` appear exactly when
`GraphFeature.INTERRUPTS` or a background policy is enabled, fail during
`GraphConfig` construction. Interrupt and background execution are mutually
exclusive.

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

Graph adapters receive an immutable, protocol-neutral `GraphRequest` from either
API's decoder. It exposes
the shared `model`, `metadata`, `user`, normalized client function `tools`,
selected server-tool names in `server_tools`, `tool_choice`, and `parallel_tool_calls`
values. `NamedFunctionToolChoice` identifies a required client function, while
`NamedCustomToolChoice` identifies a required registered custom tool. A
single `web_search` declaration with `tool_choice="required"` requires search.
Named built-in choices are outside the supported subset.
Raw OpenAI transport models are
not part of the graph-adapter interface.

Runtime context is separate from `RunnableConfig`:

| Value | LGOS/LangGraph path | Intended use |
| --- | --- | --- |
| Graph input | `graph.ainvoke(input, ...)` or `graph.astream(input, ...)` | Messages and mutable workflow state. |
| Runtime context | public settings → optional `context_factory` → `context=` → `Runtime.context` | Immutable per-run application values and dependencies. |
| Runnable config | `config=` | Callbacks, tags, tracing, and other execution controls. |
| Interrupt run | server scope + model + optional `metadata.lgos_run_id` UUID → internal checkpoint key | Isolate, retry, interrupt, and resume one operation. |

LGOS assembles runnable config from `runtime_callbacks` and, for an
interrupt-enabled run, a fixed-length SHA-256 checkpoint key derived from the
server-trusted scope, registered model, and operation UUID. This is deliberately
not a UI chat or thread ID. There is intentionally no adapter for placing
arbitrary OpenAI request fields into `config["configurable"]`; use typed runtime
context for values consumed by nodes.

### Langfuse Tracing

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
`lgos.graph_run` for both endpoints and adds `RunnableConfig.metadata` fields for the
request ID, registered graph model, (for interrupt runs) operation ID, and (when
the request supplies `metadata.conversation_id`) the Langfuse-recognized
`langfuse_session_id`. LangGraph also propagates primitive configurable values
during execution, so callbacks on interrupt runs receive the derived checkpoint
`thread_id`. LGOS does not set LangChain's native tracer `run_id` or force a
custom Langfuse trace ID. See [Production Logging and Request
Correlation](how-to-guides/production-logging.md#langfuse-correlation).

The `features` set is returned in the versioned `lgos.features` extension and
enables server behavior where applicable. `GraphFeature.CLIENT_EVENTS` enables
and advertises public
status commentary in streaming Responses. Chat Completions ignores custom
stream events and does not emit commentary.
`GraphFeature.MCP_TOOLS` advertises that a client may attach and execute tools
from its configured MCP gateway; it does not publish tool definitions or grant
access to them.
`GraphFeature.FILE_INPUTS` advertises that the graph
resolves native file content parts. `GraphFeature.INTERRUPTS` enables and
advertises the interrupt/resume flow. `GraphFeature.BACKGROUND` is derived for
model discovery from `GraphConfig.background`; declaring it directly is an
error.

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
`metadata.lgos_settings` string. Clients omit values equal to the advertised
defaults. System instructions remain ordinary OpenAI messages and are
independent of `ClientSettings`; native OpenAI fields keep their standard
request semantics.

LGOS validates defaults and generates the discovery JSON Schema when the graph
is registered, then validates settings on every request. Without
`context_factory`, the settings become `Runtime.context`. A factory can instead
combine them with server-derived identity, authorization, database clients, and
other dependencies.

The serialized descriptor appears only on model retrieval as
`lgos.client_settings`, with independent `schema_version`,
`json_schema`, and `defaults` fields. All client settings use the fixed
`metadata.lgos_settings` key. Clients use the descriptor's
validated `defaults` object as the baseline; `default` keywords within the
generated JSON Schema are annotations, not the runtime baseline. The schema's
`$schema` keyword declares the JSON Schema 2020-12 dialect independently of the
LGOS descriptor version.

See [Configure LangGraph Runtime Settings](how-to-guides/langgraph-runtime-settings.md)
for the runtime settings flow, and
[Runtime Settings](explanation/openai-compatibility.md#runtime-settings) for the
request lifecycle.

Interrupt-enabled graphs have additional registration requirements:

- compile the graph with an asynchronous checkpointer that supports
  `aget_tuple()`, `aput()`, `aput_writes()`, and `adelete_thread()`;
- configure an asynchronous `run_coordinator`; and
- use a durable checkpointer and cross-process coordinator in production.

The initial request does not require metadata. LGOS generates a UUID operation
ID and embeds it in the paused Response ID. A caller that needs deterministic
initial-request retries can instead supply a non-nil UUID in
`metadata.lgos_run_id`. `InMemoryRunCoordinator` is suitable only for
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
for checkpoints and
[`AsyncPostgresStore`](https://reference.langchain.com/python/langgraph.store.postgres/aio/AsyncPostgresStore)
for application data. The LGOS adapter supplies only the cross-worker
interrupt/background run lease; it does not replace either storage primitive.
Background Response rows use the separate `PostgresResponseStore` described
below. Run each
configured storage adapter's `setup()` once before API workers start. A shared
pool must follow the upstream connection requirements: `autocommit=True`,
`prepare_threshold=0`, and mapping rows.

`PostgresRunCoordinator(pool, max_concurrent_leases=...)` accepts an existing
`psycopg_pool.AsyncConnectionPool` configured with mapping rows and the default
`close_returns=False`; physical session closure is the safety fallback for an
indeterminate lock operation. When persistence adapters share that pool, set
the lease limit below the pool maximum so at least one connection remains
available for persistence I/O. Create one coordinator per process-owned pool
so that this capacity limit is not accidentally multiplied. Session advisory
locks require direct PostgreSQL connections or session-mode pooling;
transaction-mode poolers cannot preserve the lease. Lock contention itself
fails immediately through PostgreSQL's `pg_try_advisory_lock`; connection
checkout still follows the pool's configured timeout. The
[demo deployment](demo/docker.md#demo-services) uses one pool for both
storage adapters and interrupt coordination, plus a separate one-shot schema
setup process. Busy interrupt leases fail before streaming begins with HTTP 409
and `code: "run_busy"`.

## Background Execution

The package exports the lifecycle-level `BackgroundBackend`, `NewRun`,
`StoredRun`, `ResponseStore`, `BackgroundPolicy`, `BackgroundSettings`,
`RunJob`, and `BackgroundWorker` public interfaces. It also exports
`InMemoryBackgroundBackend` and `InMemoryResponseStore` for single-process
development. Hatchet is the only durable built-in backend.

`BackgroundSettings` defaults are:

| Field | Default |
| --- | --- |
| `admission_capacity` | `10_000` active Responses |
| `result_retention` | 1 hour |
| `stored_result_retention` | 30 days |
| `idempotency_retention` | 24 hours |
| `maintenance_batch_size` | 100 rows |

`HatchetAdapterSettings` keeps native orchestration policy in Hatchet:

| Field | Default |
| --- | --- |
| `workflow_name` | `lgos-background-response` |
| `maintenance_task_name` | `lgos-background-maintenance` |
| `retries` | 3 |
| `backoff_factor` / `backoff_max_seconds` | 2.0 / 30 |
| `schedule_timeout` | 30 minutes |
| `execution_timeout` | 20 minutes |
| `finalization_retries` / `finalization_timeout` | 3 / 5 minutes |
| `idempotency_ttl` | 24 hours |
| `maintenance_cron` / `maintenance_timeout` | every 5 minutes / 5 minutes |

`BackgroundWorker` loads the persisted envelope, obtains its coordinator lease,
and executes through the shared graph runner. `execute()` raises
`RetryableJobError` for the engine to retry. Call `finalize()` after native
retries are exhausted; it never advances an unfinished graph. Schedule
`maintain()` for checkpoint cleanup and Response expiry. The Hatchet adapter
also retries pending native cancellations on that schedule.

Install `langgraph-openai-serve[postgres]` for
`integrations.background_postgres.PostgresResponseStore`. Its `setup()` method
creates the packaged final schema explicitly. Install
`langgraph-openai-serve[hatchet]` for
`HatchetBackgroundBackend`, `HatchetAdapterSettings`,
`create_hatchet_workflows()`, and `check_hatchet_connection()`. The registered
workflow gives Hatchet native ownership of idempotency, retries, backoff,
schedule and execution timeouts, cancellation, final-failure handling, and the
maintenance schedule. Finalization and maintenance use the configured
`schedule_timeout` rather than Hatchet's shorter SDK default. The adapter is
optional and is never imported by the core package.

See [Run Responses In The Background](how-to-guides/background-responses.md)
for graph requirements, lifecycle semantics, deployment wiring, custom adapter
obligations, and operator recovery.

## Streaming Status

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

The helper writes this versioned graph-to-LGOS envelope:

```json
{
  "type": "lgos.client_event",
  "schema_version": 1,
  "event": {
    "type": "status",
    "namespace": ["media"],
    "data": {
      "description": "Generating audio",
      "done": false,
      "hidden": false
    }
  }
}
```

Status text is deliberately authored by the graph; LGOS does not infer it from
internal node names or state. Responses exposes the description as commentary
and suppresses hidden updates; the namespace, `done`, and `hidden` fields do not
become nonstandard Response fields.

The event envelope has its own schema version, independent of model discovery
and client settings. The v1 event vocabulary is `status`, `progress`, and
`artifact`.
`client_event("status", data)` remains the lower-level equivalent when an
application already has validated status data; prefer `status_event()` for its
typed fields. Event data must be JSON-safe, and every namespace segment must be a
string. The namespace is a stable, author-defined path; LGOS does not expose
LangGraph's dynamic execution namespace.

Status is streaming-only and always requires the graph feature. Responses needs
no metadata opt-in and emits each visible update as a standard
`phase="commentary"` message. The Chat Completions API is strictly for simple
graphs and plain text streaming; it ignores custom stream events and does not emit
commentary. Responses ignores `progress` and `artifact`. Use standard Responses
function calls plus the Files API for portable durable rich output. Unknown custom
events remain available only to direct runner consumers.

See [Streaming status](explanation/openai-compatibility.md#streaming-status) for
the wire contract and
[Stream final text and commentary](tutorials/openai-clients.md#stream-final-text-and-commentary)
for consumption.

## Citations

Put citations on the final LangChain `AIMessage`:

```python
from langchain_core.messages import AIMessage
from langchain_core.messages.content import create_citation, create_text_block

message = AIMessage(
    content=[
        create_text_block(
            text="Read the source [1].",
            annotations=[
                create_citation(
                    url="https://example.com/source",
                    title="Example source",
                    start_index=9,
                    end_index=14,
                    cited_text="source",
                )
            ],
        )
    ]
)
```

Put visible inline citations in the assistant text. Structured annotations add
machine-readable provenance; clients are not required to invent marker text
from annotation indices.

LangChain citation indices refer to their containing text block. LGOS offsets
them into the final response text and preserves OpenAI's inclusive `end_index`.
Use `citation_slice(start_index, end_index, text)` to validate them and create a
Python slice. Responses maps citations to `output_text.annotations` and emits
the typed annotation event while streaming. Chat maps them to completed
`message.annotations`; its final streaming delta uses the compatibility
extension.

See [Citation ownership](explanation/openai-compatibility.md#citation-ownership)
for transport and client behavior.

The streaming graph runner preserves LangGraph's native `CustomStreamPart`
values, including their execution namespace. Non-streaming invocation does not
subscribe to or replay custom events.

::: langgraph_openai_serve
