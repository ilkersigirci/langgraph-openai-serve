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
| `GET` | `/v1/responses/{response_id}` | Retrieve an authorized background Response. |
| `POST` | `/v1/responses/{response_id}/cancel` | Cancel an active background Response or return a finished one. |
| `POST` | `/v1/chat/completions` | Run a graph through OpenAI chat completions. |
| `GET` | `/v1/health` | Health check. |

FastAPI docs for the mounted OpenAI app are disabled by default. Set
`LGOS_OPENAI_API_DOCS_ENABLED=True` to expose `{prefix}/docs`, `{prefix}/redoc`,
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
For a model declaring `GraphFeature.BACKGROUND`, `background=true` selects the
polling-only path and permits either `store=false` or `store=true`; the
background engine keeps the result either way, and Hatchet keeps it for its
[data retention](https://docs.hatchet.run/self-hosting/data-retention) period
(30 days by default when self-hosted). Background streaming and
cursor/event replay are rejected. `conversation` is unsupported in both modes.
`previous_response_id` is supported for interruptible graphs to resume execution
(and rejected for non-interruptible graphs); new `instructions` are rejected on
those resumes. The route also rejects unregistered custom tools, client-supplied
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
| `LGOS_OPENAI_API_DOCS_ENABLED` | `False` | Enables docs only for the mounted OpenAI app. |
| `LGOS_ENABLE_LANGFUSE` | `False` | Lazily adds the package Langfuse callback to every graph run. |

Settings prefixed with `DEMO_` belong to the independent example applications
and are documented under [Demo Settings and Commands](demo/reference.md).

## Public API

Use `LanggraphOpenaiServe` to bind OpenAI-compatible routes to a FastAPI app.
After binding, `server.openai_app` exposes the mounted FastAPI application for
host integrations such as manual middleware or telemetry instrumentation.
Use `GraphRegistry(graphs={...}, run_coordinator=...)` to map OpenAI `model`
names to `GraphConfig` values; `registry.graphs` is a plain dict. The registry
must contain at least one graph. Model IDs must be non-empty, must not contain
`/`, and must not be `.` or `..`. `registry.get_graph(model)` returns a
`GraphConfig` or raises a 404 `model_not_found` error. `run_coordinator` is
shared by every interrupt-enabled graph and is required only when a graph
declares `GraphFeature.INTERRUPTS`.

`LanggraphOpenaiServe(registry=registry, app=app)` serves the registry.
`LanggraphOpenaiServe(..., background=backend)` accepts an optional
`BackgroundBackend`. `LanggraphOpenaiServe(..., checkpoint_scope=resolver)`
accepts an optional sync or async callable from FastAPI `Request` to a
non-empty, server-trusted string.
Interrupt checkpoint keys and background Response authorization include this
scope before model and run identity. Use
an authenticated tenant or principal identifier when runs must be isolated
between security domains; do not derive the scope from
untrusted OpenAI metadata or the OpenAI `user` field. The
default `"default"` scope is suitable only for a single-tenant or shared-trust
deployment. The resolver must return the same scope for the initial request and
its resume; changing tenant identity makes the other scope's checkpoint
deliberately unreachable.

`RequestContextFilter` adds the active request's LGOS fields, such as
`request_id`, to log records; install it on a host handler to enrich every
record. See [Production Logging](how-to-guides/production-logging.md#application-formatting).

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
- `request_to_input(request, messages)`: custom normalized request and LangChain
  messages to graph input.
- `context_factory(request, client_settings)`: compose the final typed LangGraph
  runtime context from normalized request values, server-owned values, and optional
  validated public settings.
- `output_to_message(output)`: custom graph output to a durable `AIMessage`.

`GraphConfig` is immutable after construction. Pydantic snapshots `features`
and `server_tools` as frozen sets, so later mutations of the input collections
cannot change a registered model. Freezing the declaration does
not make a caller-owned callback handler or callback manager internally
immutable.

Streaming forwards non-empty text from every `AIMessageChunk` emitted by the
graph's `messages` stream. To keep a private model call out of the stream, tag
it with LangGraph's
[`nostream`](https://docs.langchain.com/oss/python/langgraph/streaming#omit-messages-from-the-stream)
tag, for example `ChatOpenAI(..., tags=["nostream"])`; the call still runs and
returns its output, but LangGraph emits none of its tokens.
[`disable_streaming=True`](https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/disable_streaming)
is for models that cannot stream: it makes the provider request non-streaming,
and LangGraph still emits the call's complete message, which LGOS does not
forward but other consumers of the graph's stream would see.

A directly supplied compiled graph is reused and validated when its
`GraphConfig` is built. A sync or async graph factory is called for every request
and is never cached; LGOS validates each resolved value as a compiled state graph
and rechecks it before execution. Validation checks the context schema and
interrupt checkpointer capabilities, rejects static breakpoints
(`interrupt_before` or `interrupt_after`), since LGOS pauses only at
`interrupt()`, and rejects a checkpointer on a graph without
`GraphFeature.INTERRUPTS`, since clients send the full conversation with every
other request. A registry with an
interrupt-enabled graph and no `run_coordinator` fails during `GraphRegistry`
construction. A graph may declare both interrupts and background; a background
run that reaches an interrupt completes with `lgos_interrupt` function calls,
and an answer in either mode continues it.

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
| Graph input | `graph.astream(input, ...)` | Messages and mutable workflow state. |
| Runtime context | public settings → optional `context_factory` → `context=` → `Runtime.context` | Immutable per-run application values and dependencies. |
| Runnable config | `config=` | Callbacks, tags, tracing, and other execution controls. |
| Interrupt run | server scope + model + server-generated run UUID → internal checkpoint key | Isolate, interrupt, and resume one operation. |

LGOS assembles runnable config from `runtime_callbacks` and, for an
interrupt-enabled run, a fixed-length SHA-256 checkpoint key derived from the
server-trusted scope, registered model, and run UUID. This is deliberately
not a UI chat or thread ID. There is intentionally no adapter for placing
arbitrary OpenAI request fields into `config["configurable"]`; use typed runtime
context for values consumed by nodes.

### Langfuse Tracing

Langfuse is a first-class optional integration. Install it and enable the
default callback through process environment settings:

```bash
uv add "langgraph-openai-serve[tracing]"
export LGOS_ENABLE_LANGFUSE=True
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

`LANGFUSE_BASE_URL` is optional; Langfuse Cloud is the default. Set it only for
a different cloud region or a self-hosted instance. Langfuse's `CallbackHandler`
owns its standard SDK configuration and error behavior. LGOS constructs it on
the first graph run that needs runnable configuration, then reuses that
process-wide handler. Before that, LGOS creates the Langfuse client with
`should_export_span` from `langgraph_openai_serve.integrations.langfuse`:
Langfuse's default export filter without LGOS's [OpenTelemetry](#opentelemetry)
spans, so each trace keeps the callback's root observation with the run's input
and output. An application that creates its own Langfuse client first keeps its
filter; pass the same `should_export_span` to keep those trace roots. When
enabled, the deployment-level toggle is authoritative: LGOS adds Langfuse
alongside empty, list, or manager callbacks without altering the registered
`GraphConfig` or caller-owned collection. To provide a custom Langfuse handler,
leave the toggle off and pass that handler through `runtime_callbacks`.

For explicit construction, import
`langgraph_openai_serve.integrations.langfuse.get_langfuse_callback` or pass an
application-created vendor handler through `runtime_callbacks`.

LGOS gives every graph run the stable name
`lgos.graph_run` for both endpoints and adds `RunnableConfig.metadata` fields for the
request ID, registered graph model, (for interrupt runs) operation ID, and (when
the request supplies `metadata.conversation_id`) the Langfuse-recognized
`langfuse_session_id`. LangGraph also propagates primitive configurable values
during execution, so callbacks on interrupt runs receive the derived checkpoint
`thread_id`. LGOS does not set LangChain's native tracer `run_id` or force a
custom Langfuse trace ID. See [Production Logging and Request
Correlation](how-to-guides/production-logging.md#langfuse-correlation).

The `features` set is returned in the `lgos.features` extension and
enables server behavior where applicable.
`GraphFeature.MCP_TOOLS` advertises that a client may attach and execute tools
from its configured MCP gateway; it does not publish tool definitions or grant
access to them.
`GraphFeature.FILE_INPUTS` advertises that the graph
resolves native file content parts. `GraphFeature.INTERRUPTS` enables and
advertises the interrupt/resume flow. `GraphFeature.BACKGROUND` enables and
advertises polling-only background Responses.

### OpenTelemetry

LGOS reports every graph execution through the OpenTelemetry API, following the
GenAI [workflow span](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md#invoke-workflow-span)
and [workflow metric](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md#metric-gen_aiinvoke_workflowduration)
conventions. The API records nothing until the host application configures an
SDK, for example with `opentelemetry-instrument`.

| Signal | Name | Attributes |
| --- | --- | --- |
| Span, kind `INTERNAL` | `invoke_workflow {model}` | `gen_ai.operation.name=invoke_workflow`, `gen_ai.workflow.name`, `gen_ai.conversation.id` when the request supplies `metadata.conversation_id`, and `error.type` on failure |
| Histogram, unit `s` | `gen_ai.invoke_workflow.duration` | `gen_ai.workflow.name`, and `error.type` on failure |

`gen_ai.workflow.name` is the registered model. Both signals cover graph
execution through the final output; they exclude lease waits, background queue
time, and checkpoint cleanup. The span is current only while LGOS advances the
graph: spans the graph creates, such as model calls, are its children, while
code that consumes a stream keeps its own parent span between events. A run
that ends before its output exists fails: one that raises an exception, one
cancelled by a disconnecting client or a cancelled background Response, and one
whose stream the consumer closes mid-run. The span status is then `ERROR` with
the exception message, if any, and `error.type` is the exception type, qualified
by its module unless it is built in, such as `asyncio.exceptions.CancelledError`
or `GeneratorExit`. Closing a stream after its final output is not a failure.
[Design Choices](explanation/design-choices.md) lists the tracers whose
cancellation handling this follows. A raised `Exception` propagates to its
handler: LGOS routes and the background engines log it with its traceback, and
direct Python callers receive it; cancellations and closed streams are not
logged. The span does not repeat an exception as an event. The histogram uses
the conventions' bucket boundaries, which start at one second; configure an SDK
View for finer boundaries.

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
deterministic default, because clients omit values equal to the advertised
defaults; a `default_factory` must return the same value on every call.
Registration fails when the model cannot build its defaults or schema.

All public fields travel together as compact JSON text in the
`metadata.lgos_settings` string. Clients omit values equal to the advertised
defaults. System instructions remain ordinary OpenAI messages and are
independent of `ClientSettings`; native OpenAI fields keep their standard
request semantics.

LGOS validates the settings with the model's strict JSON validation on every
request. Without
`context_factory`, the settings become `Runtime.context`. A factory can instead
combine them with server-derived identity, authorization, database clients, and
other dependencies.

The serialized descriptor appears only on model retrieval as
`lgos.client_settings`, with `json_schema` and `defaults` fields. All client
settings use the fixed `metadata.lgos_settings` key. Clients use the
descriptor's validated `defaults` object as the baseline; `default` keywords
within the generated JSON Schema are annotations, not the runtime baseline. The
schema's `$schema` keyword declares the JSON Schema 2020-12 dialect.

See [Configure LangGraph Runtime Settings](how-to-guides/langgraph-runtime-settings.md)
for the runtime settings flow, and
[Runtime Settings](explanation/openai-compatibility.md#runtime-settings) for the
request lifecycle.

Interrupt-enabled graphs have additional registration requirements:

- compile the graph with an asynchronous checkpointer that supports
  `aget_tuple()`, `aput()`, `aput_writes()`, and `adelete_thread()`;
- configure an asynchronous `GraphRegistry.run_coordinator`; and
- use a durable checkpointer and cross-process coordinator in production.

The coordinator provides single-flight leases for interrupt runs: it returns an
async context manager and rejects an occupied checkpoint key instead of
queueing it. Exiting that context manager must release the lease even when the
exit is cancelled.

The initial request does not require metadata. LGOS generates a UUID run ID for
every new run and embeds it in the paused Response ID. `InMemoryRunCoordinator`
is suitable only for tests and a single-process development server; it cannot
serialize requests across workers or hosts.

### Expire Paused Runs

Pending checkpoints exist only to resume an interrupt batch returned to the
client. LGOS deletes isolated checkpoint state after terminal completion or when
execution fails or is cancelled before producing that batch. A run abandoned
after its batch is returned stays until deleted: call
`delete_expired_interrupt_runs(checkpointer, run_coordinator, older_than=...)`
from `langgraph_openai_serve.graph.interrupt` on a schedule, as LangGraph Agent
Server's [checkpointer TTL](https://docs.langchain.com/langsmith/configure-ttl)
does. In production, prefer one scheduled job over a loop in every replica, for
example a Kubernetes CronJob with `concurrencyPolicy: Forbid` or your task
queue's scheduler. It deletes the runs whose latest pause is older than
`older_than` and returns their count. It reads every checkpoint through
`alist()`, leaves threads LGOS did not create alone, and skips runs whose lease
is held. Choose a TTL longer than the longest time a user may take to answer. To
write your own cleanup, select threads whose checkpoint metadata contains
`OPERATION_ID_METADATA_KEY` from the same module, then hold each run's lease,
confirm that its latest checkpoint in any namespace is still older than your
TTL, and delete it through the checkpointer.

### PostgreSQL Coordination

Install `langgraph-openai-serve[postgres]` to use the public
`langgraph_openai_serve.integrations.postgres.PostgresRunCoordinator`. Use
LangGraph's official
[`AsyncPostgresSaver`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver)
for checkpoints and
[`AsyncPostgresStore`](https://reference.langchain.com/python/langgraph.store.postgres/aio/AsyncPostgresStore)
for application data. The LGOS adapter supplies only the cross-worker
interrupt-run lease; it does not replace either storage primitive. Run each
configured storage adapter's `setup()` before serving requests, serializing
migration attempts when workers can start together. Their migrations use
`CREATE INDEX CONCURRENTLY`, which deadlocks with a worker blocked in
`pg_advisory_lock`; retry `pg_try_advisory_lock` instead. A shared
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
storage adapters and interrupt coordination. Each process applies pending
LangGraph migrations during startup under a separate schema advisory lock.
Busy interrupt leases fail before streaming begins with HTTP 409
and `code: "run_busy"`.

## Background Execution

`LanggraphOpenaiServe(background=...)` accepts any `BackgroundBackend`: an
engine that runs `BackgroundJob`s and stores their status and result.

| Method | Contract |
| --- | --- |
| `submit(job)` | Start the job, or return the run already holding `job.idempotency_key`. |
| `get(run_id)` | Return the run's job, status, and executed Response, or `None`. |
| `cancel(run_id)` | Stop the run; a finished run keeps its outcome. |

The engine's worker calls `execute_background_job(job, run_id, registry)`. It
runs the job through the foreground Responses path and returns the Response
JSON the engine stores. Run IDs must be UUIDs because the public Response ID
embeds the run ID, so reading a Response needs no lookup table.
`InMemoryBackgroundBackend(registry)` runs jobs as tasks of one process for
development and tests; enter its `lifespan` in the application's lifespan.

Install `langgraph-openai-serve[hatchet]` for
`langgraph_openai_serve.integrations.hatchet`. `create_hatchet_task(hatchet)`
registers the task in the API and worker processes, with a 30-minute
`schedule_timeout` and a one-hour `execution_timeout` by default.
`HatchetBackgroundBackend(task, hatchet.runs)` submits, reads, and cancels its
runs. The worker's Hatchet lifespan yields the `GraphRegistry`, including its
`run_coordinator`, that the task executes. The task has no retries, and its
24-hour Hatchet idempotency key is the scoped `Idempotency-Key` digest. The
adapter is never imported by the core package.

Background create accepts an optional `Idempotency-Key` header of 1 to 255
characters. A retry returns the original Response without starting a second
run; reuse with different content returns `422` with
`code="idempotency_key_reused"`. A background interrupt run chooses its run ID
at submission.

See [Run Responses In The Background](how-to-guides/background-responses.md)
for the client contract, graph requirements, and deployment wiring.

## Streaming Status

Inside a long-running graph node or tool, publish user-facing status with
`status_event()`:

```python
from langgraph.config import get_stream_writer
from langgraph_openai_serve import status_event

writer = get_stream_writer()
writer(status_event("Generating audio"))

# Perform the long-running work.

writer(status_event("Audio ready"))
```

The helper returns this custom stream value:

```json
{
  "type": "lgos.status",
  "description": "Generating audio"
}
```

Status text is deliberately authored by the graph; LGOS does not infer it from
internal node names or state. Graphs need no feature declaration to publish
status.

Status is streaming-only. Streaming Responses needs no metadata opt-in and emits
each non-empty description as a standard `phase="commentary"` message.
Non-streaming Responses and Chat Completions ignore statuses. Both APIs ignore
any other custom stream data. Use standard Responses function calls plus the
Files API for portable durable rich output.

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
