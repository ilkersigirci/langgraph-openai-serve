# Production Logging and Request Correlation

LGOS emits standard-library `logging.LogRecord` objects. The host application
chooses how those records are formatted, collected, retained, and exported.

## LGOS responsibilities

For HTTP requests handled by the mounted OpenAI-compatible application, LGOS:

- accepts `X-Request-ID`, or generates a UUID4 when the header is missing or
  unusable;
- returns the request ID in the `X-Request-ID` response header, including on
  validation, error, and streaming responses;
- adds `request_id`, `model`, `stream`, and, for interrupt operations, the LGOS
  `operation_id` through request-scoped context; and
- logs unexpected request failures and application-level server errors with
  the request context.

LGOS does not emit one generic log record for every successful or client-error
request. The ASGI server or ingress proxy owns access logs, including method,
path, status, and request duration. This prevents duplicate access records and
keeps latency measurement in the server or observability layer that owns it.

LGOS does not deliberately attach prompts, messages, responses, tool data,
credentials, full headers, client metadata, or LangGraph state as structured
log fields. That is not a data-safety guarantee: accepted `X-Request-ID` values
are caller-controlled, and exception messages or tracebacks from graphs and
dependencies can contain arbitrary application data. Treat exported records as
potentially sensitive. `request_id` identifies one HTTP invocation; it is not a
LangGraph `run_id`, a session ID, or a tracing ID.

## Deployment responsibilities

The application or deployment configures:

- text or JSON formatting;
- stdout/stderr routing and collection;
- trusted request-ID policy, retention, redaction, and access controls;
- ASGI server or ingress access logs; and
- OpenTelemetry instrumentation, metrics, traces, and OTLP export. The demo's
  optional [Demo OpenTelemetry](../demo/opentelemetry.md) overlay shows one
  Collector-based deployment pattern.

For latency percentiles and distributed request timing, use the deployment's
metrics and tracing system rather than adding a per-request application log.

LGOS does not configure the root logger, install output handlers, select a
formatter, write log files, or create OpenTelemetry trace and span IDs.

If browser code needs to read the returned request ID, expose
`X-Request-ID` in the host application's CORS configuration. Restrict accepted
upstream IDs to a trusted proxy or redact them before export when callers must
not control retained telemetry.

## Application formatting

Configure logging in the host application. The runnable demo's
`demo/api/src/lgos_demo_api/logging.py` shows how to format LGOS and Uvicorn
server records together without changing the LGOS package. The demo disables
Uvicorn access logs; enable access logging at the ASGI server or ingress layer
that owns request timing and retention.

The formatter can include LGOS context fields such as `request_id`, `model`,
`stream`, and `operation_id`. If OpenTelemetry's Python logging
instrumentation is enabled, its `otelTraceID` and `otelSpanID` fields can
coexist with these application fields.

Set severity floors on each output handler, not only on the root logger. A
propagated record is offered directly to ancestor handlers, so an ancestor
logger's level does not filter a child logger that explicitly emits a lower
level.

## Langfuse correlation

When `LGOS_ENABLE_LANGFUSE=true`, LGOS passes the following correlation
metadata through LangChain's `RunnableConfig`:

- `lgos.request_id`, when an HTTP request context exists;
- `lgos.operation_id`, for an interrupt operation;
- `lgos.model`, the registered OpenAI-compatible graph model;
- `langfuse_session_id`, when the request contains a non-empty
  `metadata.session_id` string.

LGOS also uses the stable run name `lgos.chat_completion`. LangGraph propagates
primitive `configurable` fields to callback metadata during execution, so an
interrupt-enabled run also exposes the derived checkpoint `thread_id` to
callbacks. That checkpoint identifier is operation state, not conversation
identity. LGOS does not set LangChain's native execution `run_id` or Langfuse's
generated trace ID.

Langfuse's LangChain integration creates a trace for each invocation by
default. Treat one Chat Completions request as one trace. When several requests
belong to one conversation, send the same UI-owned value as
`metadata.session_id`; LGOS maps it to Langfuse's `langfuse_session_id`, which
groups the independent traces into one
[session](https://langfuse.com/docs/observability/features/sessions). This is
correlation only: LGOS remains stateless and the client must still send the
conversation messages needed by each completion. The
[Langfuse trace ID guidance](https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing)
supports deterministic custom trace IDs for a trusted external ID, but LGOS's
`X-Request-ID` is a correlation header and may be client-supplied or reused.
It is therefore metadata here, not a custom trace ID.

For end-to-end distributed tracing, configure the host application and proxy
to propagate W3C/OpenTelemetry context. LGOS does not install a second tracing
system or create a parent span around the HTTP request.
