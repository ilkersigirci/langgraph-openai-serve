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
  operation ID in the `run_id` log field through request-scoped context; and
- logs unexpected request failures and application-level server errors with
  the request context.

LGOS does not emit one generic log record for every successful or client-error
request. The ASGI server or ingress proxy owns access logs, including method,
path, status, and request duration. This prevents duplicate access records and
keeps latency measurement in the server or observability layer that owns it.

LGOS does not log prompts, messages, responses, tool data, credentials, full
headers, client metadata, or LangGraph state by default. `request_id` identifies
one HTTP invocation; it is not a LangGraph `run_id`, a session ID, or a tracing
ID.

## Deployment responsibilities

The application or deployment configures:

- text or JSON formatting;
- stdout/stderr routing and collection;
- retention, redaction, and access controls;
- ASGI server or ingress access logs; and
- OpenTelemetry instrumentation, metrics, traces, and OTLP export.

For latency percentiles and distributed request timing, use the deployment's
metrics and tracing system rather than adding a per-request application log.

LGOS does not configure the root logger, install output handlers, select a
formatter, write log files, or create OpenTelemetry trace and span IDs.

## Application formatting

Configure logging in the host application. The runnable demo's [JSON logging
configuration](../../demo/api/src/lgos_demo_api/logging.py) shows how to format
LGOS and Uvicorn server records together without changing the LGOS package. The
demo disables Uvicorn access logs; enable access logging at the ASGI server or
ingress layer that owns request timing and retention.

The formatter can include LGOS context fields such as `request_id`, `model`,
`stream`, and `run_id`. If OpenTelemetry is enabled, its `trace_id` and
`span_id` fields can coexist with these application fields.

## Langfuse correlation

When `LGOS_ENABLE_LANGFUSE=true`, LGOS passes the following non-sensitive
metadata through LangChain's `RunnableConfig`:

- `lgos.request_id`, when an HTTP request context exists;
- `lgos.operation_id`, for an interrupt operation;
- `lgos.model`, the registered OpenAI-compatible graph model.

LGOS also uses the stable run name `lgos.chat_completion`. These values make a
Langfuse observation searchable alongside application logs without changing
LangChain's native execution `run_id` or Langfuse's generated trace ID.

Langfuse's LangChain integration creates a trace for each invocation by
default. Treat one Chat Completions request as one trace; use a Langfuse
session only when several requests form one user interaction. The
[Langfuse trace ID guidance](https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing)
supports deterministic custom trace IDs for a trusted external ID, but LGOS's
`X-Request-ID` is a correlation header and may be client-supplied or reused.
It is therefore metadata here, not a custom trace ID.

For end-to-end distributed tracing, configure the host application and proxy
to propagate W3C/OpenTelemetry context. LGOS does not install a second tracing
system or create a parent span around the HTTP request.
