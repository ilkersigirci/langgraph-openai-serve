# Production Logging and Request Correlation

LGOS emits standard-library `logging.LogRecord` objects. The host application
chooses how those records are formatted, collected, retained, and exported.

## LGOS responsibilities

For HTTP requests handled by the mounted OpenAI-compatible application, LGOS:

- accepts `X-Request-ID`, or generates a UUID4 when the header is missing or
  unusable;
- returns the request ID in the `X-Request-ID` response header, including on
  validation, error, and streaming responses;
- adds `request_id`, `model`, `stream`, and, when available, LangGraph `run_id`
  to LGOS log records through request-scoped context; and
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
LGOS and Uvicorn records together without changing the LGOS package.

The formatter can include LGOS context fields such as `request_id`, `model`,
`stream`, and `run_id`. If OpenTelemetry is enabled, its `trace_id` and
`span_id` fields can coexist with these application fields.
