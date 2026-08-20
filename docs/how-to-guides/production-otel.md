# Production OpenTelemetry

Use the optional `demo/docker/compose/otel.yml` overlay when the deployment owns
an OpenTelemetry Collector. It keeps the application integration small:

```mermaid
flowchart LR
    proxy[Ingress or Traefik] --> api[LGOS API]
    api -->|OTLP traces, metrics, and logs| collector[Local OTel Collector]
    collector -->|OTLP/HTTP| gateway[Host or platform Collector gateway]
    api -->|stdout JSON diagnostics| runtime[Container runtime]
    api -->|Langfuse native OTLP exporter| langfuse[Langfuse]
```

The repository does not add a backend-specific exporter to the Collector. The
Collector forwards standard OTLP signals to the gateway owned by the host
deployment over OTLP/HTTP. Langfuse's existing LangChain callback remains
responsible for Langfuse observations and uses the shared OpenTelemetry
provider when one is already configured. The shared provider sends its spans
through the Collector, while Langfuse's processor separately selects and
exports Langfuse observations to Langfuse.

This follows the OpenTelemetry [Python instrumentation
model](https://opentelemetry.io/docs/languages/python/instrumentation/): the
application configures the SDK, while libraries remain instrumentation-neutral.
The [Collector documentation](https://opentelemetry.io/docs/collector/)
recommends a Collector for production batching, retries, filtering, and
network-boundary concerns. The deployment can place the gateway on the host,
per node, or in a separate platform service; a per-API sidecar is not required
for this Compose deployment.

## Run the overlay

From `demo/`, copy `.env.example` to `.env` and set
`OTEL_COLLECTOR_GATEWAY_ENDPOINT` to the OTLP/HTTP base URL of the host's
gateway. Use the Traefik OTLP router rather than the Grafana UI URL:

```dotenv
OTEL_COLLECTOR_GATEWAY_ENDPOINT=https://grafana-rpc.example.com
OTEL_COLLECTOR_GATEWAY_INSECURE=false
```

The Collector appends `/v1/traces`, `/v1/metrics`, and `/v1/logs` to this base
URL. The remote Traefik OTLP/HTTP router handles those paths. The Grafana UI
is available at `https://grafana.example.com`. This example does not send an
authorization header because the inspected gateway is restricted to trusted
LAN/VPN source ranges. If that trust boundary changes, configure matching
authentication at both the exporter and gateway; merely sending a header
without gateway-side validation provides no security.

Validate and start the published-image stack:

```bash
cd demo
docker compose -f docker/compose/demo.yml -f docker/compose/otel.yml config --quiet
make compose-otel
```

For a checkout-local image that includes the current API lockfile, use the
development build overlay as well:

```bash
docker compose -f docker/compose/demo.yml -f docker/compose/development.yml -f docker/compose/otel.yml up --build
```

Generate a request after the stack is healthy:

```bash
curl -i http://localhost:3004/v1/models \
  -H 'X-Request-ID: lgos-otel-e2e'
```

This exercises the FastAPI server span and request metrics without calling an
upstream model. A graph invocation can be used afterward when you also want
to inspect LangGraph and Langfuse observations.

The overlay:

- starts one `lgos-otel-collector` on separate ingest and egress networks;
- wraps both API workers with the official `opentelemetry-instrument` launcher;
- exports API traces, metrics, and standard-library logs to the local Collector
  over OTLP/gRPC;
- forwards those signals from the local Collector to the configured gateway
  over OTLP/HTTP;
- gives both replicas the same `service.name` and different
  SDK-generated `service.instance.id` values;
- opts into stable HTTP semantic conventions;
- aligns Go's memory limit and the Collector memory limiter with the 1 GiB
  container limit;
- enables parent-based trace sampling, configurable with
  `OTEL_TRACES_SAMPLE_RATE`;
- uses a persistent Collector sending queue under
  `demo/docker/volumes/otel-collector/`, with each queue database capped at
  256 MiB; and
- exports the Collector's own operational metrics directly to the gateway so
  queue pressure, rejected data, and send failures can be monitored without a
  self-referential pipeline.

The API container health check uses `/v1/health`, and the overlay excludes that
endpoint from FastAPI instrumentation so routine probes do not dominate HTTP
traces and metrics. The Collector remains fail-open for the application: its
SDK exporters retry startup races and temporary telemetry outages.

The local Collector accepts OTLP/gRPC on `4317` and OTLP/HTTP on `4318` only on
the dedicated ingest network. The overlay attaches the API replicas; attach
other producers deliberately. It does not publish either port to the host. If
an external Traefik instance must send signals to this Collector, attach it to
the ingest network or send Traefik to the host's gateway instead.

## Verify in Grafana

Open [Grafana](https://grafana.example.com), select **Explore**, and
choose the relevant built-in datasource:

- **Loki:** `{service_name="lgos-demo-api"}`
- **Prometheus:** `{service_name="lgos-demo-api"}`
- **Tempo:** search for service `lgos-demo-api`, or use TraceQL
  `{ resource.service.name = "lgos-demo-api" }`

The two API replicas share `service_name` and can be separated with
`service_instance_id`. The log records carry the OTel trace and span IDs, so a
request can be followed from a Traefik span to an API span and its application
log record. The HTTP request ID and LGOS interrupt operation ID remain
application-level correlation fields; they are not replaced by OTel.

The overlay opts into stable HTTP semantic conventions with
`OTEL_SEMCONV_STABILITY_OPT_IN=http`. Existing dashboards that use experimental
names such as `http_server_duration_milliseconds` must migrate to the stable
HTTP metric and attribute names.

Monitor the `lgos-otel-collector` service for
`otelcol_exporter_queue_size`, `otelcol_exporter_queue_capacity`, and the
`otelcol_exporter_*_failed_*` counters. Alert before the persistent queue or
file-storage budget is exhausted.

## Logs and correlation

The API keeps its structured JSON logs on stdout for container diagnostics. The
overlay also sets `OTEL_LOGS_EXPORTER=otlp` and enables the official Python
logging auto-instrumentation, so the same standard-library records are sent to
the Collector as native OpenTelemetry logs. `OTEL_PYTHON_LOG_CORRELATION=true`
adds fields such as `otelTraceID` and `otelSpanID` to the records, while the
OpenTelemetry log record context carries the trace/span relationship.
`OTEL_PYTHON_LOG_HANDLER_LEVEL=info` keeps the OTLP handler's severity floor
aligned with the demo's stdout handler.

OpenTelemetry currently marks Python logs as [Development
status](https://opentelemetry.io/docs/languages/python/). This deployment opts
into that native signal deliberately; pin the instrumentation and Collector
versions and review changes when upgrading them.

The request `X-Request-ID`, LGOS interrupt operation ID, and OpenTelemetry
`trace_id` remain different identifiers. The interrupt protocol exposes the
operation ID as `run_id`; logs and callback metadata call it `operation_id` to
avoid collision with LangChain execution run IDs. See [Production Logging and
Request Correlation](production-logging.md) for the ownership boundaries.

## Langfuse

When `LGOS_ENABLE_LANGFUSE=true`, the API's existing Langfuse callback creates
Langfuse observations. Langfuse documents using a [shared global provider with
its Langfuse span processor and an OTLP exporter](https://langfuse.com/faq/all/existing-otel-setup)
when an application already uses OpenTelemetry. That is the behavior provided
by this overlay: the shared provider sends application and Langfuse spans
through the Collector, while the Langfuse span processor also exports only its
Langfuse observations through the Langfuse endpoint. The same observation is
not sent to Langfuse twice.

Langfuse's LangChain callback records chain and model inputs and outputs. With a
shared provider, the general OTLP exporter also sends those span attributes to
the Collector and its backend. Configure Langfuse
[masking](https://langfuse.com/docs/observability/features/masking) and any
Collector-side filtering for both destinations before enabling it on sensitive
workloads. Use an isolated Langfuse tracer provider only when keeping these
spans out of the general APM is more important than one continuous distributed
trace.

Do not add a second Collector exporter to Langfuse unless the deployment has a
specific requirement to centralize that export. If you do, remove the direct
Langfuse exporter deliberately and verify sampling and billing; forwarding both
paths creates duplicate observations. Langfuse also recommends reviewing
[sensitive-data handling](https://opentelemetry.io/docs/security/handling-sensitive-data/)
before exporting application attributes or payloads.

## What the overlay does not do

The overlay does not:

- replace the ingress proxy's access logs or latency metrics;
- remove stdout diagnostics when OTLP log export is enabled;
- add manual spans to the LGOS library;
- instrument application payloads by itself (optional integrations such as
  Langfuse can add them); or
- turn the local Collector into a backend.

The upstream gateway is a required environment setting so the deployment must
choose its own network, TLS, authorization, retention, sampling, and backend
policy instead of silently sending telemetry to an example endpoint.
Exception messages, tracebacks, and caller-controlled correlation values can
still contain sensitive data. Apply minimization or redaction in the
application or Collector before remote export; the example pipeline does not
claim a content-safety boundary.
