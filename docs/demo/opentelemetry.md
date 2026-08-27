# Demo OpenTelemetry Overlay

The optional `demo/docker/compose/otel.yml` overlay instruments the demo
deployment. It does not add telemetry behavior to the
`langgraph-openai-serve` package.

The demo applications send standard OTLP signals to one local OpenTelemetry
Collector. That Collector filters, enriches, batches, and forwards them to the
gateway selected by the deployment. The remote gateway and observability
backend are intentionally not bundled.

## Signal Path

```mermaid
flowchart LR
  subgraph demo["Demo Compose deployment"]
    direction TB
    clients["Chainlit and Open WebUI"]
    bifrost["Bifrost"]
    apis["LGOS API A and B"]
    collector["Local OpenTelemetry Collector"]

    clients -->|"traces"| collector
    bifrost -->|"traces"| collector
    apis -->|"traces, metrics, and logs"| collector
  end

  collector -->|"OTLP/HTTP"| gateway["External Collector gateway"]
  gateway --> backend["External observability backend<br/>(for example Grafana LGTM)"]
  apis -.->|"LangGraph observations"| langfuse["Langfuse"]
```

The local Collector handles standard OTLP signals. Langfuse remains a separate
native export path from the API processes.

## Run The Overlay

From `demo/`, copy `.env.example` to `.env`, then configure the required values:

```dotenv
OTEL_COLLECTOR_GATEWAY_ENDPOINT=https://otel-gateway.example.com
OTEL_HOST_NAME=demo-host-1
```

`OTEL_COLLECTOR_GATEWAY_ENDPOINT` is an OTLP/HTTP base URL, not an observability
UI URL. The Collector appends `/v1/traces`, `/v1/metrics`, and `/v1/logs`.
Use `https://` unless the gateway deliberately accepts cleartext traffic.

=== "Published images"

    ```bash
    make compose-otel
    ```

=== "Current checkout"

    ```bash
    make compose-otel-dev
    ```

Generate a provider-free request after the stack becomes healthy:

```bash
curl -i http://localhost:3004/v1/models \
  -H 'X-Request-ID: lgos-otel-e2e'
```

Exact environment defaults are listed in
[Demo Settings And Commands](reference.md#opentelemetry-settings).

## Signal Ownership

| Producer | Exported signals | Demo integration |
| --- | --- | --- |
| LGOS API processes | Traces, metrics, and logs | Python auto-instrumentation plus explicit instrumentation of the mounted `/v1` app |
| Chainlit | Traces | Python auto-instrumentation; long-lived Socket.IO traffic and prompt-recording OpenAI instrumentors are excluded |
| Open WebUI | Traces | Open WebUI's native OpenTelemetry settings |
| Bifrost | Traces | Bifrost's OpenTelemetry plugin with content logging disabled |
| Local Collector | Its own metrics | Direct OTLP/HTTP export to the configured gateway |

The package itself remains instrumentation-neutral. The demo API instruments
the mounted FastAPI application so spans retain LGOS route templates without
duplicate host-application spans. W3C trace context connects requests across
the UI, proxy, gateway, and API when every hop preserves `traceparent`.

The API also keeps structured JSON logs on stdout. Enabling OTLP logs adds a
second delivery path for those standard-library records; it does not remove
container diagnostics. `X-Request-ID`, the LGOS interrupt operation ID, and
the OpenTelemetry trace ID remain separate correlation values. See
[Production Logging](../how-to-guides/production-logging.md) for their ownership.

## Collector Behavior

The local Collector:

- accepts OTLP/gRPC and OTLP/HTTP only on its internal ingest network;
- removes streamed ASGI transport spans and known prompt/response payload
  attributes before data reaches its persistent queue;
- adds the configured service namespace, environment, and host identity;
- retries and batches export through a file-backed queue capped at 256 MiB;
  and
- forwards traces, metrics, and logs over OTLP/HTTP.

These filters reduce accidental payload export but are not a complete
redaction boundary. Exceptions, tracebacks, and caller-controlled values can
still contain sensitive data.

To verify the pipeline, locate service `lgos-demo-api` in the configured
backend and correlate the request with `lgos-otel-e2e`. The two API replicas
share `service.name` and have different SDK-generated `service.instance.id`
values. Monitor the Collector's exporter queue, capacity, send-failure, and
rejected-data metrics in that backend.

## Langfuse Remains Separate

When `LGOS_ENABLE_LANGFUSE=true`, LGOS adds the Langfuse callback to graph runs.
Langfuse exports its observations through its native integration; the local
Collector is not a Langfuse proxy. Do not add a second Langfuse exporter unless
the deployment intentionally owns and tests that additional path.

The Collector removes known Langfuse and GenAI payload attributes only from the
general OTLP pipeline. Configure Langfuse's own masking and retention policy
before enabling it for sensitive workloads.

## Deployment Responsibilities

The overlay does not choose the remote backend or its authentication,
retention, sampling, access-control, and capacity policies. It also does not
replace ingress access logs or add manual spans to LGOS. Before production use,
the deployment must:

- secure the OTLP gateway and validate TLS and authentication;
- measure representative trace volume and size backend retention accordingly;
- define payload minimization and redaction rules; and
- monitor the Collector queue and remote-export failures.

The OpenTelemetry [Collector](https://opentelemetry.io/docs/collector/) and
[sensitive-data](https://opentelemetry.io/docs/security/handling-sensitive-data/)
guides define the upstream operational boundaries.
