# Demo Architecture

The Docker demo puts two independently addressable LGOS API containers behind
one Bifrost gateway. Chainlit and Open WebUI are OpenAI-compatible clients of
that gateway; neither UI imports `langgraph-openai-serve`.

```mermaid
flowchart LR
  user["Browser"]

  subgraph clients["Demo clients"]
    direction TB
    chainlit["Chainlit"]
    openwebui["Open WebUI"]
  end

  bifrost["Bifrost gateway"]

  subgraph apis["LGOS demo APIs"]
    direction TB
    api_a["lgos-demo-api-a<br/>provider: lgos-a"]
    api_b["lgos-demo-api-b<br/>provider: lgos-b"]
  end

  postgres[("PostgreSQL")]
  model["Upstream OpenAI-compatible model"]
  langfuse["Langfuse<br/>(optional, external)"]

  user --> chainlit
  user --> openwebui
  chainlit -->|"model catalog and OpenAI pass-through"| bifrost
  openwebui -->|"model catalog and OpenAI pass-through"| bifrost
  bifrost -->|"x-model-provider: lgos-a"| api_a
  bifrost -->|"x-model-provider: lgos-b"| api_b
  api_a -->|"graph model calls"| model
  api_b -->|"graph model calls"| model
  api_a -->|"checkpoints, store, and coordination"| postgres
  api_b -->|"checkpoints, store, and coordination"| postgres
  chainlit -->|"chat and element metadata"| postgres
  api_a -.->|"LangGraph observations"| langfuse
  api_b -.->|"LangGraph observations"| langfuse
```

Bifrost exposes one provider-qualified model catalog. The UIs split a selected
ID such as `lgos-a/simple-graph`, put the provider in `x-model-provider`, and
send the native graph name through Bifrost's raw `/openai_passthrough/v1`
route. Bifrost then forwards the request to the matching API container while
preserving LGOS model metadata and streaming extensions.

At startup, Compose waits for PostgreSQL, runs the one-shot API schema setup,
starts both healthy API containers, and then starts Bifrost and its UI clients.
The diagram shows runtime traffic rather than those readiness dependencies.

Both API containers currently run the same image and graph set, but Bifrost
treats them as separate providers. They share PostgreSQL for durable LangGraph
checkpoints, thread-scoped data, and cross-worker run coordination. Chainlit
uses the same database for its own UI persistence; Open WebUI keeps its state
in its bind-mounted data directory. When `LGOS_ENABLE_LANGFUSE=true`, each API
adds the Langfuse callback to graph runs and exports observations directly to
the configured Langfuse service. Langfuse is not a Compose service or a proxy
in the request path.

## OpenTelemetry Overlay

The optional OpenTelemetry Compose overlay attaches the applications to a
private telemetry network and starts one local Collector. The Grafana LGTM
backend remains external to this demo deployment.

```mermaid
flowchart LR
  subgraph demo["Demo Compose deployment"]
    direction TB
    chainlit_otel["Chainlit"]
    openwebui_otel["Open WebUI"]
    bifrost_otel["Bifrost"]
    api_a_otel["LGOS API A"]
    api_b_otel["LGOS API B"]
    collector["Local OpenTelemetry Collector<br/>OTLP/gRPC :4317 · OTLP/HTTP :4318"]
    chainlit_otel -->|"traces"| collector
    openwebui_otel -->|"traces"| collector
    bifrost_otel -->|"traces"| collector
    api_a_otel -->|"traces, metrics, and logs"| collector
    api_b_otel -->|"traces, metrics, and logs"| collector
  end

  subgraph external["External observability platform"]
    direction LR
    gateway["Host or platform<br/>OTLP/HTTP gateway"]
    lgtm["Grafana LGTM backend<br/>logs · traces · metrics"]
    gateway --> lgtm
  end

  collector -->|"buffer, retry, filter, and forward"| gateway
  api_a_otel -.->|"separate native export"| langfuse["Langfuse"]
  api_b_otel -.->|"separate native export"| langfuse
```

The local Collector receives OTLP from the instrumented services, removes
unwanted transport spans and sensitive payload attributes, adds deployment
resource metadata, and forwards signals over OTLP/HTTP to the externally
configured gateway. The external LGTM stack stores logs, traces, and metrics
and exposes them through Grafana. Langfuse delivery remains a separate direct
export from the API workers; the Collector does not forward observations to
Langfuse.
