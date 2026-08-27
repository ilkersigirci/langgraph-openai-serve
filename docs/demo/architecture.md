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
  s3[("S3-compatible object store")]
  openwebui_data[("Open WebUI data volume")]
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
  chainlit -->|"element bodies"| s3
  openwebui -->|"UI state"| openwebui_data
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
uses the same database for UI metadata and S3 for element bodies. Open WebUI
keeps its state in its bind-mounted data directory. When
`LGOS_ENABLE_LANGFUSE=true`, each API adds the Langfuse callback to graph runs
and exports observations directly to the configured Langfuse service. Langfuse
is not a Compose service or a proxy in the request path.

## OpenTelemetry Overlay

The optional Compose overlay sends application telemetry through one local
Collector while keeping Langfuse on its separate native export path.

```mermaid
flowchart LR
  subgraph demo["Demo Compose deployment"]
    direction TB
    clients_otel["Chainlit and Open WebUI"]
    bifrost_otel["Bifrost"]
    apis_otel["LGOS API A and B"]
    collector["Local OpenTelemetry Collector"]

    clients_otel -->|"traces"| collector
    bifrost_otel -->|"traces"| collector
    apis_otel -->|"traces, metrics, and logs"| collector
  end

  collector -->|"OTLP/HTTP"| gateway["External Collector gateway"]
  gateway --> lgtm["Grafana LGTM"]
  apis_otel -.->|"LangGraph observations"| langfuse_otel["Langfuse"]
```

The local Collector filters, enriches, and forwards standard OTLP signals; the
external platform stores and displays them. Configuration and operational
details live in
[Production OpenTelemetry](../how-to-guides/production-otel.md).
