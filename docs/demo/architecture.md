# Demo Architecture

The Docker demo puts two independently addressable LGOS API containers behind
one Bifrost gateway. Chainlit and Open WebUI are OpenAI-compatible clients of
that gateway; neither UI imports `langgraph-openai-serve`. See
[Package Architecture](../explanation/architecture.md) for what happens inside
each API process.

## Request Path

```mermaid
flowchart LR
  user["Browser user"]

  subgraph clients["Demo clients"]
    direction TB
    chainlit["Chainlit"]
    openwebui["Open WebUI"]
  end

  bifrost["Bifrost gateway"]

  subgraph apis["LGOS demo APIs"]
    direction TB
    api_a["API A<br/>FastAPI + LGOS + demo graphs"]
    api_b["API B<br/>FastAPI + LGOS + demo graphs"]
  end

  model["Upstream OpenAI-compatible model"]

  user <--> chainlit
  user <--> openwebui
  chainlit <-->|"catalog and OpenAI traffic"| bifrost
  openwebui <-->|"catalog and OpenAI traffic"| bifrost
  bifrost <-->|"provider: lgos-a"| api_a
  bifrost <-->|"provider: lgos-b"| api_b
  api_a <-->|"when a graph calls a model"| model
  api_b <-->|"when a graph calls a model"| model
```

Bifrost exposes one provider-qualified model catalog. The UIs split a selected
ID such as `lgos-a/simple-graph`, put the provider in `x-model-provider`, and
send the native graph name through Bifrost's raw `/openai_passthrough/v1`
route. Bifrost then forwards the request to the matching API container while
preserving LGOS model metadata and streaming extensions.

At startup, Compose waits for PostgreSQL, runs the one-shot API schema setup,
starts both healthy API containers, and then starts Bifrost and its UI clients.
The diagram shows request traffic rather than those readiness dependencies.

## State Ownership

The UIs own their conversations. The API stores only paused interrupt execution
and explicit graph data; it does not copy either UI transcript into LGOS.

```mermaid
flowchart LR
  subgraph clients["UI-owned state"]
    direction TB
    chainlit["Chainlit"]
    openwebui["Open WebUI"]
  end

  subgraph api["LGOS API processes"]
    direction TB
    interrupts["LGOS interrupt handling"]
    plot["persistent-plot graph"]
  end

  subgraph postgres["One PostgreSQL database"]
    direction TB
    chainlit_rows["Chainlit users, threads, and steps"]
    checkpoints["LangGraph checkpoints"]
    store["LangGraph Store documents"]
    locks["PostgreSQL advisory locks"]
  end

  s3[("S3-compatible storage<br/>Chainlit element bodies")]
  openwebui_data[("Open WebUI data volume<br/>transcripts and embeds")]

  chainlit -->|"conversation and UI metadata"| chainlit_rows
  chainlit -->|"element content"| s3
  openwebui -->|"conversation and UI state"| openwebui_data
  interrupts -->|"paused execution"| checkpoints
  interrupts -->|"same-run coordination"| locks
  plot -->|"thread-scoped chart document"| store
```

Both API containers currently run the same image and graph set, but Bifrost
treats them as separate providers. They share PostgreSQL for durable LangGraph
checkpoints, thread-scoped data, and cross-worker run coordination. Chainlit
uses the same database for UI metadata and S3 for element bodies. Open WebUI
keeps its state in its bind-mounted data directory. Detailed ownership and
recovery behavior live in [Persistent Plot](graphs/persistent-plot.md) and
[Interruptible Approval](graphs/interruptible-approval.md). When
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
