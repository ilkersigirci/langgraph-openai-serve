# Demo Architecture

The Docker demo runs two independently addressable LGOS API containers and one
logical central Files service. `OPENAI_GATEWAY_TYPE=litellm|bifrost` selects a
first-class OpenAI-compatible edge for both Chainlit and Open WebUI. LiteLLM
uses managed Responses and Bifrost uses native Responses. Both use normal
Files routing and reserve pass-through for detailed model catalogs. No UI
connects directly to an upstream container, and neither UI imports
`langgraph-openai-serve`. See
[Package Architecture](../explanation/architecture.md) for what happens inside
each API process.

!!! warning "Managed gateway normalization boundaries"

    Bifrost v2.0.0 native Responses preserves the tested standard fields, file
    input, commentary, and `phase`; normalized model detail and error metadata
    remain lossy. Its raw pass-through route passes the complete direct
    contract suite. LiteLLM 1.99.1 managed wildcard routing synthesizes streams
    and rewrites error metadata. The UIs exercise the selected gateway's
    managed/native inference path; only model-detail lookup uses a lossless
    pass-through. See [Bifrost Gateway](bifrost.md).

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
  litellm["LiteLLM 1.99.1<br/>managed inference + pass-through catalogs"]
  gateway["OPENAI_GATEWAY_TYPE<br/>selects one gateway"]
  sdk["OpenAI SDK test"]

  subgraph apis["LGOS demo APIs"]
    direction TB
    api_a["API A<br/>FastAPI + LGOS + demo graphs"]
    api_b["API B<br/>FastAPI + LGOS + demo graphs"]
  end

  files["Files service<br/>OpenAI Files API + S3 repository"]

  model["Upstream OpenAI-compatible model"]

  user <--> chainlit
  user <--> openwebui
  chainlit <-->|"OpenAI API"| gateway
  openwebui <-->|"OpenAI API"| gateway
  gateway <-.->|"bifrost"| bifrost
  gateway <-.->|"litellm"| litellm
  sdk <-->|"catalog + native/raw Responses"| bifrost
  sdk <-->|"Responses"| litellm
  bifrost <-->|"provider: lgos-a"| api_a
  bifrost <-->|"provider: lgos-b"| api_b
  bifrost <-->|"provider: lgos-files"| files
  litellm <-->|"managed + catalog detail"| api_a
  litellm <-->|"managed + catalog detail"| api_b
  litellm <-->|"provider: litellm_proxy"| files
  api_a <-->|"when a graph calls a model"| model
  api_b <-->|"when a graph calls a model"| model
```

With LiteLLM selected, the UIs merge the explicit `lgos-a` and `lgos-b`
catalog pass-throughs, qualify graph IDs, and send Responses through managed
routing. With Bifrost selected, they discover provider-qualified IDs through
its aggregate catalog, use raw pass-through only for model detail, and send
Responses through native routing with `x-model-provider`. Both choices upload
attachments through normal gateway Files routing before sending the returned
`file_id` to a graph. This preserves descriptions and runtime capabilities
without allowing UI inference to bypass the gateway's normal data plane.

LiteLLM wildcard aliases cover arbitrary graph names; exact `status-events`
entries prove native stream preservation because 1.99.1 otherwise synthesizes
streams for wildcard model names. Raw gateway routes remain available for
protocol-reference tests, but the UI clients do not use them for Responses.

At startup, Compose waits for PostgreSQL, runs the one-shot API schema setup,
starts both healthy graph APIs and the Files service, and then starts Bifrost,
LiteLLM, and the UI clients. The diagram shows request traffic rather than those
readiness dependencies. Compose runs one Files process for the demo; production
deployments may run multiple stateless replicas over the same repository.

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
    plot["persistent-plot-agent graph"]
  end

  subgraph postgres["One PostgreSQL database"]
    direction TB
    chainlit_rows["Chainlit users, threads, and steps"]
    checkpoints["LangGraph checkpoints"]
    store["LangGraph Store documents"]
    locks["PostgreSQL advisory locks"]
  end

  files_service["Central Files service"]
  s3[("S3-compatible service<br/>separate UI and Files buckets")]
  openwebui_data[("Open WebUI data volume<br/>transcripts, raw uploads, and embeds")]

  chainlit -->|"conversation and UI metadata"| chainlit_rows
  chainlit -->|"element content"| s3
  files_service -->|"opaque inference files"| s3
  openwebui -->|"conversation and UI state"| openwebui_data
  interrupts -->|"paused execution"| checkpoints
  interrupts -->|"same-run coordination"| locks
  plot -->|"thread-scoped chart document"| store
```

Both API containers run the same image and graph set, but Bifrost
treats them as separate providers. They share PostgreSQL for durable LangGraph
checkpoints, thread-scoped data, and interrupt-run coordination. Chainlit
uses the same database for UI metadata and S3 for element bodies. Open WebUI
keeps its state and native raw-upload copy in its bind-mounted data directory;
the central Files service owns the separate inference copy. Detailed ownership
and recovery behavior live in
[Persistent Plot Agent](graphs/persistent-plot-agent.md) and [Interruptible
Human Review](graphs/interruptible-approval.md). When
`LGOS_ENABLE_LANGFUSE=true`, each API adds the Langfuse callback to graph runs
and exports observations directly to the configured Langfuse service. Langfuse
is not a Compose service or a proxy in the request path.

The optional Compose overlay adds a separate telemetry path without changing
request or state ownership. Its complete signal flow and operational boundary
are documented in [Demo OpenTelemetry Overlay](opentelemetry.md).
