---
hide:
  - toc
---

# Self-Contained Demo Stack

The `demo/` directory is a portable example distribution built around
`langgraph-openai-serve`. It contains independently locked applications,
client integrations, gateway configuration, and a complete Compose stack.

!!! info "Package and demo have different owners"

    The PyPI package provides the OpenAI-compatible server integration. It
    contains no built-in graph, UI, gateway, database, or runnable application.
    Everything described in this section belongs to `demo/` and can be copied
    or operated independently of the package source checkout with published
    images. The development Compose override intentionally uses the parent
    checkout as an editable API dependency.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } __Run the API__

    Start PostgreSQL and call the example graphs through an OpenAI client.

    [:octicons-arrow-right-24: Run the API](api.md)

-   :material-file-upload-outline:{ .lg .middle } __Run the Files API__

    Start the independent OpenAI Files service backed by S3-compatible storage.

    [:octicons-arrow-right-24: Run the Files API](files-api.md)

-   :material-family-tree:{ .lg .middle } __Understand the architecture__

    See service ownership, request routing, and persistence boundaries.

    [:octicons-arrow-right-24: Demo architecture](architecture.md)

-   :material-graph-outline:{ .lg .middle } __Explore the graphs__

    Compare schema adapters, RAG, citations, client events, subgraphs, and HITL.

    [:octicons-arrow-right-24: Example graphs](graphs/index.md)

-   :material-docker:{ .lg .middle } __Run the complete stack__

    Use published images or build the demo applications with an editable LGOS
    checkout.

    [:octicons-arrow-right-24: Docker Compose](docker.md)

-   :material-message-processing-outline:{ .lg .middle } __Use Chainlit__

    Model discovery, Chat Settings, persistence, login, streaming, and HITL.

    [:octicons-arrow-right-24: Use Chainlit](chainlit.md)

-   :material-chat-outline:{ .lg .middle } __Use Open WebUI__

    A general manifold Pipe plus a dedicated `simple-graph` UserValve demo.

    [:octicons-arrow-right-24: Use Open WebUI](open-webui.md)

-   :material-transit-connection-horizontal:{ .lg .middle } __Route through Bifrost__

    Route two independently addressable LGOS APIs through one raw OpenAI
    pass-through endpoint.

    [:octicons-arrow-right-24: Bifrost gateway](bifrost.md)

-   :material-chart-timeline-variant:{ .lg .middle } __Observe the stack__

    Add the optional Collector overlay without changing the LGOS package.

    [:octicons-arrow-right-24: OpenTelemetry overlay](opentelemetry.md)

</div>

## Components

| Component | Demo-owned responsibility | Distribution |
| --- | --- | --- |
| Demo APIs | Two FastAPI graph services that may expose different graph sets | One independent uv project; Compose runs the `lgos-demo-api` image twice |
| Files API | Shared OpenAI file namespace and S3 persistence | Independent uv project and `lgos-files-api` image |
| Chainlit | Persistent OpenAI client, login, settings UI, events, and HITL UI | Independent uv project and `lgos-chainlit` image |
| Open WebUI | Dynamic generated models plus a static UserValves example | Independent uv project; Open WebUI uses its official image |
| Bifrost | Shared model catalog plus provider-selected raw pass-through | Compose configuration with the official image |
| PostgreSQL | Thread-scoped graph data, pending interrupts, cross-worker interrupt coordination, and Chainlit persistence | Official image with a demo-owned bind directory |
| S3-compatible storage | Files API objects and separate Chainlit element bodies | External endpoint with independently configured buckets |

Only the graph API project imports `langgraph-openai-serve`. The Files API
implements its independent OpenAI Files contract without importing LGOS.
Chainlit and Open WebUI exercise the graph API's OpenAI wire contract without
importing the package. Their dynamic clients use Bifrost's catalog for
provider-qualified discovery and raw pass-through for model details and chat.
The fixed-model Open WebUI example uses Bifrost pass-through without catalog
discovery. Pass-through preserves the required LGOS model-detail extension.

## Client Capabilities

| Demo client | File input | Missing LGOS metadata | Runtime settings | Interrupts | Client events | Citations |
| --- | --- | --- | --- | --- | --- | --- |
| Chainlit | Uploads attachments to the central Files API | Limited-functionality profile and warning toast | Renders supported discovered fields | Native choices and free-text input with a durable ledger | Native status, Plotly, and live activity elements | Markdown content |
| Open WebUI generated models | Uploads attachments to the central Files API | Limited-functionality model description and warning notification | Renders supported discovered fields as Chat Variables | Persisted native `ask_user` card with LGOS replay | Native status and persisted chart embeds | Native source events and Markdown |
| Open WebUI static example | Not implemented | Warning notification | Fixed `simple-graph` UserValves | None | Not requested | Assistant text only |

Ordinary graph conversations work through an OpenAI SDK without a demo adapter.
An interrupt still uses standard OpenAI `tool_calls`, but a client application
must recognize `langgraph_interrupt`, collect human answers, and replay the
canonical assistant/tool exchange. The Chainlit and Open WebUI adapters show
that client behavior without importing LGOS. See
[OpenAI Clients](../tutorials/openai-clients.md).

## Persistence Boundary

The UI owns chat history. LGOS stores resumable interrupt state and explicit
thread-scoped application data, not the transcript. PostgreSQL provides the
checkpointer, LangGraph store, and cross-worker interrupt coordination, with no
Redis service.
See [Persistent Plot Agent](graphs/persistent-plot-agent.md#ownership-boundaries)
for Store and UI ownership,
[Interruptible Human Review](graphs/interruptible-approval.md#postgresql-runtime)
for the server lifecycle, and
[OpenAI Compatibility](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for the normative replay and retention contract.

Chainlit persists its pending tool-call ledger with a documented crash window.
Open WebUI persists its native `ask_user` card and opaque graph cursor on the
assistant message. Their exact recovery boundaries are documented on the
[Chainlit](chainlit.md#interrupt-demo) and
[Open WebUI](open-webui.md#interrupt-input) pages.

For exact commands and environment ownership, use
[Demo Settings and Commands](reference.md). To build your own application,
[get started with the package](../getting-started.md).
