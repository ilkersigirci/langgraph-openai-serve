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

    Compare schema adapters, RAG, citations, direct Chat events, file output,
    subgraphs, and HITL.

    [:octicons-arrow-right-24: Example graphs](graphs/index.md)

-   :material-docker:{ .lg .middle } __Run the complete stack__

    Use published images or build the demo applications with an editable LGOS
    checkout.

    [:octicons-arrow-right-24: Docker Compose](docker.md)

-   :material-message-processing-outline:{ .lg .middle } __Use Chainlit__

    Responses, model discovery, Chat Settings, persistence, login, and HITL.

    [:octicons-arrow-right-24: Use Chainlit](chainlit.md)

-   :material-chat-outline:{ .lg .middle } __Use Open WebUI__

    A Responses manifold Pipe with generated graph-specific Workspace Models.

    [:octicons-arrow-right-24: Use Open WebUI](open-webui.md)

-   :material-transit-connection-horizontal:{ .lg .middle } __Route through Bifrost__

    Select the native UI inference path and inspect its pinned compatibility
    boundary.

    [:octicons-arrow-right-24: Bifrost gateway](bifrost.md)

-   :material-gateway:{ .lg .middle } __Use the LiteLLM edge__

    Select the managed UI inference route, catalog-detail pass-through, and
    compatibility tests.

    [:octicons-arrow-right-24: LiteLLM in Compose](docker.md)

-   :material-chart-timeline-variant:{ .lg .middle } __Observe the stack__

    Add the optional Collector overlay without changing the LGOS package.

    [:octicons-arrow-right-24: OpenTelemetry overlay](opentelemetry.md)

</div>

## Components

| Component | Demo-owned responsibility | Distribution |
| --- | --- | --- |
| Demo APIs | Two FastAPI graph services that may expose different graph sets | One independent uv project; Compose runs the `lgos-demo-api` image twice |
| Files API | Shared OpenAI file namespace and S3 persistence | Independent uv project and `lgos-files-api` image |
| Chainlit | Persistent Responses client, login, settings UI, file display, and HITL UI | Independent uv project and `lgos-chainlit` image |
| Open WebUI | Responses manifold plus dynamic generated Workspace Models | Independent uv project; Open WebUI uses its official image |
| Bifrost | Shared model catalog plus provider-selected native OpenAI routing | Compose configuration with the official image |
| LiteLLM | Selectable managed UI inference edge plus catalog-detail pass-through | Pinned official image and Compose configuration |
| PostgreSQL | Thread-scoped graph data, pending interrupts, cross-worker interrupt coordination, and Chainlit persistence | Official image with a demo-owned bind directory |
| S3-compatible storage | Files API objects and separate Chainlit element bodies | External endpoint with independently configured buckets |

Only the graph API project imports `langgraph-openai-serve`. The Files API
implements its independent OpenAI Files contract without importing LGOS.
Chainlit and Open WebUI exercise the graph API's OpenAI wire contract without
importing the package. `OPENAI_GATEWAY_TYPE=litellm|bifrost` selects their
shared edge. Responses and Files use its normal managed/native routes; only
catalog detail uses pass-through to preserve LGOS extensions.

!!! warning "Pinned managed-routing limitations"

    Bifrost v2.0.0 native Responses preserves the tested `phase`, commentary,
    file-input, and continuation contracts; only normalized model-detail and
    error metadata remain strict expected failures. Its raw pass-through route
    passes the direct contract suite. LiteLLM 1.99.1 managed wildcard routing
    synthesizes the upstream stream and rewrites standard error metadata.
    Pass-through routes remain the lossless protocol references. The UIs use
    the selected gateway's normal inference route and accept that route's
    documented limitations; see [Docker Compose](docker.md) and [Bifrost
    Gateway](bifrost.md) for the precise boundaries.

## Client Capabilities

| Demo client | File input | Missing LGOS metadata | Runtime settings | Interrupts | UI feedback | Citations |
| --- | --- | --- | --- | --- | --- | --- |
| Chainlit | Uploads attachments to the central Files API | Limited-functionality profile and warning toast | Renders supported discovered fields | Native choices and free-text input with a durable ledger | Native status and persisted image elements | Markdown content |
| Open WebUI generated models | Uploads attachments to the central Files API | Limited-functionality model description and warning notification | Renders supported discovered fields as Chat Variables | Persisted native `ask_user` card with LGOS replay | Native status and persisted file events | Native source events and Markdown |

Ordinary graph conversations work through an OpenAI SDK without a demo adapter.
An interrupt uses standard Responses function calls, but a client application
must recognize `langgraph_interrupt`, collect human answers, and replay the
canonical `function_call`/`function_call_output` exchange. The Chainlit and
Open WebUI adapters show that client behavior without importing LGOS. See
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
