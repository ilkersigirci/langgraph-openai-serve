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

-   :material-graph-outline:{ .lg .middle } __Explore the graphs__

    Compare schema adapters, RAG, citations, client events, subgraphs, and HITL.

    [:octicons-arrow-right-24: Example graphs](graphs.md)

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

    Combine two independently deployable LGOS services behind one model catalog.

    [:octicons-arrow-right-24: Bifrost gateway](bifrost.md)

</div>

## Components

| Component | Demo-owned responsibility | Distribution |
| --- | --- | --- |
| Demo APIs | Two FastAPI graph services behind distinct Bifrost model prefixes | Independent uv project; the demo runs `lgos-demo-api` twice |
| Chainlit | Persistent OpenAI client, login, settings UI, events, and approval UI | Independent uv project and `lgos-chainlit` image |
| Open WebUI | Dynamic generated models plus a static UserValves example | Independent uv project; Open WebUI uses its official image |
| Bifrost | Combined model catalog and raw provider-selected pass-through | Compose configuration with the official image |
| PostgreSQL | LangGraph checkpoints and Chainlit persistence | Official image with a demo-owned bind directory |

Only the APIs import `langgraph-openai-serve`. Chainlit and Open WebUI exercise
the OpenAI wire contract without importing the package. Bifrost lets both
clients discover the two services as one catalog while preserving LGOS
extensions through raw pass-through requests.

## Client Capabilities

| Demo client | Runtime settings | Interrupts | Client events | Citations |
| --- | --- | --- | --- | --- |
| Chainlit | Renders supported discovered fields | Dedicated approval UI | Native status task list and live activity panel | Markdown content |
| Open WebUI generated models | Renders supported discovered fields as Chat Variables | Approval through the Pipe | Native status updates | Streaming annotations and Markdown |
| Open WebUI static example | Fixed `simple-graph` UserValves | None | Not requested | Assistant text only |

Direct OpenAI SDK clients need no demo adapter. They can use every core field
their own application handles, as shown in
[OpenAI Clients](../tutorials/openai-clients.md).

For exact commands and environment ownership, use
[Demo Settings and Commands](reference.md). To build your own application,
[get started with the package](../getting-started.md).
