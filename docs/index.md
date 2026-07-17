---
hide:
  - path
  - toc
---

# LangGraph OpenAI Serve

Serve registered LangGraph graphs as OpenAI `model` values through a standard
OpenAI-compatible `/v1` API.

[Get started :octicons-arrow-right-24:](tutorials/getting-started.md){ .md-button .md-button--primary }
[API reference](reference.md){ .md-button }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } __Run the demo__

    ---

    Start PostgreSQL and call a runnable graph with the OpenAI Python client.

    [:octicons-arrow-right-24: Getting started](tutorials/getting-started.md)

-   :material-code-braces:{ .lg .middle } __Use your SDK__

    ---

    Connect from Python or JavaScript, with regular or streaming responses.

    [:octicons-arrow-right-24: OpenAI clients](tutorials/openai-clients.md)

-   :material-graph-outline:{ .lg .middle } __Register a graph__

    ---

    Adapt schemas, publish runtime settings, configure streaming,
    and enable interrupts.

    [:octicons-arrow-right-24: Custom graphs](tutorials/custom-graphs.md)
    [:octicons-arrow-right-24: Runtime settings](how-to-guides/langgraph-runtime-settings.md)

-   :material-source-branch:{ .lg .middle } __Understand the contract__

    ---

    See how OpenAI requests flow through FastAPI into LangGraph and back.

    [:octicons-arrow-right-24: Architecture](explanation/architecture.md)

-   :material-transit-connection-horizontal:{ .lg .middle } __Use an integration__

    ---

    Connect Chainlit, Open WebUI, or an OpenAI-compatible proxy.

    [:octicons-arrow-right-24: Integrations](integrations/index.md)

</div>

!!! info "OpenAI compatibility is the public contract"

    LGOS keeps client ingestion on the OpenAI SDK path. Graph-specific behavior
    is adapted behind `/v1`, so clients do not need a project-specific protocol.

## Configure, Operate, And Deploy

Use the [runtime settings guide](how-to-guides/langgraph-runtime-settings.md)
to publish safe per-request graph settings, the
[authentication guide](how-to-guides/authentication.md) to add bearer tokens,
and the [Docker guide](how-to-guides/docker.md) to run a stack. Optional clients
and gateways are documented under [Integrations](integrations/index.md). See the
[reference](reference.md) for endpoints, settings, demo models, and public
classes.
