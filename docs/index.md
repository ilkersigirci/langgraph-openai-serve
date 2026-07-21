---
hide:
  - path
  - toc
---

# LangGraph OpenAI Serve

Serve registered LangGraph graphs as OpenAI `model` values through a standard
OpenAI-compatible `/v1` API.

[Build your first app :octicons-arrow-right-24:](getting-started.md){ .md-button .md-button--primary }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } __Use the package__

    ---

    Install LGOS, register your graph, and expose it through `/v1`.

    [:octicons-arrow-right-24: Getting started](getting-started.md)

-   :material-view-dashboard-outline:{ .lg .middle } __Explore the demo stack__

    ---

    Run the independent API, graphs, PostgreSQL, Chainlit, Open WebUI, and
    Bifrost examples.

    [:octicons-arrow-right-24: Demo capabilities](demo/index.md)

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

-   :material-transit-connection-horizontal:{ .lg .middle } __Use a gateway__

    ---

    Preserve metadata, discovery, tool calls, and streaming behavior through an
    OpenAI-compatible proxy.

    [:octicons-arrow-right-24: Proxy guide](how-to-guides/openai-proxies.md)

</div>

!!! info "OpenAI compatibility is the public contract"

    LGOS keeps client ingestion on the OpenAI SDK path. Graph-specific behavior
    is adapted behind `/v1`, so clients do not need a project-specific protocol.

!!! note "Package versus demo"

    The package supplies the server integration and public Python API. The
    repository's `demo/` directory supplies example graphs, applications,
    images, UIs, gateway configuration, and database-backed workflows. Demo
    components are independently locked and are not installed with LGOS.

## Configure And Operate

Use the [runtime settings guide](how-to-guides/langgraph-runtime-settings.md)
to publish safe per-request graph settings, the
[authentication guide](how-to-guides/authentication.md) to add bearer tokens,
and the [proxy guide](how-to-guides/openai-proxies.md) to preserve the contract
through a gateway. See the package [reference](reference.md) for endpoints,
settings, events, and public classes. Demo-owned models, settings, and commands
have their own [reference](demo/reference.md).
