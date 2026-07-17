---
hide:
  - toc
---

# How-To Guides

Task-oriented guides for configuring, securing, and deploying an LGOS
application.

<div class="grid cards" markdown>

-   :material-tune-variant:{ .lg .middle } __Configure runtime settings__

    Let OpenAI clients discover and choose a safe, typed subset of per-request
    graph behavior.

    [:octicons-arrow-right-24: Configure runtime settings](langgraph-runtime-settings.md)

-   :material-shield-lock-outline:{ .lg .middle } __Add authentication__

    Protect `/v1` with standard bearer tokens while preserving OpenAI client
    compatibility.

    [:octicons-arrow-right-24: Configure authentication](authentication.md)

-   :material-docker:{ .lg .middle } __Deploy with Docker__

    Run the complete demo stack or package a custom FastAPI application.

    [:octicons-arrow-right-24: Use Docker](docker.md)

-   :material-transit-connection-horizontal:{ .lg .middle } __Configure a proxy__

    Preserve graph features and runtime settings through an
    OpenAI-compatible gateway.

    [:octicons-arrow-right-24: Configure an OpenAI proxy](openai-proxy.md)

</div>
