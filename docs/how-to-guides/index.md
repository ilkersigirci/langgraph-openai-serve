---
hide:
  - toc
---

# How-To Guides

Task-oriented guides for configuring, securing, and connecting an LGOS
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

-   :material-text-box-search-outline:{ .lg .middle } __Operate production logs__

    Correlate HTTP requests with `X-Request-ID` and connect LGOS records to the
    logging and tracing system owned by your deployment.

    [:octicons-arrow-right-24: Configure production logging](production-logging.md)

-   :material-chart-timeline-variant:{ .lg .middle } __Export OpenTelemetry__

    Add an opt-in Collector overlay for traces, metrics, and application logs
    while keeping stdout diagnostics and Langfuse observations available.

    [:octicons-arrow-right-24: Configure production OpenTelemetry](production-otel.md)

-   :material-transit-connection-horizontal:{ .lg .middle } __Use a proxy__

    Preserve metadata, tool calls, discovery extensions, events, and streaming
    cancellation through an OpenAI-compatible gateway.

    [:octicons-arrow-right-24: Configure a proxy](openai-proxies.md)

</div>
