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

-   :material-file-upload-outline:{ .lg .middle } __Accept and display files__

    Store opaque uploads in an external Files service and give graphs native
    OpenAI `file_id` references, then present generated files with function
    calls.

    [:octicons-arrow-right-24: Configure files](file-inputs.md)

-   :material-transit-connection-horizontal:{ .lg .middle } __Use a proxy__

    Preserve native Responses items, metadata, Files operations, errors, and
    streaming cancellation through an OpenAI-compatible gateway.

    [:octicons-arrow-right-24: Configure a proxy](openai-proxies.md)

</div>
