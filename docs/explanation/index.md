---
hide:
  - toc
---

# How It Works

LGOS is an HTTP adapter around application-owned LangGraph graphs. Start with
the architecture, then read only the detail needed for your task.

<div class="grid cards" markdown>

-   :material-source-branch:{ .lg .middle } __Architecture__

    The request path, package boundary, and ownership of conversation and graph
    state.

    [:octicons-arrow-right-24: Start here](architecture.md)

-   :material-swap-horizontal:{ .lg .middle } __LangGraph integration__

    How registration, adapters, the runner, streaming, interrupts, and
    cancellation work internally.

    [:octicons-arrow-right-24: Follow the execution](langgraph-integration.md)

-   :material-api:{ .lg .middle } __OpenAI compatibility__

    The precise client-facing contract for discovery, events, citations,
    errors, and interrupt replay.

    [:octicons-arrow-right-24: Read the contract](openai-compatibility.md)

</div>
