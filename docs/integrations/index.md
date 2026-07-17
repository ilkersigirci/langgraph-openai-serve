---
hide:
  - toc
---

# Integrations

The repository includes optional clients and gateway configurations built on
the OpenAI-compatible LGOS contract. They are not part of the core API.

<div class="grid cards" markdown>

-   :material-message-processing-outline:{ .lg .middle } __Chainlit__

    Model discovery, Chat Settings, persistence, login, streaming, and HITL.

    [:octicons-arrow-right-24: Use Chainlit](chainlit.md)

-   :material-chat-outline:{ .lg .middle } __Open WebUI__

    Model discovery, streaming, citations, and interrupt approval through the
    included manifold Pipe.

    [:octicons-arrow-right-24: Use Open WebUI](open-webui.md)

-   :material-transit-connection-horizontal:{ .lg .middle } __Proxies__

    Route inference and detailed model discovery through Bifrost, LiteLLM, or
    another OpenAI-compatible gateway.

    [:octicons-arrow-right-24: Configure a proxy](openai-proxies.md)

</div>

## Capabilities

| Integration | Runtime settings | Interrupts | Discovery requirement |
| --- | --- | --- | --- |
| Chainlit | Top-level booleans and strings | Separate HITL UI | Detailed model retrieval |
| Open WebUI Pipe | Server defaults only | Approval through the Pipe | Standard model list |
| OpenAI-compatible proxy | Preserved when metadata is forwarded | Preserved when tool calls and metadata are forwarded | Detailed extension requires direct LGOS or a raw pass-through route |

Direct OpenAI SDK clients use the core contract without an integration adapter.
See [OpenAI clients](../tutorials/openai-clients.md).
