# Configure an OpenAI Proxy

LGOS exposes registered graphs through the OpenAI-compatible Chat Completions
API. Any proxy that supports a configurable OpenAI-compatible upstream,
including LiteLLM and Bifrost, can therefore proxy LGOS inference without a
project-specific adapter.

Normal proxy routes are sufficient for chat completions. Model discovery needs
extra care because proxies may rebuild `/v1/models` responses and discard the
`langgraph_openai_serve` extension.

## Bifrost

Register LGOS as a keyless custom OpenAI provider:

```json
{
  "providers": {
    "lgos": {
      "keys": [],
      "network_config": {
        "base_url": "http://YOUR_IP_ADDRESS:8000",
        "allow_private_network": true
      },
      "custom_provider_config": {
        "base_provider_type": "openai",
        "is_key_less": true,
        "allowed_requests": {
          "list_models": true,
          "chat_completion": true,
          "chat_completion_stream": true,
          "passthrough": true,
          "passthrough_stream": true
        }
      }
    }
  }
}
```

Set `allow_private_network` only when the LGOS URL resolves to a private network.
The base URL must not include `/v1`; Bifrost forwards the path after removing
`/openai_passthrough`.

Select the `lgos` provider explicitly because Bifrost's OpenAI pass-through
route defaults to its `openai` provider:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://gateway.example/openai_passthrough/v1",
    api_key="BIFROST_API_KEY",
    default_headers={"x-model-provider": "lgos"},
)

models = client.models.list()
```

Use `/openai_passthrough/v1/models` for discovery. Use Bifrost's normal
`/v1/chat/completions` route for inference when you need request parsing,
routing, caching, or fallbacks; select graphs there as `lgos/<graph-name>`.

Pass-through still uses Bifrost's core and plugin pipeline, but skips
route-level conversion:

| Retained | Reduced or unavailable |
| --- | --- |
| Logging, tracing, metrics, authentication, governance, budgets, rate limits, and raw upstream responses | Normalization, semantic caching, routing transformations, model/provider load balancing, configured cross-provider fallbacks, and plugins that require a parsed request |

See the [Bifrost pass-through contract](https://docs.getbifrost.ai/integrations/passthrough)
and [custom-provider configuration](https://docs.getbifrost.ai/providers/custom-providers).

## Other Proxies

Use the proxy's native OpenAI route for chat completions. For graph feature
discovery, use only a documented raw pass-through route that can target LGOS,
and verify `/models` after proxy upgrades. LiteLLM documents
[`/openai_passthrough`](https://docs.litellm.ai/docs/pass_through/openai_passthrough)
for direct OpenAI access; its native model catalog does not guarantee that LGOS
extensions survive. Use direct LGOS discovery when no configurable raw route is
available.
