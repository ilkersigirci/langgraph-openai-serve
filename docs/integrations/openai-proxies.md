# OpenAI-Compatible Proxies

Use an OpenAI-compatible proxy for inference without an LGOS-specific adapter.
The proxy requirements for model discovery, request metadata, and streaming
disconnects are defined by the
[OpenAI compatibility contract](../explanation/openai-compatibility.md).
This page applies that contract to Bifrost and LiteLLM.

## Bifrost

The Compose stack mounts the repository's
[Bifrost configuration](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docker/bifrost/config.json).
It dedicates Bifrost's built-in `openai` provider to LGOS because the documented
`/openai_passthrough` route selects that provider by default.

Set `allow_private_network` only when the LGOS URL resolves to a private network.
Keep the provider base URL free of `/v1`, and replace the demo's `DUMMY` key when
LGOS requires authentication. See [Docker](../how-to-guides/docker.md) for
startup and endpoint URLs.

Use Bifrost's normal OpenAI-compatible route for inference:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://gateway.example/v1",
    api_key="BIFROST_API_KEY",
)

response = client.chat.completions.create(
    model="openai/simple-graph",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Use Bifrost's documented raw OpenAI pass-through for detailed model discovery:

```python
discovery_client = OpenAI(
    base_url="https://gateway.example/openai_passthrough/v1",
    api_key="BIFROST_API_KEY",
)

model = discovery_client.models.retrieve("simple-graph")
```

The pass-through returns the upstream body without route-level conversion, so
the `langgraph_openai_serve` extension is preserved.

The [Chainlit integration](chainlit.md) keeps the two URLs explicit:

```dotenv
DEMO_CHAINLIT_INFERENCE__BASE_URL=https://gateway.example/v1
DEMO_CHAINLIT_INFERENCE__API_KEY=BIFROST_API_KEY
DEMO_CHAINLIT_INFERENCE_MODEL_PREFIX=openai/
DEMO_CHAINLIT_DISCOVERY__BASE_URL=https://gateway.example/openai_passthrough/v1
DEMO_CHAINLIT_DISCOVERY__API_KEY=BIFROST_API_KEY
```

See the [Bifrost pass-through contract](https://docs.getbifrost.ai/integrations/passthrough)
and [provider configuration](https://docs.getbifrost.ai/quickstart/gateway/provider-configuration).

## LiteLLM

LiteLLM's native model catalog can serve standard model objects but does not
guarantee forwarding LGOS extensions. Configure a distinct pass-through prefix
that targets LGOS and includes subpaths:

```yaml
general_settings:
  pass_through_endpoints:
    - path: "/lgos-discovery"
      target: "http://lgos-api:8000"
      include_subpath: true
      methods: ["GET"]
```

The resulting discovery base URL is
`https://gateway.example/lgos-discovery/v1`; the normal LiteLLM `/v1` base URL
remains the inference URL. Configure the complete `DEMO_CHAINLIT_DISCOVERY`
endpoint when the pass-through route uses LiteLLM authentication. See LiteLLM's
[custom pass-through endpoint documentation](https://docs.litellm.ai/docs/proxy/pass_through).

## Other Proxies

Use the proxy's native OpenAI route for chat completions. For graph features and
runtime settings discovery, use only a documented raw pass-through route that
can target LGOS. Verify both `models.list()` and `models.retrieve(model)` after
proxy upgrades. Also verify that a chat completion preserves
`metadata.langgraph_runtime_settings` before relying on dynamic settings. Use
direct LGOS discovery when no configurable raw route is available, and treat a
missing extension as a normal fallback to server defaults.
