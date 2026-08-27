# Bifrost Gateway

The Compose stack runs two LGOS API services behind one pinned Bifrost gateway.
Both services use the same demo image and graph set. Their separate provider
identities demonstrate how independently deployed APIs can share one proxy
endpoint. The configuration at
`demo/docker/configs/bifrost/config.json` belongs to the demo, not the LGOS
package.

## Run The Gateway

```bash
cd demo
cp .env.example .env
docker compose -f docker/compose/demo.yml up --wait lgos-bifrost
```

Bifrost exposes each service as a custom provider:

| Provider | Upstream | Example UI model ID |
| --- | --- | --- |
| `lgos-a` | `lgos-demo-api-a:8000` | `lgos-a/simple-graph` |
| `lgos-b` | `lgos-demo-api-b:8000` | `lgos-b/simple-graph` |

Use Bifrost's normalized OpenAI endpoint as the shared model catalog and its raw
pass-through endpoint for detailed model metadata and inference:

```python title="Use Bifrost catalog and pass-through clients"
from openai import OpenAI

catalog = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="DUMMY",
)
passthrough = OpenAI(
    base_url="http://localhost:3000/openai_passthrough/v1",
    api_key="DUMMY",
)

model_ids = [
    model.id
    for model in catalog.models.list().data
    if model.owned_by == "langgraph-openai-serve"
]

print(model_ids)

selected = "lgos-b/custom-input-output-context"
provider, model_id = selected.split("/", 1)
headers = {"x-model-provider": provider}
model = passthrough.models.retrieve(model_id, extra_headers=headers)
extension = (model.model_extra or {}).get("langgraph_openai_serve")
assert extension is not None

response = passthrough.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "Show me custom schemas."}],
    extra_headers=headers,
)
print(response.choices[0].message.content)
```

Bifrost's catalog owns the provider-qualified IDs. For pass-through requests,
the client splits that ID, sends the prefix as `x-model-provider`, and sends the
remaining native graph name as the OpenAI model. Bifrost strips
`/openai_passthrough` and forwards the remaining path and body. The raw route
preserves detailed LGOS model extensions and extension-only stream chunks.

The normalized `/v1` route returns provider-qualified catalog IDs but does not
preserve the LGOS model extension. Both dynamic UI integrations therefore use
it only for discovery. Metadata-bearing model listing, detailed retrieval, and
chat use pass-through.

## Configuration Boundary

Both Bifrost custom providers use `openai` as their base provider and enable
model listing, Chat Completions, and streaming and non-streaming pass-through.
Chat Completions keep the normalized route useful for the limited-functionality
demonstration; pass-through preserves the complete LGOS protocol. Their
upstream base URLs omit `/v1`, and private-network access is enabled for the
Compose network.

The demo uses `DUMMY` upstream keys because LGOS authentication is not enabled.
Replace each key when its target application enforces authentication.

!!! note "Pass-through tradeoffs"

    Pass-through intentionally skips response normalization. Do not rely on
    Bifrost response additions, model-catalog routing, cross-provider
    fallbacks, or semantic caching on this route. Configured authentication,
    request-based governance, transport retries, and observability remain
    available.

    Usage-based token and cost controls require provider-reported token counts.
    LGOS returns aggregated usage in complete responses and, when requested
    with `stream_options.include_usage`, in the standard final streaming usage
    chunk. Providers that do not report usage produce no usage object.

Open WebUI's dynamic integration and Chainlit discover provider-qualified LGOS
models from Bifrost's catalog, then add `x-model-provider` only for pass-through
requests. Neither contains a provider list. The fixed-model Open WebUI example
uses pass-through without catalog discovery. Chainlit omits its optional catalog
URL when it runs directly against one LGOS API. From the package checkout, run
`make test-bifrost` after starting the gateway to verify model listing, LGOS
metadata, inference, and client events through both APIs with the OpenAI SDK.

See Bifrost's
[custom-provider documentation](https://docs.getbifrost.ai/providers/custom-providers),
and [pass-through contract](https://docs.getbifrost.ai/integrations/passthrough)
for gateway-owned behavior.
