# Bifrost Gateway

The Compose stack runs two LGOS API containers behind one pinned Bifrost
gateway. Both containers currently use the same demo image because their graph
dependencies do not conflict. A deployment can replace either service with an
independently locked application image without changing the clients.

The configuration at `demo/docker/bifrost/config.json` belongs to the demo, not
the LGOS package.

## Run The Gateway

```bash
cd demo
cp .env.example .env
docker compose -f compose.yaml up --wait bifrost
```

Bifrost exposes the two servers as custom OpenAI providers:

| Provider | Upstream | Example catalog model |
| --- | --- | --- |
| `lgos-a` | `lgos-demo-api-a:8000` | `lgos-a/simple-graph` |
| `lgos-b` | `lgos-demo-api-b:8000` | `lgos-b/simple-graph` |

Use the standard route for the combined catalog and the raw route for
provider-selected LGOS requests:

| Need | Base URL | Routing |
| --- | --- | --- |
| Combined model list | `http://localhost:3000/v1` | Bifrost adds the provider prefix |
| Inference and detailed retrieval | `http://localhost:3000/openai_passthrough/v1` | Unprefixed model plus `x-model-provider` |

The demo clients translate `lgos-a/simple-graph` into model `simple-graph` and
header `x-model-provider: lgos-a`. Bifrost uses the header to select the custom
provider, strips `/openai_passthrough`, and forwards the remaining OpenAI path
and body unchanged. This preserves detailed LGOS model extensions and
extension-only stream chunks.

```python title="Call one model from the combined catalog"
from openai import OpenAI

catalog = OpenAI(base_url="http://localhost:3000/v1", api_key="DUMMY")
client = OpenAI(
    base_url="http://localhost:3000/openai_passthrough/v1",
    api_key="DUMMY",
)

print([model.id for model in catalog.models.list()])

response = client.chat.completions.create(
    model="custom-input-output-context",
    messages=[{"role": "user", "content": "Show me custom schemas."}],
    extra_headers={"x-model-provider": "lgos-b"},
)
print(response.choices[0].message.content)
```

## Configuration Boundary

Each Bifrost custom provider uses `openai` as its base provider and enables only
model listing plus streaming and non-streaming pass-through. The upstream base
URLs omit `/v1`, and private-network access is enabled for the Compose network.

The demo uses `DUMMY` upstream keys because LGOS authentication is not enabled.
Replace each key when its target application enforces authentication.

!!! note "Pass-through tradeoffs"

    Pass-through intentionally skips response normalization. Do not rely on
    Bifrost response additions, model-catalog routing, cross-provider fallbacks,
    or semantic caching on this route. Configured authentication,
    request-based governance, transport retries, and observability remain
    available.

    Usage-based token and cost controls require a standard upstream `usage`
    object. LGOS does not currently emit usage in streaming chunks, so enforce
    request limits independently of streaming token totals.

The Compose defaults already connect Chainlit and the Open WebUI Functions to
these two routes. From the package checkout, run `make test-bifrost` after
starting the gateway to verify the combined catalog, raw detailed retrieval,
and inference through both providers.

See Bifrost's
[custom-provider documentation](https://docs.getbifrost.ai/providers/custom-providers),
[pass-through contract](https://docs.getbifrost.ai/integrations/passthrough),
and the pinned version's
[provider-header routing](https://github.com/maximhq/bifrost/blob/df1644338ad98216cffa78231b6ca19e8e42e8f2/transports/bifrost-http/integrations/utils.go#L573-L578)
for the gateway-owned behavior.
