# Bifrost Gateway

The Compose stack runs two LGOS API services behind one pinned Bifrost gateway.
Both services use the same demo image and graph set today. They remain separate
so the demo proves that independently deployed APIs, including APIs with
different graph sets later, can share one proxy endpoint. The configuration at
`demo/docker/bifrost/config.json` belongs to the demo, not the LGOS package.

## Run The Gateway

```bash
cd demo
cp .env.example .env
docker compose -f compose.yaml up --wait bifrost
```

Bifrost exposes each service as a custom provider:

| Provider | Upstream | Example UI model ID |
| --- | --- | --- |
| `lgos-a` | `lgos-demo-api-a:8000` | `lgos-a/simple-graph` |
| `lgos-b` | `lgos-demo-api-b:8000` | `lgos-b/simple-graph` |

Configure one OpenAI client with the raw pass-through base URL. Send
`x-model-provider` per request to list either API, retrieve its detailed model,
or run inference:

```python title="Use one pass-through OpenAI client"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/openai_passthrough/v1",
    api_key="DUMMY",
)

providers = ("lgos-a", "lgos-b")
model_ids = []
for provider in providers:
    headers = {"x-model-provider": provider}
    for listed in client.models.list(extra_headers=headers).data:
        model_ids.append(f"{provider}/{listed.id}")

print(model_ids)

selected = "lgos-b/custom-input-output-context"
provider, model_id = selected.split("/", 1)
headers = {"x-model-provider": provider}
model = client.models.retrieve(model_id, extra_headers=headers)
extension = (model.model_extra or {}).get("langgraph_openai_serve")
assert extension is not None

response = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "Show me custom schemas."}],
    extra_headers=headers,
)
print(response.choices[0].message.content)
```

Bifrost uses `x-model-provider` to select `lgos-a` or `lgos-b`, strips
`/openai_passthrough`, and forwards the remaining OpenAI path and body. The
provider-qualified model ID exists only in the UI: the upstream receives its
native graph name and the provider header. The raw route preserves detailed
LGOS model extensions and extension-only stream chunks.

The normalized `/openai/v1` route is deliberately not the demo default. It
returns provider-qualified catalog IDs but does not preserve the LGOS model
extension. To demonstrate this mode, point a demo client at `/openai/v1` and
clear its model-route configuration. The client lists once, sends Bifrost's
returned model IDs back verbatim, and shows **Limited functionality**. It does
not inspect the URL or create a separate catalog client: list, retrieve, and
chat requests all use the same client and base URL.

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

    Usage-based token and cost controls require a standard upstream `usage`
    object. LGOS does not currently emit usage in streaming chunks, so enforce
    request limits independently of streaming token totals.

The Compose defaults connect Chainlit and the Open WebUI Functions to this one
route. Bifrost-specific selection headers live in their deployment
configuration as model-route values, not in URL-dependent client logic. A
different proxy can supply different headers without changing either UI. From
the package checkout, run `make test-bifrost` after starting the gateway to
verify model listing, LGOS metadata, inference, and client events through both
APIs with the same SDK client.

See Bifrost's
[custom-provider documentation](https://docs.getbifrost.ai/providers/custom-providers),
[pass-through contract](https://docs.getbifrost.ai/integrations/passthrough),
and the pinned version's
[provider-header routing](https://github.com/maximhq/bifrost/blob/df1644338ad98216cffa78231b6ca19e8e42e8f2/transports/bifrost-http/integrations/utils.go#L573-L578)
for the gateway-owned behavior.
