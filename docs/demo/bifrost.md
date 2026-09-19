# Bifrost Gateway

The Compose stack runs two LGOS API services behind one pinned Bifrost gateway.
Both services use the same demo image and graph set. Their separate provider
identities demonstrate how independently deployed APIs can share one proxy
endpoint. The configuration at
`demo/docker/configs/bifrost/config.json` belongs to the demo, not the LGOS
package.

!!! info "Native Responses preserves phase"

    With `responses` and `responses_stream` enabled for both graph providers,
    the bundled Bifrost gateway's normalized
    `/openai/v1` route preserves the tested `user`, `input_file`,
    function-continuation, final-answer `phase`, and multiple commentary
    `phase` and `store: false` contracts. Two narrower gaps remain: normalized
    model detail does not expose LGOS extensions, and normalized errors replace
    the upstream OpenAI `type`, `param`, and `code`. The raw
    `/openai_passthrough/v1` route
    preserves the successful-request contracts, while governance rejects an
    unknown model before its upstream OpenAI error can pass through.

## Run The Gateway

```bash
cp demo/.env.example demo/.env
just demo/up lgos-bifrost
```

Bifrost exposes each service as a custom provider:

| Provider | Upstream | Example UI model ID |
| --- | --- | --- |
| `openai` | `lgos-demo-api-a:8000` | `background-report-agent` polling lifecycle only |
| `lgos-a` | `lgos-demo-api-a:8000` | `lgos-a/simple-graph` |
| `lgos-b` | `lgos-demo-api-b:8000` | `lgos-b/simple-graph` |
| `lgos-files` | `lgos-files-api:8000` | Files only |

It also exposes the `LGOS PostgreSQL Reports` Virtual MCP at
`http://localhost:3000/mcp/lgos-postgres`. This named bundle selects six tools
from the `lgos_postgres` source client; that client reaches the internal DBHub
service with a separate bearer token. Both the source client and the Virtual
MCP use explicit tool-name lists, so neither opts future tools in
automatically. The UIs connect to the gateway's aggregate `/mcp` endpoint with
the same virtual key used for OpenAI requests; its Virtual MCP grant determines
the tools they discover. The named endpoint remains a gateway-owned interface
for the same bundle. See Bifrost's
[Virtual MCP documentation](https://docs.getbifrost.ai/mcp/virtual-mcps) and
[PostgreSQL Through Native MCP](graphs/mcp-postgres.md).

Use Bifrost's normalized OpenAI endpoint to inspect the shared model catalog:

```python title="Inspect the Bifrost catalog"
import os

from openai import OpenAI

catalog = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key=os.environ["OPENAI_GATEWAY_API_KEY"],
)
model_ids = [
    model.id
    for model in catalog.models.list().data
    if model.owned_by == "langgraph-openai-serve"
]

print(model_ids)
```

Bifrost's catalog owns the provider-qualified IDs. With Bifrost selected, the
UIs split an ID and send its prefix as `x-model-provider`. Inference goes to
native `/openai/v1/responses`; only provider-specific model list and retrieval
go to `/openai_passthrough/v1`, so LGOS descriptions and client settings
survive unchanged. The UI adapter discovers providers from the aggregate
catalog; it does not contain a provider list.

Select the Bifrost values in the shared
[`.env.example`](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/demo/.env.example)
for both demo UIs.

The clients derive the Responses route, catalog-detail route, Files provider,
and model-header routing from that explicit configuration. Host-side commands
must instead receive a URL reachable from the host.

The bundled gateway requires `OPENAI_GATEWAY_API_KEY` on inference, Files,
catalog, and MCP requests. Bifrost loads it as one native virtual key whose
provider policies allow `lgos-a`, `lgos-b`, and `lgos-files`, plus only
`background-report-agent` on the fixed standard `openai` provider. The key is
attached to only the fixed PostgreSQL Virtual MCP. Replace the demo value
before exposing the gateway and retain Bifrost's required `sk-bf-` prefix.

The local dashboard and management API omit administrator authentication so
Compose can perform its startup reconciliation. Before exposing Bifrost beyond
a trusted development host, follow Bifrost's
[authentication guidance](https://docs.getbifrost.ai/deployment-guides/config-json/client#authentication),
configure an encryption key, restrict browser origins, and authenticate that
management request. A virtual key alone protects the data plane, not the
management API.

The dedicated `lgos-files` provider enables Bifrost's normalized `file_upload`,
`file_list`, `file_retrieve`, `file_content`, and `file_delete` operations.
Normalized Files operations are disabled on the graph providers. A client sends
`provider=lgos-files` as a query parameter for Files operations, then sends the
returned native `file_id` to either graph provider. Bifrost does not store the
bytes or replace the ID with an S3 URL.

Bifrost routes Files and Batch operations through the same key pool, so the
provider's key sets `use_for_batch_api: true`. Despite the field name, Files
uploads fail before reaching the upstream service when no key is opted into
that pool.

## Configuration Boundary

The dedicated `openai` provider is a standard Bifrost provider pinned to API A
and allowlists only `background-report-agent`. All Bifrost custom providers use
`openai` as their base provider. `lgos-a` and
`lgos-b` enable model listing, native Responses and streaming, and pass-through
for catalog detail and protocol-reference tests. `lgos-files` enables only Files
operations and targets the standalone S3-backed demo Files service. Upstream
base URLs omit `/v1`, and private-network access is enabled for the Compose
network.

Enable `responses`, `responses_stream`, `responses_retrieve`, and
`responses_cancel` explicitly under each custom graph provider's
`allowed_requests`. Bifrost loads this configuration at startup, so
restart the service after changing it. The graph providers do not enable Chat
Completions or Responses-to-Chat fallback.

## Background Responses

Use the dedicated standard provider through Bifrost's normalized OpenAI route:

```python title="Poll a background report through Bifrost"
import asyncio
import os

from openai import AsyncOpenAI


async def main() -> None:
    client = AsyncOpenAI(
        base_url="http://localhost:3000/openai/v1",
        api_key=os.environ["OPENAI_GATEWAY_API_KEY"],
    )
    response = await client.responses.create(
        model="background-report-agent",
        input="Write a short reliability report.",
        background=True,
        store=True,
    )
    while response.status in {"queued", "in_progress"}:
        await asyncio.sleep(1)
        response = await client.responses.retrieve(response.id)
    print(response.output_text)


asyncio.run(main())
```

No `x-model-provider` header is used. Retrieve and cancel requests contain only
the opaque Response ID, so Bifrost must always send this lifecycle to an LGOS
deployment sharing the same PostgreSQL Response store. The dedicated provider
gives that route a stable target. Generic `lgos-a`/`lgos-b` selection cannot
recover the creation provider from the ID when upstream stores are isolated.

The pinned Bifrost 2.1.1 configuration passes background create, retrieval with
a fresh client, cancellation, and polling-only validation. Run:

```bash
just demo/test-background-gateway --editable
```

The pinned LiteLLM community image is not a fallback for this demo lifecycle;
its managed background create path attempts to load an unavailable enterprise
hook. See [Run Responses In The Background](../how-to-guides/background-responses.md)
for the versioned gateway support matrix and recovery model.

The client header allowlist forwards `traceparent`, `tracestate`, and
`user-agent` through managed Responses requests. This preserves distributed
trace context and the originating UI's identity at LGOS. See the
[OpenTelemetry guide](opentelemetry.md#signal-ownership).

The gateway uses `DUMMY` only for its private upstream connections because LGOS
authentication is not enabled. This is separate from the required client-facing
virtual key. Replace each upstream key when its target application enforces
authentication.

Bifrost's MCP manager initializes only when its native config store is enabled.
The bundled configuration therefore uses an ephemeral SQLite database at
`/tmp/config.db`; the checked-in JSON remains the source of truth on every
restart, matching the demo's otherwise stateless gateway configuration.
The JSON declares the Virtual MCP's key assignment. On a fresh store, Bifrost
2.1.1 reconciles that bundle before its file-defined key, so Compose repeats
the idempotent attachment after startup and keeps the health check red until it
is visible. The configuration also sets `disable_auto_tool_inject=true`, so MCP
tools reach a model only when Chainlit or Open WebUI supplies their schemas;
unrelated calls do not inherit database access.

## Usage Accounting

Usage-based token and cost controls require provider-reported token counts.
LGOS returns aggregated usage on a completed Response, including the terminal
streaming Response. Providers that do not report usage produce no usage object.

Open WebUI and Chainlit use Bifrost native Responses when
`OPENAI_GATEWAY_TYPE=bifrost`, discover provider-qualified models from its
aggregate catalog, and add `x-model-provider` to native inference and
catalog-detail requests. Neither client contains a provider list or uses raw
pass-through for inference.

Run `just demo/test-bifrost --editable` after starting the gateway.
The command requires the native Responses data-plane contracts to pass, records
the normalized model-detail and error-metadata gaps as strict expected failures,
and then requires the raw pass-through OpenAI SDK suite to pass except for its
strict unknown-model governance expectation.

See Bifrost's
[custom-provider documentation](https://docs.getbifrost.ai/providers/custom-providers)
for gateway-owned behavior.
