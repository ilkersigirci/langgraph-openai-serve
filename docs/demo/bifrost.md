# Bifrost Gateway

The Compose stack runs the demo API and the coding-agent API behind one pinned
Bifrost gateway. The demo API serves the example graphs; the coding-agent
API serves its own graph. Both use the public `lgos/` model namespace while
remaining independently deployed. The configuration at
`demo/docker/configs/bifrost/config.json` belongs to the demo, not the LGOS
package.

!!! info "Native Responses preserves phase"

    With `responses` and `responses_stream` enabled for the graph providers,
    the bundled Bifrost gateway's normalized
    `/openai/v1` route preserves the tested `user`, `input_file`,
    function-continuation, final-answer `phase`, and multiple commentary
    `phase` and `store: false` contracts, plus the upstream error `type` and
    `param`. Normalized model detail does not expose LGOS extensions, and
    governance rejects an unknown provider-qualified model with its own
    `model_blocked` error before LGOS sees the request.

## Run The Gateway

```bash
cp demo/.env.example demo/.env
# Configure the required credentials and storage in demo/.env.
just demo/compose --dev
```

The development command builds this checkout. Use `just demo/compose` with a
published demo image containing the catalog sync. Both commands prepare and
synchronize metadata automatically; no dashboard edits are required.

Bifrost exposes the graph and Files services as custom providers and uses its
built-in `openai` provider for upstream model and vector-store requests:

| Provider | Upstream | Example UI model ID |
| --- | --- | --- |
| `lgos` | `lgos-demo-api:8000` | `lgos/simple-graph` |
| `coding-agent` (routing target) | `lgos-api-coding-agent:8000` | `lgos/coding-agent` |
| `lgos-files` | `lgos-files-api:8000` | Files only |
| `openai` | `api.openai.com` | `openai/gpt-4.1-mini` and `openai/gpt-6-luna`; graph, embedding, and speech models, plus vector-store passthrough |

The coding-agent provider raises Bifrost's request timeout, which bounds a
whole non-streaming response; a
[coding-agent request](graphs/coding-agent.md#streaming-and-state) can wait for
the shared workspace and then run until its own time limit. Streams keep the
default stream-idle timeout: LGOS
[keepalive comments](../explanation/openai-compatibility.md#streaming) reset it
while a request waits or runs a long command.

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

Use Bifrost's native `/v1/models` endpoint to inspect the shared model catalog:

```python title="Inspect the Bifrost catalog"
import json
import os

from openai import OpenAI

catalog = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key=os.environ["OPENAI_GATEWAY_API_KEY"],
)
for model in catalog.models.list().data:
    if model.id.startswith("lgos/"):
        attributes = model.model_extra["additional_attributes"]
        metadata = json.loads(attributes["lgos"])
        print(model.id, metadata["description"], metadata["features"])
```

Bifrost's catalog owns the provider-qualified IDs. With Bifrost selected, the
UIs send a catalog ID such as `lgos/simple-graph` unchanged to native
`/openai/v1/responses`. Bifrost selects the provider from that prefix and
forwards `simple-graph` upstream. The `coding-agent` alias on `lgos` makes
`lgos/coding-agent` discoverable, and a native governance routing rule sends it
to the separate `coding-agent` provider. Model listing is disabled on that
provider, so it does not publish another public ID. Bifrost aliases omit
`owned_by`; the UIs identify graph models by the `lgos/` namespace.

The same catalog response supplies complete
LGOS descriptions, features, and client settings through
`additional_attributes.lgos`, encoded as a JSON string. Bifrost also displays
`additional_attributes.description` in its model editor.

!!! warning "Bifrost ignores `x-model-provider` on Responses"

    A bare `simple-graph` with `x-model-provider: lgos` is spread across every
    provider the virtual key allows for that model, so keep the provider in the
    model ID.

Select the Bifrost values in the shared
[`.env.example`](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/demo/.env.example)
for both demo UIs.

The clients derive the Responses route, native catalog route, and Files provider
from that explicit configuration. Host-side commands
must instead receive a URL reachable from the host.

## Declarative Model Metadata

The demo API's graph registrations remain the source of truth. Bifrost stores
model attributes only on pricing rows, and only its pricing datasheet creates
those rows. Compose therefore runs two jobs from the demo API image:

1. `lgos-bifrost-catalog` runs before Bifrost starts. It reads every graph's
   complete detail from both graph APIs and writes a datasheet of zero-priced
   Responses rows, plus the matching attributes, under
   `demo/docker/volumes/bifrost/catalog/`. Bifrost loads that datasheet through
   `framework.pricing.pricing_url`; with an empty config store, it does not
   start unless the datasheet loads.
2. `lgos-bifrost-sync` runs after Bifrost is healthy. It reloads the datasheet
   and provider model lists, then replaces the `description` and `lgos`
   attributes of every graph row in one `PUT /api/models/catalog` transaction.

`just demo/compose` runs both jobs; open either UI after it finishes. A failed
job stops the command, and a failed graph API keeps the previous datasheet.
Dashboard edits to these attributes last until the next sync.

After changing graph descriptions, features, or settings, run:

```bash
just demo/sync-bifrost --dev
just demo/sync-openwebui
```

Omit `--dev` when using published images. Chainlit rereads the catalog when a
profile is selected. Open WebUI's generated Workspace Models need the second
command to refresh their descriptions and forms. Add independently deployed
APIs to the catalog job with another `--source-url URL`. Graph IDs must be
unique across APIs. Add the backend provider with model listing disabled and
allow it on the virtual key, because Bifrost checks the routed provider. Then
expose its graph IDs as aliases on `lgos` and route them with native
governance rules, following the coding-agent configuration. For replicas of
the same API, keep the same provider and point it at a stable service URL.
A failed source or duplicate graph ID leaves the previous catalog intact.

The gateway's SQLite config store is ephemeral, so a restarted or recreated
gateway loses the attributes. `just demo/compose` republishes them; after
restarting only Bifrost, run `just demo/sync-bifrost`.

!!! note "Graph pricing is zero"

    Graph model calls reach Bifrost as separate `openai/*` requests, so
    its logs record their tokens. The generated datasheet contains only the
    graph rows and omits Bifrost's public prices, so cost reports and monetary
    budgets do not measure that spend. It sets no token limits or model
    capabilities.

Use `/v1/models` for metadata. The `/openai/v1/models` conversion and
normalized model-detail route omit these attributes. The UI adapters decode the
JSON string and keep **Limited functionality** handling for missing or
malformed metadata.

The bundled gateway requires `OPENAI_GATEWAY_API_KEY` on inference, Files,
catalog, speech, and MCP requests. Bifrost loads it as one native virtual key whose
provider policies allow `lgos`, `coding-agent`, and `lgos-files`, plus only the
default graph, embedding, and speech models on `openai`. The graph APIs and
coding agent present the same key for their model calls, so add any other
`DEMO_API_OPENAI_*` model to the virtual key's `openai` allowlist. The key is
attached to only the fixed PostgreSQL Virtual MCP. Replace the demo value
before exposing the gateway and retain Bifrost's required `sk-bf-` prefix.

The local dashboard and management API omit administrator authentication so
Compose can perform its startup reconciliation and catalog sync. Before
exposing Bifrost beyond a trusted development host, follow Bifrost's
[authentication guidance](https://docs.getbifrost.ai/deployment-guides/config-json/client#authentication),
configure an encryption key, restrict browser origins, and authenticate those
management requests. A virtual key alone protects the data plane, not the
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

All Bifrost custom providers use `openai` as their base provider. `lgos`
enables only model listing and native Responses. `lgos-files` enables
only Files operations and targets the standalone S3-backed demo Files service.
Upstream base URLs omit `/v1`, and the providers on the Compose network enable
private-network access.

Enable `responses`, `responses_stream`, `responses_retrieve`, and
`responses_cancel` explicitly under each custom graph provider's
`allowed_requests`. The upstream is Bifrost's built-in `openai` provider with
`OPENAI_UPSTREAM_API_KEY`; graphs and Codex reach it through `/v1`, and the
advanced graph's vector store through `/openai_passthrough/v1`, which targets
that provider by default. A built-in provider accepts every request type, so
passthrough and normalized requests without a `provider` parameter, such as
Files, Batches, and vector stores, reach the upstream account. The virtual
key's model allowlist still applies to requests that name a model. Bifrost resolves no
environment reference in `base_url`, so the upstream URL is literal in
`config.json`. To use another OpenAI-compatible upstream, copy the file to the
gitignored `config.local.json` beside it, change `base_url`, and set
`DEMO_BIFROST_CONFIG=../configs/bifrost/config.local.json`. Bifrost loads this
configuration at startup, so
restart the service after changing it. The graph providers do not enable Chat
Completions or Responses-to-Chat fallback.

The client header allowlist forwards `Idempotency-Key`, `traceparent`, and
`tracestate` through managed Responses requests. The first supports safe
background-create retries; the others preserve distributed trace context. See
the [OpenTelemetry guide](opentelemetry.md#signal-ownership).

## Background Responses

The UIs create background Responses with an `lgos/<graph>` model.
The `lgos` provider targets the demo API, which owns the background engine.
Retrieve and cancel carry no model, so the UIs send the same provider
in Bifrost's `provider` query parameter; without it, Bifrost routes them to its
built-in `openai` provider, the upstream, which does not know LGOS Responses. See
[Background Mock](graphs/background-mock.md) for startup and
[Run Responses In The Background](../how-to-guides/background-responses.md) for
the lifecycle contract.

The gateway uses `DUMMY` only for its private upstream connections because LGOS
authentication is not enabled. This is separate from the required client-facing
virtual key. Replace each upstream key when its target application enforces
authentication.

Bifrost's MCP manager initializes only when its native config store is enabled.
The bundled configuration therefore uses an ephemeral SQLite database at
`/tmp/config.db`; the checked-in JSON remains the source of truth on every
restart, matching the demo's otherwise stateless gateway configuration.
The JSON declares the Virtual MCP's key assignment. On a fresh store, Bifrost
reconciles that bundle before its file-defined key, so Compose repeats
the idempotent attachment after startup and keeps the health check red until it
is visible. The configuration also sets `disable_auto_tool_inject=true`, so MCP
tools reach a model only when Chainlit or Open WebUI supplies their schemas;
unrelated calls do not inherit database access.

## Usage Accounting

Usage-based token and cost controls require provider-reported token counts.
LGOS returns aggregated usage on a completed Response, including the terminal
streaming Response. Providers that do not report usage produce no usage object.
The graphs' own model calls use the same virtual key, so its token counts
include both a graph's reported usage and the upstream calls behind it.

Open WebUI and Chainlit use Bifrost native Responses when
`OPENAI_GATEWAY_TYPE=bifrost`, discover provider-qualified models from its
native `/v1/models` catalog, decode `additional_attributes.lgos`, and send those
IDs unchanged to native inference. Neither client contains a provider list.

Run `just demo/test-bifrost --editable` after starting the gateway.
The command requires the native Responses, Files, MCP, and catalog-metadata
contracts to pass and records the normalized model-detail gap as a strict
expected failure.

See Bifrost's
[custom-provider documentation](https://docs.getbifrost.ai/providers/custom-providers)
for gateway-owned behavior.
