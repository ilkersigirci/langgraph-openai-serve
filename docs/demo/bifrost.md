# Bifrost Gateway

The Compose stack runs two LGOS API services behind one pinned Bifrost gateway.
Both services use the same demo image and graph set. Their separate provider
identities demonstrate how independently deployed APIs can share one proxy
endpoint. The configuration at
`demo/docker/configs/bifrost/config.json` belongs to the demo, not the LGOS
package.

!!! info "Native Responses preserves phase"

    Bifrost v2.0.0 contains the fix from
    [PR #3530](https://github.com/maximhq/bifrost/pull/3530). With `responses`
    and `responses_stream` enabled for both graph providers, its normalized
    `/openai/v1` route preserves the tested `user`, `store: false`,
    `input_file`, function-continuation, final-answer `phase`, and multiple
    commentary `phase` contracts. Two narrower gaps remain: normalized model
    detail does not expose LGOS extensions, and normalized errors replace the
    upstream OpenAI `type`, `param`, and `code`. The raw
    `/openai_passthrough/v1` route preserves the complete tested contract.

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
| `lgos-files` | `lgos-files-api:8000` | Files only |

Use Bifrost's normalized OpenAI endpoint to inspect the shared model catalog:

```python title="Inspect the Bifrost catalog"
from openai import OpenAI

catalog = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="DUMMY",
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

Select Bifrost for both demo UIs with one environment value:

```dotenv
OPENAI_GATEWAY_TYPE=bifrost
```

The clients derive the local or Compose URL, Responses route, catalog-detail
route, Files provider, and model-header routing from that selector. Their
optional gateway-root settings remain available for non-demo deployments.

The dedicated `lgos-files` provider enables Bifrost's normalized `file_upload`,
`file_list`, `file_retrieve`, `file_content`, and `file_delete` operations.
Normalized Files operations are disabled on the graph providers. A client sends
`provider=lgos-files` as a query parameter for Files operations, then sends the
returned native `file_id` to either chat provider. Bifrost does not store the
bytes or replace the ID with an S3 URL.

Bifrost routes Files and Batch operations through the same key pool, so the
provider's key sets `use_for_batch_api: true`. Despite the field name, Files
uploads fail before reaching the upstream service when no key is opted into
that pool.

## Configuration Boundary

All Bifrost custom providers use `openai` as their base provider. `lgos-a` and
`lgos-b` enable model listing, Responses, Chat Completions, and their streaming
variants. Chat remains available for direct compatibility demonstrations; the
maintained UIs use Responses exclusively. `lgos-files` enables only Files
operations and targets the standalone S3-backed demo Files service. Upstream
base URLs omit `/v1`, and private-network access is enabled for the Compose
network.

Enable both `responses` and `responses_stream` explicitly under each graph
provider's `allowed_requests`. Bifrost loads this configuration at startup, so
restart the service after changing it. If native Responses is disabled while
Chat remains enabled, Bifrost can fall back to a Responses-to-Chat conversion;
that synthesized stream does not prove native `phase` preservation.

The demo uses `DUMMY` upstream keys because LGOS authentication is not enabled.
Replace each key when its target application enforces authentication.

## Usage Accounting

Usage-based token and cost controls require provider-reported token counts.
LGOS returns aggregated usage on a completed Response, including the terminal
streaming Response. Direct Chat streams use `stream_options.include_usage`.
Providers that do not report usage produce no usage object.

Open WebUI and Chainlit use Bifrost native Responses when
`OPENAI_GATEWAY_TYPE=bifrost`, discover provider-qualified models from its
aggregate catalog, and add `x-model-provider` to native inference and
catalog-detail requests. Neither client contains a provider list or uses raw
pass-through for inference.

From the package checkout, run `make test-bifrost` after starting the gateway.
The command requires the native Responses data-plane contracts to pass, records
only the normalized model-detail and error-metadata gaps as strict expected
failures, and then requires the complete raw pass-through OpenAI SDK suite to
pass.

See Bifrost's
[custom-provider documentation](https://docs.getbifrost.ai/providers/custom-providers)
for gateway-owned behavior.
