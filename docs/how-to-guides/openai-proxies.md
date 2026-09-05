# Use An OpenAI-Compatible Proxy

Direct LGOS is the protocol reference, while the maintained demo UIs enter
through either LiteLLM 1.99.1 or Bifrost v2.0.0. A proxy can either normalize
managed model routes or forward an authenticated OpenAI-compatible pass-through
route. In both cases,
it must carry the native Responses contract without an LGOS-specific response
adapter or plugin. LiteLLM and Bifrost remain deployment choices and are not
part of the package.

## Native Responses Requirements

Configure a standard `/v1` OpenAI base URL and verify the proxy preserves:

- `POST /v1/responses`, including `store: false`, `user`, string-valued
  `metadata`, and function tools;
- typed Responses SSE events, item IDs, output indices, sequence numbers, and
  assistant `phase` values;
- complete `function_call` items and matching `function_call_output` items for
  stateless continuation and interrupt resume;
- standard OpenAI error `type`, `param`, and `code` values;
- Files upload, list, retrieve, content, and delete operations through one file
  namespace independent of graph routing; and
- downstream disconnect propagation to the upstream streaming request.

LGOS does not require the proxy to retain Responses. It rejects
`previous_response_id`, `conversation`, `store: true`, and background mode, so
the client owns the input ledger and replays complete returned items. A proxy
must not silently turn `store: false` into a stored response.

`GET /v1/models` is sufficient for ordinary graph selection. A client that uses
LGOS descriptions, feature discovery, or runtime-settings forms also needs
`GET /v1/models/{model}` and the namespaced `langgraph_openai_serve` property.
Those extensions improve presentation but are not prerequisites for a standard
Responses request.

## Verify The Deployed Version

Test the actual image digest and configuration with the ordinary OpenAI SDK.
At minimum cover non-streaming and streaming text, more than one commentary
item, function continuation, file upload and input, standard errors, and
disconnect cleanup. A proxy that returns valid final text can still be
incompatible if it synthesizes a new stream or drops `phase` and call IDs.

| Path | Current demo result | Intended use |
| --- | --- | --- |
| Direct LGOS | Full maintained contract | Protocol reference and diagnostics |
| LiteLLM 1.99.1 pass-through | Full maintained contract | UI catalog detail and protocol reference |
| LiteLLM 1.99.1 managed routing | Responses, Files, file input, exact-model commentary, and continuation pass; wildcard streams are synthesized, error metadata is rewritten, and some streams raise a background success-log error | LiteLLM-selected UI inference and Files |
| Bifrost v2.0.0 raw pass-through | Full maintained contract | UI catalog detail and protocol reference |
| Bifrost v2.0.0 normalized route | Native Responses fields, Files, file input, commentary `phase`, and continuation pass; model-detail extensions are unavailable and error metadata is rewritten | Bifrost-selected UI inference and Files |

The [demo Docker guide](../demo/docker.md) documents the pinned LiteLLM image,
native-stream fixture, and test command. The [Bifrost guide](../demo/bifrost.md)
records its exact remaining strict expected failures. Do not hide an upstream
failure with a Chat fallback, custom proxy plugin, or LGOS-specific response
field.

## Routing

A gateway may expose provider-qualified model IDs such as
`lgos-a/simple-graph`. Its native Responses route may require a documented
provider selector or may translate the prefix itself. Keep that behavior in
gateway configuration and send the unqualified graph name upstream. Clients
connected directly to LGOS use the registered graph name unchanged.

LiteLLM's documented
[Responses endpoint](https://docs.litellm.ai/docs/response_api) and wildcard
routing can route arbitrary graph names, but 1.99.1 still replaces a wildcard
model's upstream event stream with a final-only synthetic stream. The managed
test surface therefore keeps exact `status-events` entries with
`supports_native_streaming: true`. It does not duplicate the graph catalog.

The maintained UIs use `OPENAI_GATEWAY_TYPE=litellm|bifrost`. With LiteLLM,
their catalog clients read `/models` and `/models/{model}` through authenticated
`/v1/lgos-a` and `/v1/lgos-b` pass-throughs, then retain the matching prefix and
send Responses and Files to LiteLLM's managed `/v1` route. Files requests select
the configured `litellm_proxy` provider. This preserves both LGOS catalogs'
descriptions and capability extensions while keeping LiteLLM's managed routing,
accounting, policy, retry, and fallback features available for inference.
Managed-stream, error-normalization, and background usage-logging limitations
therefore apply when LiteLLM is selected. The demo uses the full official
LiteLLM image because the smaller `litellm-gateway` image's reduced gateway app
removes arbitrary configured routes from its data-plane allowlist.

Bifrost custom providers expose both normalized and raw OpenAI routes. The
pinned v2.0.0 native Responses route preserves `phase`, multiple commentary
items, file input, and function continuation. It still omits LGOS extensions
from normalized model detail and rewrites upstream error metadata.
`/openai_passthrough/v1` passes the complete direct suite when the client
supplies the catalog-discovered provider in `x-model-provider`. The UIs use
that route only for provider-specific catalog detail. Responses use native
`/openai/v1/responses`, and Files use normalized `/v1` with the dedicated
`lgos-files` provider. No plugin or response adapter is required.

## Direct Chat Compatibility

Chat Completions remains available for direct compatibility clients. If such a
client is placed behind a proxy, verify modern tool calls, metadata, usage, and
stream cancellation separately. The optional LGOS Chat client-event extension
(`status`, `progress`, and `artifact`) is not part of the standard Chat schema
and may be removed by a normalizing proxy; use standard Responses commentary,
function calls, and Files for portable maintained UI behavior.

## Request Correlation

Forward a trusted gateway-generated `X-Request-ID` unchanged. At the first
trusted edge, discard or replace a public client's value unless the deployment
explicitly permits caller-controlled correlation data. LGOS validates only the
header's shape; never use it for identity or authorization. See
[Production Logging and Request Correlation](production-logging.md#lgos-responsibilities).
