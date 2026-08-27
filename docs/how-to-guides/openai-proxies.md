# Use An OpenAI-Compatible Proxy

Use an OpenAI-compatible proxy for inference without an LGOS-specific adapter.
The proxy requirements for model discovery, request metadata, and streaming
disconnects are defined by the
[OpenAI compatibility contract](../explanation/openai-compatibility.md).
This page defines the gateway behavior an LGOS application needs and points to
the repository's tested Bifrost example.

## Requirements

A proxy-facing client continues to use the standard OpenAI API. Configure the
gateway so that it:

- forwards Chat Completions request metadata without changing string values;
- preserves assistant tool calls and matching `tool` messages for interrupt
  resume;
- propagates downstream disconnects to the upstream streaming request;
- preserves `langgraph_openai_serve` on model retrieval;
- preserves extension-only stream chunks; and
- exposes metadata-bearing model listing, model retrieval, and Chat Completions
  beneath one pass-through OpenAI base URL per selected provider.

Configure that pass-through base URL in the client. A federating gateway may
also expose a normalized catalog for discovering provider-qualified routing
IDs. Do not treat that catalog as LGOS capability metadata: list or retrieve the
selected provider again through pass-through. If the selected route strips the
required model extension, the UI should keep standard chat available but show
**Limited functionality**.

## Client Event Compatibility

LGOS client events are top-level extensions on otherwise valid
`chat.completion.chunk` objects. Event-only chunks have an empty
`choices[0].delta`, so a proxy that parses and rebuilds the stream may discard
them even while ordinary assistant text continues to work.

| Proxy path | Client events | Assistant text |
| --- | --- | --- |
| Direct LGOS | Preserved | Preserved |
| Schema-normalizing OpenAI route | Not guaranteed | Preserved |
| Documented raw pass-through route | Preserved when byte-transparent | Preserved |

Use the raw pass-through route for metadata-bearing model listing, detailed
model retrieval, request metadata, and chat streams. A normalized federation
catalog may precede those operations but cannot replace them. A missing model
extension is the detectable degraded state; the completion may remain valid,
but the UI labels the model as limited. Verify model metadata, event count, and
event/text order with the real client SDK after proxy upgrades.

## Bifrost

Bifrost provides both a schema-normalizing OpenAI route and a provider
pass-through route. Keep an LGOS provider base URL free of `/v1`; the route adds
the OpenAI subpath. Its normalized catalog can discover provider-qualified IDs;
send the provider prefix as `x-model-provider` when repeating model listing and
making detail or chat requests through pass-through. Enable private-network
access only when the upstream LGOS application actually uses a private address.

The repository's [Bifrost demo](../demo/bifrost.md) records the pinned version,
configuration, endpoints, Chainlit settings, and verified event behavior. Those
assets belong to `demo/`, while the requirements above remain applicable to any
LGOS deployment.

## Request Correlation

Forward a trusted gateway-generated `X-Request-ID` unchanged. At the first
trusted edge, discard or replace a public client's value unless the deployment
explicitly permits caller-controlled correlation data. LGOS validates only the
header's shape; never use it for identity or authorization. The complete
request-ID contract and its relationship to OpenTelemetry are documented in
[Production Logging and Request
Correlation](production-logging.md#lgos-responsibilities).

## Other Proxies

Use one documented raw pass-through route for the complete LGOS OpenAI client.
Verify `models.list()`, `models.retrieve(model)`, request metadata, and
event/text stream order after proxy upgrades. When raw pass-through is not
available, connect directly to LGOS or expose the deployment as limited
functionality in the UI.
