# Use An OpenAI-Compatible Proxy

Use an OpenAI-compatible proxy for inference without an LGOS-specific adapter.
The proxy requirements for model discovery, request metadata, and streaming
disconnects are defined by the
[OpenAI compatibility contract](../explanation/openai-compatibility.md).
This page defines the gateway behavior an LGOS application needs and provides a
LiteLLM pass-through example.

## Requirements

A proxy-facing client continues to use the standard OpenAI API. Configure the
gateway so that it:

- forwards Chat Completions request metadata without changing string values;
- preserves assistant tool calls and matching `tool` messages for interrupt
  resume;
- propagates downstream disconnects to the upstream streaming request;
- exposes model retrieval when clients need LGOS feature or runtime-settings
  discovery; and
- provides a documented byte-transparent route when clients consume LGOS
  extension-only stream chunks.

## Client Event Compatibility

LGOS client events are top-level extensions on otherwise valid
`chat.completion.chunk` objects. Event-only chunks have an empty
`choices[0].delta`, so a proxy that parses and rebuilds the stream may discard
them even while ordinary assistant text continues to work.

| Proxy path | Client events | Assistant text |
| --- | --- | --- |
| Direct LGOS | Preserved | Preserved |
| Schema-normalizing Chat Completions route | Not guaranteed | Preserved |
| Documented raw pass-through route | Preserved when byte-transparent | Preserved |

Use a raw pass-through route for both inference and discovery when a client
requests `metadata.langgraph_stream_events`. A missing event extension is a
safe degradation: the completion remains valid, but event-driven UI is absent.
Verify event count and event/text order with the real client SDK after proxy
upgrades.

## Bifrost

Bifrost provides both a schema-normalizing OpenAI route and a provider
pass-through route. Keep an LGOS provider base URL free of `/v1`; the route adds
the OpenAI subpath. Enable private-network access only when the upstream LGOS
application actually uses a private address.

The repository's [Bifrost demo](../demo/bifrost.md) records the pinned version,
configuration, endpoints, Chainlit settings, and verified event behavior. Those
assets belong to `demo/`, while the requirements above remain applicable to any
LGOS deployment.

## LiteLLM

LiteLLM's normal Chat Completions stream handler does not retain LGOS
event-only chunks. Configure a distinct pass-through prefix that targets LGOS
and includes subpaths for event-enabled inference and detailed discovery:

```yaml
general_settings:
  pass_through_endpoints:
    - path: "/lgos"
      target: "http://lgos-api:8000"
      include_subpath: true
      methods: ["GET", "POST"]
```

Use `https://gateway.example/lgos/v1` as both the inference and discovery base
URL, with unprefixed LGOS model names. LiteLLM's custom pass-through streams
upstream bytes directly; it also bypasses LiteLLM's normal response conversion
and model routing for that endpoint. See LiteLLM's
[custom pass-through documentation](https://docs.litellm.ai/docs/proxy/pass_through),
[raw streaming handler](https://github.com/BerriAI/litellm/blob/dc9297d36f6b9ef0965ff365664c7696bc4131a8/litellm/proxy/pass_through_endpoints/streaming_handler.py#L52-L67),
and
[normal stream filtering](https://github.com/BerriAI/litellm/blob/dc9297d36f6b9ef0965ff365664c7696bc4131a8/litellm/litellm_core_utils/streaming_handler.py#L760-L807).

## Other Proxies

Use a proxy's native OpenAI route for strict standard chat completions. Use only
a documented raw pass-through route for client events and detailed model
extensions. Verify `models.list()`, `models.retrieve(model)`, request metadata,
and event/text stream order after proxy upgrades. When raw pass-through is not
available, connect directly to LGOS or accept standard text streaming without
client events.
