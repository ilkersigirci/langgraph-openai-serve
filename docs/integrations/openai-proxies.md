# OpenAI-Compatible Proxies

Use an OpenAI-compatible proxy for inference without an LGOS-specific adapter.
The proxy requirements for model discovery, request metadata, and streaming
disconnects are defined by the
[OpenAI compatibility contract](../explanation/openai-compatibility.md).
This page applies that contract to Bifrost and LiteLLM.

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

The Compose stack mounts the repository's
[Bifrost configuration](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docker/bifrost/config.json).
It dedicates Bifrost's built-in `openai` provider to LGOS because the documented
`/openai_passthrough` route selects that provider by default.

Set `allow_private_network` only when the LGOS URL resolves to a private network.
Keep the provider base URL free of `/v1`, and replace the demo's `DUMMY` key when
LGOS requires authentication. See [Docker](../how-to-guides/docker.md) for
startup and endpoint URLs.

Choose the route according to whether the client needs LGOS events:

| Need | Base URL | Model |
| --- | --- | --- |
| Standard inference | `https://gateway.example/v1` | `openai/simple-graph` |
| Event-enabled inference | `https://gateway.example/openai_passthrough/v1` | `simple-graph` |
| Detailed discovery | `https://gateway.example/openai_passthrough/v1` | `simple-graph` |

The repository-pinned Bifrost v1.6.3 preserves assistant text but discards
LGOS event-only chunks on its standard inference route. Event-enabled clients
must send the usual `metadata={"langgraph_stream_events": "v1"}` opt-in through
the pass-through route.

The pass-through returns the upstream body without route-level conversion, so
the `langgraph_openai_serve` extension is preserved. Bifrost still runs its
core and plugin pipelines, including configured logging and observability.

The [Chainlit integration](chainlit.md) keeps inference and discovery URLs
explicit:

=== "Standard stream"

    ```dotenv
    DEMO_CHAINLIT_INFERENCE__BASE_URL=https://gateway.example/v1
    DEMO_CHAINLIT_INFERENCE__API_KEY=BIFROST_API_KEY
    DEMO_CHAINLIT_INFERENCE_MODEL_PREFIX=openai/
    DEMO_CHAINLIT_DISCOVERY__BASE_URL=https://gateway.example/openai_passthrough/v1
    DEMO_CHAINLIT_DISCOVERY__API_KEY=BIFROST_API_KEY
    ```

=== "Client events"

    ```dotenv
    DEMO_CHAINLIT_INFERENCE__BASE_URL=https://gateway.example/openai_passthrough/v1
    DEMO_CHAINLIT_INFERENCE__API_KEY=BIFROST_API_KEY
    DEMO_CHAINLIT_INFERENCE_MODEL_PREFIX=
    DEMO_CHAINLIT_DISCOVERY__BASE_URL=https://gateway.example/openai_passthrough/v1
    DEMO_CHAINLIT_DISCOVERY__API_KEY=BIFROST_API_KEY
    ```

!!! note "Bifrost feature boundary"

    Pass-through intentionally skips response normalization. Do not rely on
    Bifrost response additions, model-catalog routing, cross-provider
    fallbacks, or semantic caching on this route. Authentication,
    request-based governance, transport retries, and observability remain
    available when configured.

    Usage-based token and cost controls require a standard upstream `usage`
    object. LGOS does not currently emit usage in streaming chunks, so enforce
    request limits independently of streaming token totals.

The configured `openai` provider is dedicated to LGOS. Use a separate Bifrost
deployment when the same gateway must also route directly to OpenAI.

See the [Bifrost pass-through contract](https://docs.getbifrost.ai/integrations/passthrough)
and [provider configuration](https://docs.getbifrost.ai/quickstart/gateway/provider-configuration).
Bifrost's current
[semantic-cache request list](https://github.com/maximhq/bifrost/blob/df1644338ad98216cffa78231b6ca19e8e42e8f2/plugins/semanticcache/utils.go#L24-L45)
and
[model-catalog resolver](https://github.com/maximhq/bifrost/blob/df1644338ad98216cffa78231b6ca19e8e42e8f2/plugins/modelcatalogresolver/main.go#L58-L66)
show the pass-through exclusions.

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
