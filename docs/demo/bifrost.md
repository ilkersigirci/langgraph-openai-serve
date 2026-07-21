# Bifrost Gateway

The demo Compose stack places a pinned Bifrost gateway in front of the demo
API. Its configuration at `demo/docker/bifrost/config.json` is a demo asset,
not configuration installed by the LGOS package.

Use the package [proxy guide](../how-to-guides/openai-proxies.md) for the
transport requirements that apply to any gateway.

## Run The Gateway

```bash
cd demo
cp .env.example .env
docker compose -f compose.yaml up --wait bifrost
```

Compose starts the dependent demo API and PostgreSQL services. Choose a route
according to the client behavior you need:

| Need | Base URL | Model |
| --- | --- | --- |
| Standard inference | `http://localhost:8081/v1` | `openai/simple-graph` |
| Event-enabled inference | `http://localhost:8081/openai_passthrough/v1` | `simple-graph` |
| Detailed model discovery | `http://localhost:8081/openai_passthrough/v1` | `simple-graph` |

The standard inference route preserves assistant text but the demo-pinned
Bifrost v1.6.3 discards LGOS event-only chunks. The pass-through route preserves
the `langgraph_openai_serve` extension used by detailed discovery and client
events.

## Configuration Boundary

The demo dedicates Bifrost's built-in `openai` provider to LGOS because the
documented `/openai_passthrough` route selects that provider by default. Its
base URL points to `http://lgos-demo-api:8000` without `/v1`, and private-network
access is enabled for the Compose network.

Replace the demo's `DUMMY` key when the target LGOS application enforces
authentication. Use a separate Bifrost deployment if the same gateway must
also route directly to OpenAI.

## Connect The Demo Chainlit Client

The Chainlit project keeps inference and detailed discovery URLs separate:

=== "Standard stream"

    ```dotenv
    DEMO_CHAINLIT_INFERENCE__BASE_URL=http://localhost:8081/v1
    DEMO_CHAINLIT_INFERENCE__API_KEY=DUMMY
    DEMO_CHAINLIT_INFERENCE_MODEL_PREFIX=openai/
    DEMO_CHAINLIT_DISCOVERY__BASE_URL=http://localhost:8081/openai_passthrough/v1
    DEMO_CHAINLIT_DISCOVERY__API_KEY=DUMMY
    ```

=== "Client events"

    ```dotenv
    DEMO_CHAINLIT_INFERENCE__BASE_URL=http://localhost:8081/openai_passthrough/v1
    DEMO_CHAINLIT_INFERENCE__API_KEY=DUMMY
    DEMO_CHAINLIT_INFERENCE_MODEL_PREFIX=
    DEMO_CHAINLIT_DISCOVERY__BASE_URL=http://localhost:8081/openai_passthrough/v1
    DEMO_CHAINLIT_DISCOVERY__API_KEY=DUMMY
    ```

!!! note "Pass-through tradeoffs"

    Pass-through intentionally skips response normalization. Do not rely on
    Bifrost response additions, model-catalog routing, cross-provider fallbacks,
    or semantic caching on this route. Configured authentication, request-based
    governance, transport retries, and observability remain available.

    Usage-based token and cost controls require a standard upstream `usage`
    object. LGOS does not currently emit usage in streaming chunks, so enforce
    request limits independently of streaming token totals.

From the LGOS package checkout, run `make test-bifrost` after starting the stack
to compare detailed model metadata returned directly and through Bifrost.

See the
[Bifrost pass-through contract](https://docs.getbifrost.ai/integrations/passthrough)
and
[provider configuration](https://docs.getbifrost.ai/quickstart/gateway/provider-configuration)
for gateway-owned behavior. The pinned Bifrost source's
[semantic-cache request list](https://github.com/maximhq/bifrost/blob/df1644338ad98216cffa78231b6ca19e8e42e8f2/plugins/semanticcache/utils.go#L24-L45)
and
[model-catalog resolver](https://github.com/maximhq/bifrost/blob/df1644338ad98216cffa78231b6ca19e8e42e8f2/plugins/modelcatalogresolver/main.go#L58-L66)
show the pass-through exclusions described above.
