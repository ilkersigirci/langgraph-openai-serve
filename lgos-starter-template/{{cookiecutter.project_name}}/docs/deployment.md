# Deployment

## Containers

After `just setup` and configuring `.env`, run:

```bash
just compose-config
just up
```

The image installs locked production dependencies, runs as an unprivileged
user, and starts `lgos serve`. It serves `/v1/health` for health checks and
emits JSON logs to stdout. Compose starts PostgreSQL for durable interrupts and
binds the API port to localhost on the host.

For a managed database, point `DOCKER_POSTGRES_URI` (containers) or
`LGOS_POSTGRES_URI` (local processes) at it, remove the Compose `postgres`
service, and manage backups and retention in your deployment.

## Background execution

Set `LGOS_BACKGROUND=hatchet` and `HATCHET_CLIENT_TOKEN` for an existing Hatchet
tenant; the API and worker must share the token, endpoints, and
`HATCHET_CLIENT_NAMESPACE`. Run the worker with `just worker`, or add
`background` to `COMPOSE_PROFILES`. Until then, containers reject background
requests; `just run` alone keeps them in its process for development. Interrupt
runs that move between the API and the worker need a shared `LGOS_POSTGRES_URI`.
No Hatchet server is deployed here.

The image enables Hatchet's
[worker health check](https://docs.hatchet.run/v1/worker-healthchecks): under
`lgos worker`, port 8001 serves `/health`, which returns 200 once the worker is
connected to Hatchet, and Prometheus `/metrics`. Compose checks the worker
container with `/health`; point your orchestrator's probes at it too.

For an external service, its address must be reachable from inside the
containers. `localhost` inside a container names that container. On a local
Docker host, `host.docker.internal` is available for host services.

## Observability

JSON logging is always enabled, and LGOS adds request and model fields to
records from your graphs too. Set `LGOS_ENABLE_LANGFUSE=True` with the
`LANGFUSE_*` credentials of an existing project to trace graph runs. Langfuse
can record model inputs and outputs; configure masking and retention there.

OpenTelemetry is activated through its launcher:

=== "Local process"

    Set a reachable `OTEL_EXPORTER_OTLP_ENDPOINT`, then:

    ```bash
    just run-otel
    ```

=== "Compose Collector"

    Set `OTEL_COLLECTOR_UPSTREAM` to an external OTLP/HTTP receiver, then:

    ```bash
    docker compose -f compose.yaml -f docker/otel.yaml up --build --wait
    ```

The overlay starts a Collector and launches the API and worker with
`opentelemetry-instrument`. Its configuration removes model payload attributes
before forwarding traces.

## Access and streaming

`lgos serve` has no built-in authentication. Put a gateway with bearer-token
authentication and TLS in front of it, and set explicit browser origins with
`LGOS_CORS_ORIGINS`. To authenticate in Python or add custom routes,
[extend the application](https://ilkersigirci.github.io/langgraph-openai-serve/latest/how-to-guides/server/#extend-the-application).
To isolate checkpoints per tenant, host LGOS in your own FastAPI application
instead; see
[authentication](https://ilkersigirci.github.io/langgraph-openai-serve/latest/how-to-guides/authentication/).

Configure your reverse proxy to pass streaming responses without buffering and
to allow requests for your graph's expected duration. Pin image digests and
review dependency updates according to your release process.
