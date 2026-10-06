# Run The LGOS Server

`lgos serve` runs a graph registry as a complete application: JSON logging,
OpenTelemetry, Langfuse, CORS, interrupt persistence and expiry, and background
Responses. Your project supplies only the graphs.

```bash
uv add "langgraph-openai-serve[server]"
```

Use the [library API](../getting-started.md) instead when your own FastAPI
application must own authentication, custom routes, or middleware.

## Registry Factory

Point the command at a function, in `module:attribute` form, that receives the
server's `ServerResources` and returns a `GraphRegistry`:

```python title="my_app/registry.py"
from langgraph_openai_serve import GraphConfig, GraphFeature, GraphRegistry
from langgraph_openai_serve.server import ServerResources

from my_app.graphs import build_approval_graph, chat_graph


def create_registry(resources: ServerResources) -> GraphRegistry:
    return GraphRegistry(
        graphs={
            "chat": GraphConfig(graph=chat_graph, description="Answer questions."),
            "approval": GraphConfig(
                graph=build_approval_graph(resources.checkpointer),
                description="Pause for approval.",
                features={GraphFeature.INTERRUPTS},
            ),
        },
        run_coordinator=resources.run_coordinator,
    )
```

```bash
lgos serve my_app.registry:create_registry
```

`ServerResources` carries the `checkpointer` and `store` to compile graphs with,
and the `run_coordinator` that leases interrupt runs. Compile only
interrupt-enabled graphs with the checkpointer. When graphs own clients that
must be closed, make the factory an `asynccontextmanager` that yields the
registry; the server keeps it open for the process lifetime.

Like Uvicorn, the command imports from the working directory first, so an
installed package and a local module both work, and `LGOS_REGISTRY` can name the
factory instead of the argument, for example once in a container image.

## Persistence And Background Work

Without `LGOS_POSTGRES_URI`, the checkpointer, store, and run coordinator live
in process memory: paused runs end with the process and only one process can
serve them. Set `LGOS_POSTGRES_URI` for durable interrupts across restarts and
replicas. The server applies LangGraph's checkpoint and store migrations at
startup under an advisory lock, and deletes paused runs older than
`LGOS_INTERRUPT_TTL_MINUTES`.

`LGOS_BACKGROUND` selects the [background engine](background-responses.md):

=== "none"

    Background requests are rejected. This is the default.

=== "memory"

    Runs execute in the server process and are lost when it stops. Use it for
    development and tests.

=== "hatchet"

    The API submits runs to Hatchet, and a separate worker executes them:

    ```bash
    lgos worker my_app.registry:create_registry
    ```

    The worker runs the same factory, so both processes serve one catalog.
    Interrupt runs that move between them need a shared `LGOS_POSTGRES_URI`.

## Logs And Telemetry

Both commands write JSON records to stdout. Records from LGOS, Uvicorn,
Hatchet, and the registry's top-level package appear at `INFO`; other loggers
at `WARNING`. Every record handled during a request carries the
[LGOS request fields](production-logging.md), including records from graph
nodes.

Launch either command with `opentelemetry-instrument` to export traces,
metrics, and logs through the standard `OTEL_*` variables:

```bash
opentelemetry-instrument lgos serve my_app.registry:create_registry
```

Set `LGOS_ENABLE_LANGFUSE=True` and the `LANGFUSE_*` credentials to trace graph
runs in Langfuse.

## Settings

The server reads these from the process environment, alongside the
[package settings](../reference.md#settings), `HATCHET_CLIENT_*`, `LANGFUSE_*`,
and `OTEL_*`. An empty value counts as unset. In `.env` files that Just or
`uv run --env-file` load, single-quote JSON lists, as in
`LGOS_CORS_ORIGINS='["https://app.example.com"]'`; those loaders strip
unquoted double quotes.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LGOS_REGISTRY` | unset | Registry factory used when the command names none |
| `LGOS_HOST`, `LGOS_PORT` | `127.0.0.1`, `8000` | Listen address |
| `LGOS_CORS_ORIGINS` | `[]` | JSON list of allowed browser origins |
| `LGOS_POSTGRES_URI` | unset | PostgreSQL for checkpoints, store, and run leases |
| `LGOS_POSTGRES_POOL_SIZE` | `5` | Connections per process; one serves checkpoints and each running interrupt holds one of the rest |
| `LGOS_INTERRUPT_TTL_MINUTES` | `43200` | Age at which paused runs are deleted |
| `LGOS_INTERRUPT_SWEEP_INTERVAL_MINUTES` | `5` | Interval between expiry sweeps; `0` disables them |
| `LGOS_BACKGROUND` | `none` | `none`, `memory`, or `hatchet` |
| `LGOS_HATCHET_WORKER_SLOTS` | `4` | Concurrent worker runs; with PostgreSQL, below `LGOS_POSTGRES_POOL_SIZE` |

The command has no built-in authentication. Put a gateway in front of it, or
host LGOS yourself as described in [authentication](authentication.md).

## Test The Application

`create_app` builds the same application in tests. Pass explicit settings so
variables from a developer's `.env` cannot reach the test:

```python
from langgraph_openai_serve.server import ServerSettings, create_app

app = create_app(
    create_registry,
    settings=ServerSettings(POSTGRES_URI=None, BACKGROUND="memory"),
)
async with app.router.lifespan_context(app):
    ...  # Call app through httpx2.ASGITransport.
```

The [starter template](../starter-template.md) generates a project that uses
this layout.
