# Run The LGOS Server

`lgos serve` runs a graph registry as a complete application: JSON logging,
OpenTelemetry, Langfuse, CORS, interrupt persistence and expiry, and background
Responses. Your project supplies only the graphs.

```bash
uv add "langgraph-openai-serve[server]"
```

To add authentication, custom routes, or middleware,
[extend the application](#extend-the-application) or use the
[library API](../getting-started.md) in your own FastAPI application.

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

## HTTP Server

`lgos serve` is [Uvicorn's command](https://uvicorn.dev/settings/) with the
registry in place of the application. Every Uvicorn option applies through its
flag or its `UVICORN_*` variable, and so do `WEB_CONCURRENCY` and
`FORWARDED_ALLOW_IPS`. `lgos serve --help` lists them.

```bash
lgos serve my_app.registry:create_registry --port 8080 --reload
UVICORN_HOST=0.0.0.0 UVICORN_WORKERS=4 lgos serve my_app.registry:create_registry
```

LGOS changes two defaults: it writes [structured logs](#logs-and-telemetry),
which `--log-config` replaces, and turns off access logs, which `--access-log`
enables. `--env-file` loads `LGOS_*` settings, but as with `UVICORN_*`
variables, name the registry on the command line or in the process environment.

Each Uvicorn worker is a separate process, like a replica. Workers need a shared
`LGOS_POSTGRES_URI` for interrupts and `LGOS_BACKGROUND=hatchet` for background
Responses, and each opens its own `LGOS_POSTGRES_POOL_SIZE` connections.
`lgos serve` refuses to start more than one worker with
`LGOS_BACKGROUND=memory`.

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
    For container health checks, set Hatchet's
    `HATCHET_CLIENT_WORKER_HEALTHCHECK_ENABLED=true`: the worker then serves
    [`/health`](https://docs.hatchet.run/v1/worker-healthchecks) on port 8001,
    which returns 200 once it is connected to Hatchet.

## Logs And Telemetry

Both commands write JSON records to stdout, or readable lines when stdout is a
terminal. Records from LGOS, Uvicorn, Hatchet, and the registry's top-level
package appear at `INFO`; other loggers at `WARNING`. Every record handled
during a request carries the [LGOS request fields](production-logging.md),
including records from graph nodes.

Launch either command with `opentelemetry-instrument` to export traces,
metrics, and logs through the standard `OTEL_*` variables:

```bash
opentelemetry-instrument lgos serve my_app.registry:create_registry
```

Set `LGOS_ENABLE_LANGFUSE=True` and the `LANGFUSE_*` credentials to trace graph
runs in Langfuse.

## Settings

The server reads these from the process environment, alongside the
[package settings](../reference.md#settings), [`UVICORN_*`](#http-server),
`HATCHET_CLIENT_*`, `LANGFUSE_*`, and `OTEL_*`. An empty value counts as unset.
In `.env` files that Just or `uv run --env-file` load, single-quote JSON lists,
as in `LGOS_CORS_ORIGINS='["https://app.example.com"]'`; those loaders strip
unquoted double quotes.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LGOS_REGISTRY` | unset | Registry factory used when the command names none |
| `LGOS_CORS_ORIGINS` | `[]` | JSON list of allowed browser origins |
| `LGOS_POSTGRES_URI` | unset | PostgreSQL for checkpoints, store, and run leases |
| `LGOS_POSTGRES_POOL_SIZE` | `5` | Connections per process; one serves checkpoints and each running interrupt holds one of the rest |
| `LGOS_INTERRUPT_TTL_MINUTES` | `43200` | Age at which paused runs are deleted |
| `LGOS_INTERRUPT_SWEEP_INTERVAL_MINUTES` | `5` | Interval between expiry sweeps; `0` disables them |
| `LGOS_BACKGROUND` | `none` | `none`, `memory`, or `hatchet` |
| `LGOS_HATCHET_WORKER_SLOTS` | `4` | Concurrent worker runs; with PostgreSQL, `lgos worker` needs it below `LGOS_POSTGRES_POOL_SIZE` |

The command has no built-in authentication. Put a gateway in front of it, or
[extend the application](#extend-the-application).

## Extend The Application

`create_app` returns the FastAPI application that `lgos serve` runs. To add
routes, middleware such as the [API key middleware](authentication.md), or
other FastAPI features, extend it in your own module:

```python title="my_app/asgi.py"
from langgraph_openai_serve.server import create_app

from my_app.auth import APIKeyMiddleware
from my_app.registry import create_registry
from my_app.routes import router

app = create_app(create_registry)
app.add_middleware(APIKeyMiddleware)
app.include_router(router)
```

Serve the module with Uvicorn, or any other ASGI server:

```bash
uvicorn my_app.asgi:app --host 0.0.0.0
```

Uvicorn then applies its own logging configuration. To keep JSON records with
the LGOS request fields, pass a `--log-config` that follows
[production logging](production-logging.md).

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
