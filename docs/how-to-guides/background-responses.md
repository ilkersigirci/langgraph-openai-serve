# Run Responses In The Background

LGOS can accept an OpenAI Responses request, run its graph in a separate
worker, and let the caller poll the standard Response resource. Hatchet is the
built-in durable backend. It owns queueing, retries, backoff, timeouts,
cancellation, final-failure handling, and recurring maintenance.

Install the supplied persistence and backend adapters:

```bash
uv add "langgraph-openai-serve[postgres,hatchet]"
```

The Response store, checkpointer, and run coordinator are each replaceable; see
[Configure Persistence And Coordination](infrastructure.md).

## Client Contract

Create with `background=True`, retain the returned ID, and poll until the
Response becomes terminal:

```python title="Create and poll"
import asyncio

from openai import AsyncOpenAI


async def main() -> None:
    async with AsyncOpenAI(
        base_url="https://gateway.example.com/v1",
        api_key="gateway-key",
    ) as client:
        accepted = await client.responses.create(
            model="background-report-agent",
            input="Summarize the migration risks and next actions.",
            background=True,
            store=True,
        )

        while True:
            response = await client.responses.retrieve(accepted.id)
            if response.status not in {"queued", "in_progress"}:
                break
            await asyncio.sleep(1)

        print(response.status, response.output_text)


asyncio.run(main())
```

Use `POST /v1/responses/{response_id}/cancel` to cancel. Completion and
cancellation use one atomic terminal transition, so whichever commits first
wins, and a late graph result cannot overwrite a cancellation. Cancelling a
terminal Response returns it unchanged. Cancellation cannot undo an external
side effect that already happened.

Creation and retrieval are polling-only. LGOS rejects background streaming and
retrieval cursors; it does not persist or replay SSE events. The server-trusted
`checkpoint_scope` is also the authorization scope for retrieve and cancel.
`metadata.lgos_run_id` is reserved for interrupt-enabled foreground operations
and is rejected on background requests.

### Idempotent Creation

The OpenAI SDK, LiteLLM, and Bifrost all resend a create after timeouts,
connection failures, or `5xx` responses, and Bifrost fallbacks can resend it to
another LGOS instance. Without a key, each resend starts another run. Send one
new `Idempotency-Key` (for example a UUID) per logical create and reuse it on
every retry:

| Result | Response |
| --- | --- |
| Same owner scope, model, key, and request | The original Response; no second run starts |
| Same key with different request content | `422` with `code="idempotency_key_reused"` |
| Key missing | A new Response, as in OpenAI |

The key is unique in the shared Response store, so concurrent retries and
retries routed to another instance still create one run. It lives as long as
its Response: `result_retention` or `stored_result_retention` after the run
ends. LGOS stores only a SHA-256 digest of the owner scope, model, and key.

Each gateway needs the key in its own transport. LGOS receives the same header
in both cases:

=== "Bifrost"

    Add `idempotency-key` to `client.header_filter_config.allowlist`, then send
    the header:

    ```python
    extra_headers = {"Idempotency-Key": key}
    ```

=== "LiteLLM"

    LiteLLM drops a client `Idempotency-Key` header but forwards per-request
    upstream headers from the body:

    ```python
    extra_body = {"extra_headers": {"Idempotency-Key": key}}
    ```

## Make The Graph Recoverable

Opt in with a background version:

```python
from langgraph_openai_serve import GraphConfig
from langgraph_openai_serve.integrations.coordination.postgres import (
    PostgresRunCoordinator,
)


config = GraphConfig(
    graph=lambda: compiled_graph,
    description="Creates a durable report.",
    run_coordinator=PostgresRunCoordinator(
        pool,
        max_concurrent_leases=7,
    ),
    background_version="report-v1",
)
```

The version prevents old queued inputs from running against incompatible graph
code. Retry counts and timeouts belong in `HatchetAdapterSettings`, not in the
graph configuration.

A background graph must:

- use an asynchronous LangGraph checkpointer with thread deletion, persistent
  for durable deployments;
- configure a run coordinator, cross-process for multi-worker deployments;
- reconstruct its output from checkpointed state;
- keep `output_to_message` deterministic and side-effect free;
- support `durability="sync"`; and
- avoid LangGraph interrupts, which are mutually exclusive with background
  execution.

Hatchet retries a failed task. On each attempt, LGOS inspects the checkpoint,
applies initial input only when no checkpoint exists, and resumes unfinished
work otherwise. A completed node normally is not rerun, but unfinished node
side effects still need application-level idempotency.

## Try It In One Process

For local trials, pair LangGraph's `InMemorySaver` with
`InMemoryRunCoordinator`, then run the backend in the application lifespan:

```python
from fastapi import FastAPI
from langgraph_openai_serve import InMemoryBackgroundBackend, LanggraphOpenaiServe

background = InMemoryBackgroundBackend(graphs=graphs)
app = FastAPI(lifespan=background.lifespan)
server = LanggraphOpenaiServe(app=app, graphs=graphs, background=background)
server.bind_openai_api()
```

!!! warning
    This backend keeps all state and tasks in one process. It has no durable
    queue, retries, or timeouts, and restarts lose every Response. A maintenance
    loop in its lifespan deletes checkpoints and expired Responses every
    `maintenance_interval` (one minute by default). Do not deploy it.

## Configure PostgreSQL And Hatchet

Create the Response-store schema during deployment setup:

```python
from langgraph_openai_serve.integrations.background.postgres import (
    PostgresResponseStore,
)

response_store = PostgresResponseStore(pool)
await response_store.setup()
```

`setup()` is safe to repeat or call concurrently. Run it before starting API
or background-worker processes rather than from every worker.

Register the same Hatchet workflow definitions in the API and worker
processes. The API process only triggers and cancels them:

```python
from datetime import timedelta

from langgraph_openai_serve import BackgroundSettings, LanggraphOpenaiServe
from langgraph_openai_serve.integrations.background.hatchet import (
    HatchetAdapterSettings,
    HatchetBackgroundBackend,
    create_hatchet_workflows,
)

response_settings = BackgroundSettings(stored_result_retention=timedelta(days=7))
hatchet_settings = HatchetAdapterSettings(
    retries=3,
    schedule_timeout=timedelta(minutes=30),
    execution_timeout=timedelta(minutes=20),
)
workflows = create_hatchet_workflows(hatchet, settings=hatchet_settings)
backend = HatchetBackgroundBackend(
    workflow=workflows.response,
    runs=hatchet.runs,
    store=response_store,
    settings=response_settings,
)

server = LanggraphOpenaiServe(
    app=app,
    graphs=graphs,
    checkpoint_scope=authenticated_scope,
    background=backend,
)
server.bind_openai_api()
```

The worker process runs them. Hatchet tasks use the `BackgroundWorker` that
the native worker
[lifespan](https://docs.hatchet.run/reference/python/lifespans) yields, so
database pools open inside that lifespan:

```python
from langgraph_openai_serve import BackgroundWorker


async def lifespan():
    async with open_resources() as resources:
        yield BackgroundWorker(
            graphs=resources.graphs,
            store=resources.response_store,
            settings=response_settings,
        )


native_worker = hatchet.worker(
    name="background-agent-worker",
    slots=8,
    workflows=list(workflows.registrations),
    lifespan=lifespan,
)
native_worker.start()
```

The response workflow uses Hatchet-native retries, backoff, timeouts, and an
`on_failure_task` that only publishes checkpointed output or a failure; it never
advances an unfinished graph. A maintenance cron deletes terminal checkpoints
and expired Responses. LGOS triggers the workflow with the Response ID as input
and as `lgos_response_id` run metadata, which cancellation and replay use to
find the run.

!!! note "Failed submissions are resubmitted"
    LGOS has no transactional outbox. If the trigger fails, or the API process
    dies after persisting a Response but before triggering Hatchet, the Response
    stays `queued`. The maintenance cron resubmits Responses queued longer than
    `BackgroundSettings.resubmit_after` (one minute). The workflow's Hatchet
    idempotency key is the Response ID, so resubmitting a run Hatchet already
    has is a no-op for `HatchetAdapterSettings.max_queue_time`, the longest time
    Hatchet can keep a submitted run queued.

Supply `HATCHET_CLIENT_TOKEN` and the SDK's standard endpoint/TLS settings to
both processes.

## Ownership And Retention

The boundary is deliberately small:

| Owner | Responsibilities |
| --- | --- |
| Hatchet | queue, concurrency, attempts, retry/backoff, schedule and execution timeouts, cancellation, failure task, maintenance cron |
| LGOS Response store | authorization, public status/result, retention, pending checkpoint cleanup |
| LangGraph checkpointer | recoverable graph progress and final state |
| LGOS worker | request decoding, checkpoint-aware graph invocation, OpenAI Response rendering |

`BackgroundSettings` contains only LGOS-owned limits:

| Field | Default |
| --- | --- |
| `result_retention` | 10 minutes for `store=false`, like OpenAI |
| `stored_result_retention` | 30 days for `store=true` |
| `resubmit_after` | 1 minute queued before maintenance resubmits a run |
| `maintenance_batch_size` | 100 rows per maintenance run |

## Operate Failure Recovery

Alert on failed `*-finalize-on-failure` tasks and on Responses that remain
`queued` or `in_progress` beyond the configured schedule, execution, and retry
budgets. Once the worker, PostgreSQL, or checkpointer is healthy, replay the
failed workflow with Hatchet's dashboard or its native SDK:

```python
from hatchet_sdk.features.runs import BulkCancelReplayOpts, RunFilter

await hatchet.runs.aio_bulk_replay(
    BulkCancelReplayOpts(
        filters=RunFilter(
            since=created_at,
            additional_metadata={"lgos_response_id": response_id},
        )
    )
)
```

The replay re-enters the same checkpoint-aware workflow with the same Response
ID. Keep the failed Hatchet run and its active Response row until replay
publishes a terminal result.

Use the OpenAI cancellation endpoint for application cancellations. Directly
cancelling a run in the Hatchet dashboard bypasses the atomic public Response
transition and is reserved for operator intervention followed by recovery.

## Bring Another Engine

`BackgroundBackend` is the escape hatch for applications that own another
engine. Hatchet remains the only durable built-in adapter. A backend only starts
and stops work; LGOS owns every Response state change:

```python
class BackgroundBackend:
    store: ResponseStore
    settings: BackgroundSettings

    async def submit(self, run: StoredRun) -> None: ...
    async def stop(self, run: StoredRun) -> None: ...
```

LGOS persists the run, then calls `submit`; an idempotent replay is never
submitted again. If `submit` raises, the run stays `queued`, and
`worker.maintain(resubmit=backend.submit)` submits it again, so `submit` must
ignore a run the engine already has. On cancel, LGOS commits the cancelled Response first and calls `stop`
only when that cancellation won; a failed `stop` is logged, not returned.

Pass the backend to `LanggraphOpenaiServe(background=...)`. To reuse LGOS
execution, compose `BackgroundWorker` with any implementation of
`ResponseStore`. `PostgresResponseStore` is the supplied durable adapter;
`InMemoryResponseStore` is for local development:

| Engine event | LGOS operation |
| --- | --- |
| Submit | Enqueue the Response ID given to `backend.submit()` |
| Deliver | `worker.execute()`; retry any exception it raises |
| Retries exhausted | `worker.finalize()`; retry any exception it raises |
| Recurring maintenance | `worker.maintain(resubmit=...)` |
| Cancel | Cancel natively in `backend.stop()` |

The engine must supply durable queueing, retries/backoff, schedule and execution
timeouts, cancellation, a retries-exhausted hook, recurring tasks, and replay or
redrive. Test completion-versus-cancellation, exhausted retries, and
maintenance before deployment.

## Gateway Compatibility

Retrieve and cancel must reach an LGOS instance that shares the Response store;
every such instance answers for any Response ID, so gateway routing mistakes do
not lose Responses. Treat Response IDs as opaque:

- LiteLLM encodes the creating deployment in the Response ID it returns and
  routes retrieve and cancel back to it. A replayed create may get a different
  proxy alias for the same LGOS Response.
- Bifrost sends retrieve and cancel to its default provider unless the client
  passes `?provider=`. Provider-prefixed models make create placement
  predictable and avoid automatic fallbacks to other instances.

See the [OpenAI-compatible proxy guide](openai-proxies.md) and the
[background report demo](../demo/graphs/background-report-agent.md).
