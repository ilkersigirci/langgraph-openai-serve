# Run Responses In The Background

LGOS can accept an OpenAI Responses request, run its graph in a separate
worker, and let the caller poll the standard Response resource. Hatchet is the
built-in durable backend. It owns queueing, retries, backoff, timeouts,
cancellation, final-failure handling, and recurring maintenance.

Install the supplied persistence and backend adapters:

```bash
uv add "langgraph-openai-serve[postgres,hatchet]"
```

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
            metadata={
                "lgos_run_id": "5cb3d79a-e37f-4f22-9d0d-959e024ad1c8"
            },
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
wins. LGOS records any undelivered native cancellation with that terminal
decision. The Hatchet maintenance task retries it after a transient API failure,
so delivery does not depend on the client cancelling twice.

Creation and retrieval are polling-only. LGOS rejects background streaming and
retrieval cursors; it does not persist or replay SSE events. The server-trusted
`checkpoint_scope` is also the authorization scope for retrieve and cancel.

### Idempotent Creation

Set `metadata.lgos_run_id` to a non-null UUID. A retry with the same scope,
model, UUID, and normalized request returns the same Response. Reusing it with
different content returns `409`. This is strongly recommended around API or
network failures.

LGOS stores the queued Response, submits its stable run reference to Hatchet,
stores Hatchet's workflow ID, and only then acknowledges the POST. If the
submission result is ambiguous, Hatchet's native trigger retries and workflow
idempotency recover the existing workflow ID. If those retries are exhausted,
LGOS makes the stored Response terminal so it no longer consumes admission. An
exact retry with the same `metadata.lgos_run_id` returns that stable failed
Response rather than submitting a second workflow.

## Make The Graph Recoverable

Opt in with a versioned background policy:

```python
from langgraph_openai_serve import BackgroundPolicy, GraphConfig
from langgraph_openai_serve.integrations.postgres import PostgresRunCoordinator


config = GraphConfig(
    graph=lambda: compiled_graph,
    description="Creates a durable report.",
    run_coordinator=PostgresRunCoordinator(
        pool,
        max_concurrent_leases=7,
    ),
    background=BackgroundPolicy(version="report-v1"),
)
```

The version prevents old queued inputs from running against incompatible graph
code. Retry counts and timeouts belong in `HatchetAdapterSettings`, not in the
graph policy.

A background graph must:

- use an asynchronous LangGraph checkpointer with thread deletion, persistent
  for durable deployments;
- configure a run coordinator, cross-process for multi-worker deployments;
- reconstruct its output from checkpointed state;
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
    queue, retries, or timeouts, and restarts lose every Response. Do not deploy
    it.

## Configure PostgreSQL And Hatchet

Create the Response-store schema once during deployment setup:

```python
from langgraph_openai_serve.integrations.background_postgres import (
    PostgresResponseStore,
)

response_store = PostgresResponseStore(pool)
await response_store.setup()
```

Build the same worker and Hatchet workflow definitions in the API and worker
processes:

```python
from datetime import timedelta

from langgraph_openai_serve import (
    BackgroundSettings,
    BackgroundWorker,
    LanggraphOpenaiServe,
)
from langgraph_openai_serve.integrations.hatchet import (
    HatchetAdapterSettings,
    HatchetBackgroundBackend,
    create_hatchet_workflows,
)

response_settings = BackgroundSettings(admission_capacity=1_000)
hatchet_settings = HatchetAdapterSettings(
    retries=3,
    schedule_timeout=timedelta(minutes=30),
    execution_timeout=timedelta(minutes=20),
)
worker = BackgroundWorker(
    graphs=graphs,
    store=response_store,
    settings=response_settings,
)
workflows = create_hatchet_workflows(
    hatchet,
    worker,
    settings=hatchet_settings,
)
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

The separate worker registers both native objects:

```python
native_worker = hatchet.worker(
    name="background-agent-worker",
    slots=8,
    workflows=list(workflows.registrations),
)
native_worker.start()
```

The response workflow uses Hatchet-native TTL idempotency, retries, exponential
backoff, schedule timeout, execution timeout, and an `on_failure_task`. The
failure task only inspects a completed checkpoint or publishes a failed
Response; it never advances an unfinished graph. The maintenance task is a
Hatchet cron that retries persisted native cancellations, deletes terminal
checkpoint lineages, and expires retained Response data. Failed maintenance
rows move behind later work before the next bounded pass, so one bad
checkpoint cannot monopolize a bounded batch.

Supply `HATCHET_CLIENT_TOKEN` and the SDK's standard endpoint/TLS settings to
both processes. `check_hatchet_connection()` provides a read-only deployment
connectivity probe.

## Ownership And Retention

The boundary is deliberately small:

| Owner | Responsibilities |
| --- | --- |
| Hatchet | queue, concurrency, attempts, retry/backoff, schedule and execution timeouts, cancellation, failure task, maintenance cron |
| LGOS Response store | authorization, public status/result, create idempotency, retention, Hatchet workflow ID, undelivered cancellation intent |
| LangGraph checkpointer | recoverable graph progress and final state |
| LGOS worker | request decoding, checkpoint-aware graph invocation, OpenAI Response rendering |

`BackgroundSettings` contains only LGOS-owned limits:

| Field | Default |
| --- | --- |
| `admission_capacity` | 10,000 active Responses |
| `result_retention` | 1 hour for `store=false` |
| `stored_result_retention` | 30 days for `store=true` |
| `idempotency_retention` | 24 hours |
| `maintenance_batch_size` | 100 rows per Hatchet cron run |

Cancellation makes the stored Response terminal before calling Hatchet so a
late graph result cannot overwrite it. It cannot undo an external side effect
that already happened.

## Operate Failure Recovery

Alert on failed `*-finalize-on-failure` tasks and on Responses that remain
`queued` or `in_progress` beyond the configured schedule, execution, and retry
budgets. Once the worker, PostgreSQL, or checkpointer is healthy, replay the
failed workflow with Hatchet's dashboard or its native SDK:

```python
stored = await backend.retrieve(response_id, owner_scope)
if stored is None or stored.workflow_run_id is None:
    raise RuntimeError("Background Response has no recoverable Hatchet run.")
await hatchet.runs.aio_replay(stored.workflow_run_id)
```

The Response store retains `workflow_run_id`, and the replay re-enters the same
checkpoint-aware workflow using the same stable LGOS run ID. It does not need a
second LGOS dispatcher or status reconciler. Keep the failed Hatchet run and its
active Response row until replay publishes a terminal result.

Use the OpenAI cancellation endpoint for application cancellations. Directly
cancelling a run in the Hatchet dashboard bypasses the atomic public Response
transition and is reserved for operator intervention followed by recovery.

## Bring Another Engine

`BackgroundBackend` is the escape hatch for applications that own another
engine. Hatchet remains the only durable built-in adapter. Implement three
methods:

```python
class BackgroundBackend:
    async def create(self, run: NewRun) -> StoredRun: ...
    async def retrieve(self, response_id: str, owner_scope: str) -> StoredRun | None: ...
    async def cancel(
        self,
        response_id: str,
        owner_scope: str,
        response: dict[str, JsonValue],
        *,
        stored: bool,
    ) -> StoredRun | None: ...
```

`stored` is the normalized `store` value from the persisted Response snapshot.

Pass it to `LanggraphOpenaiServe(background=...)`. To reuse LGOS execution,
compose `BackgroundWorker` with `PostgresResponseStore`, or
`InMemoryResponseStore` locally:

| Engine event | LGOS operation |
| --- | --- |
| Submit | `store.accept()`, submit `RunJob`, then `store.record_workflow_run()` |
| Deliver | `worker.execute()`; retry `RetryableJobError` |
| Retries exhausted | `worker.finalize()`; retry `RetryableJobError` |
| Recurring maintenance | `worker.maintain()` |
| Retrieve | `store.get()` |
| Cancel | `store.request_cancellation()`, native cancel, then `store.finish_cancellation()` |

Submission must be durable and idempotent by `run_id`; resolve ambiguous submits
with that ID and record the native run ID before acknowledging create. If native
submission retries are exhausted, publish a terminal failed Response instead of
leaving an active row without a receipt. Cancellation must become terminal in
the store before native cancellation. If native cancellation fails, leave
`cancellation_pending` set and retry `store.claim_cancellations()` rows on the
recurring task.

The engine must supply durable queueing, retries/backoff, schedule and execution
timeouts, cancellation, a retries-exhausted hook, recurring tasks, and replay or
redrive. Test ambiguous submission, completion-versus-cancellation, exhausted
retries, transient cancellation failure, and maintenance before deployment.

## Gateway Compatibility

After a client restart, the opaque Response ID must still route to an LGOS
replica sharing the same PostgreSQL Response store. The demo verifies direct
LGOS routing plus the bundled LiteLLM and Bifrost routes. A gateway must
forward create, retrieve, and cancel while preserving enough routing state for
later lifecycle calls.

See the [OpenAI-compatible proxy guide](openai-proxies.md) and the
[background report demo](../demo/graphs/background-report-agent.md).
