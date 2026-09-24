# Run Responses In The Background

LGOS can accept an OpenAI Responses request, run its graph in a separate
worker, and let the caller poll the standard Response resource. A background
engine runs the work and stores its status and result. Hatchet is the built-in
engine; an in-memory engine serves local trials.

```bash
uv add "langgraph-openai-serve[hatchet]"
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
        )

        while True:
            response = await client.responses.retrieve(accepted.id)
            if response.status not in {"queued", "in_progress"}:
                break
            await asyncio.sleep(1)

        print(response.status, response.output_text)


asyncio.run(main())
```

Use `POST /v1/responses/{response_id}/cancel` to cancel. Cancelling a finished
Response returns it unchanged. Cancellation cannot undo an external side effect
that already happened.

LGOS configures no retries: when a run raises, the Response becomes `failed`;
create a new Response to try again. A request problem found by the worker, such
as a stale interrupt answer, keeps its message; other failures report a generic
message and leave details in the worker logs. Hatchet
[reassigns](https://docs.hatchet.run/v1/faq) a run whose worker dies, and that
run starts over, so graph side effects must tolerate a second execution.

Creation and retrieval are polling-only. LGOS rejects background streaming and
retrieval cursors. The server-trusted `checkpoint_scope` is also the
authorization scope for retrieve and cancel. `metadata.lgos_run_id` is reserved
for interrupt-enabled foreground operations and is rejected on background
requests.

### Idempotent Creation

The OpenAI SDK, LiteLLM, and Bifrost all resend a create after timeouts,
connection failures, or `5xx` responses, and Bifrost fallbacks can resend it to
another LGOS instance. Without a key, each resend starts another run. Send one
new `Idempotency-Key` (for example a UUID) per logical create and reuse it on
every retry:

| Result | Response |
| --- | --- |
| Same owner scope, key, and request | The original Response; no second run starts |
| Same key with different request content | `422` with `code="idempotency_key_reused"` |
| Key missing | A new Response, as in OpenAI |

The engine holds the key, so concurrent retries and retries routed to another
instance still create one run. Hatchet keeps it for 24 hours. LGOS sends only a
SHA-256 digest of the owner scope and key.

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

## Opt A Graph In

Declare the background feature:

```python
from langgraph_openai_serve import GraphConfig, GraphFeature

config = GraphConfig(
    graph=compiled_graph,
    description="Creates a report.",
    features={GraphFeature.BACKGROUND},
)
```

The worker runs the graph like a non-streaming foreground request. Register
every background model in the worker under the same model ID as in the API.

A graph may declare both `GraphFeature.INTERRUPTS` and
`GraphFeature.BACKGROUND`. Its checkpointer and run coordinator must then be
shared by the API and worker processes, for example LangGraph's
`AsyncPostgresSaver` and LGOS's `PostgresRunCoordinator`. A background run that
reaches an interrupt completes with the same `lgos_interrupt` function calls as
a foreground turn. Answer it with `previous_response_id` and
`function_call_output` items, in either mode; see
[interrupt continuation](../explanation/openai-compatibility.md#resuming-an-interrupt):

```python
answer = await client.responses.create(
    model="advanced-graph",
    background=True,
    previous_response_id=paused.id,
    input=[
        {"type": "function_call_output", "call_id": call.call_id, "output": "approve"}
        for call in paused.output
        if call.type == "function_call" and call.name == "lgos_interrupt"
    ],
)
```

A background answer is validated when its run holds the checkpoint lease. When
two answers race, or an answer is stale, that answer's Response fails with the
conflict message and the run continues from the answer that won.

## Try It In One Process

Run jobs as tasks of the application process:

```python
from fastapi import FastAPI
from langgraph_openai_serve import InMemoryBackgroundBackend, LanggraphOpenaiServe

background = InMemoryBackgroundBackend(graphs)
app = FastAPI(lifespan=background.lifespan)
server = LanggraphOpenaiServe(app=app, graphs=graphs, background=background)
server.bind_openai_api()
```

!!! warning
    This backend keeps every Response in memory until the process exits, and a
    restart loses them. Do not deploy it.

## Deploy With Hatchet

The API process submits, reads, and cancels Hatchet runs:

```python
from hatchet_sdk import Hatchet
from langgraph_openai_serve import LanggraphOpenaiServe
from langgraph_openai_serve.integrations.hatchet import (
    HatchetBackgroundBackend,
    create_hatchet_task,
)

hatchet = Hatchet()
backend = HatchetBackgroundBackend(create_hatchet_task(hatchet), hatchet.runs)
server = LanggraphOpenaiServe(
    app=app,
    graphs=graphs,
    checkpoint_scope=authenticated_scope,
    background=backend,
)
server.bind_openai_api()
```

The worker process runs the task. Its Hatchet
[lifespan](https://docs.hatchet.run/reference/python/lifespans) yields the
`GraphRegistry` of background models, so database pools open inside it:

```python
async def lifespan():
    async with open_resources() as resources:
        yield resources.background_graphs


hatchet = Hatchet()
worker = hatchet.worker(
    name="background-agent-worker",
    slots=8,
    workflows=[create_hatchet_task(hatchet)],
    lifespan=lifespan,
)
worker.start()
```

Supply `HATCHET_CLIENT_TOKEN` and the SDK's standard endpoint and TLS settings
to both processes. Hatchet stores each run's request and Response, so its
[data retention](https://docs.hatchet.run/self-hosting/data-retention) decides
how long a Response stays retrievable. A run that waits in the queue longer than
`schedule_timeout` (30 minutes), or runs longer than `execution_timeout` (one
hour), is [cancelled](https://docs.hatchet.run/home/timeouts) by Hatchet:

```python
from datetime import timedelta

task = create_hatchet_task(hatchet, execution_timeout=timedelta(hours=2))
```

Idempotent creation uses
[Hatchet idempotency keys](https://docs.hatchet.run/v1/idempotency); run a
Hatchet engine release that supports them.

## Bring Another Engine

`BackgroundBackend` is the contract behind the polling API:

```python
class BackgroundBackend(Protocol):
    async def submit(self, job: BackgroundJob) -> BackgroundRun: ...
    async def get(self, run_id: str) -> BackgroundRun | None: ...
    async def cancel(self, run_id: str) -> None: ...
```

An engine must:

- give each run a UUID; the public Response ID embeds it;
- start at most one run per `job.idempotency_key` and return that run again for
  a repeated key;
- execute a run with `execute_background_job(job, run_id, graphs)` and keep the
  returned Response JSON as the run's result;
- report `queued`, `in_progress`, `completed`, `failed`, or `cancelled`.

## Gateway Compatibility

Retrieve and cancel must reach an LGOS instance that uses the same engine;
every such instance answers for any Response ID, so gateway routing mistakes do
not lose Responses. Treat Response IDs as opaque:

- LiteLLM encodes the creating deployment in the Response ID it returns and
  routes retrieve and cancel back to it. A replayed create may get a different
  proxy alias for the same LGOS Response.
- Bifrost sends retrieve and cancel to its default provider unless the
  request carries an `x-model-provider` header. Provider-prefixed models make
  create placement predictable and avoid automatic fallbacks to other
  instances.

See the [OpenAI-compatible proxy guide](openai-proxies.md) and the
[background report demo](../demo/graphs/background-report-agent.md).
