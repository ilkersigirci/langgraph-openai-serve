# Run Responses In The Background

LGOS can accept an OpenAI Responses request, run its graph in a separate
worker, and let the caller stream its events or poll the standard Response
resource. A background engine runs the work and stores its status, events, and
result. Hatchet is the built-in engine; an in-memory engine serves local trials.

```bash
uv add "langgraph-openai-serve[hatchet]"
```

## Client Contract

Create with `background=True` and retain the returned ID. Add `stream=True` to
see the graph's statuses and answer text as they happen; without it, poll until
the Response becomes terminal. Polling shows no progress: `output` stays empty
until then.

=== "Stream"

    ```python
    import asyncio

    from openai import AsyncOpenAI


    async def main() -> None:
        async with AsyncOpenAI(
            base_url="https://gateway.example.com/v1",
            api_key="gateway-key",
        ) as client:
            stream = await client.responses.create(
                model="background-mock",
                input="Summarize the migration risks and next actions.",
                background=True,
                stream=True,
                store=True,
            )
            async for event in stream:
                if event.type == "response.created":
                    response_id = event.response.id
                print(event.sequence_number, event.type)


    asyncio.run(main())
    ```

=== "Poll"

    ```python
    import asyncio

    from openai import AsyncOpenAI


    async def main() -> None:
        async with AsyncOpenAI(
            base_url="https://gateway.example.com/v1",
            api_key="gateway-key",
        ) as client:
            accepted = await client.responses.create(
                model="background-mock",
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

Use `POST /v1/responses/{response_id}/cancel` to cancel. As in OpenAI, cancelling
a cancelled Response returns it again, and cancelling a finished one returns
`400`. Cancellation cannot undo an external side effect that already happened.

LGOS configures no retries: when a run raises, the Response becomes `failed`;
create a new Response to try again. A request problem found by the worker, such
as a stale interrupt answer, keeps its message; other failures report a generic
message and leave details in the worker logs. Hatchet
[reassigns](https://docs.hatchet.run/v1/faq) a run whose worker dies, and that
run starts over, so graph side effects must tolerate a second execution.

The server-trusted `checkpoint_scope` is also the authorization scope for
retrieve, stream, and cancel.

### Following A Stream

The stream opens with `response.created` and `response.queued` as soon as the
engine accepts the job, so the ID is known before a worker picks it up. The
worker's events follow: `response.in_progress`, commentary, answer deltas, and
tool items, exactly as in a foreground stream. While a run is quiet, the stream
sends OpenAI's `keepalive` event, whose `sequence_number` counts keepalives only:
never use it as a cursor.

| How the run ends | How the stream ends |
| --- | --- |
| Completes | `response.completed` or `response.incomplete` |
| Fails | `response.failed`, with the message polling would report |
| Cancelled, including by Hatchet's timeouts | `response.failed`, whose Response has status `cancelled` |

OpenAI defines no cancelled event and may leave such a stream open, which gateways
would report as broken, so LGOS always ends the stream.

Closing the stream or losing the connection never stops the run; only cancel
does. To read on, retrieve with the last `sequence_number` you received:

```python
stream = await client.responses.retrieve(
    response_id, stream=True, starting_after=last_sequence_number
)
```

Omit `starting_after` to replay from the start. Only a Response created with
`stream=True` can be streamed, as in OpenAI; any other returns `400`. **After a
stream breaks, retrieve the Response** and resume while it is active: the stream
only reports progress, and gateways differ in how they report the end of a run; see
[Gateway Compatibility](#gateway-compatibility).

The events are progress and the run's status is the truth. LGOS follows both, so
a stream still ends with the final Response when the events stop early, for
example when the tenant has no durable streams. When a worker dies, the rerun
starts over with new items, and OpenAI's stream cannot tell a client to discard
what it has. The rerun therefore publishes nothing: followers receive no more
events until its final Response, whose text can differ from what they already
received.

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

The worker runs the graph like the matching foreground request: streaming when
the request set `stream=True`, otherwise only for the final Response. Register
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
    model="deep-research",
    background=True,
    previous_response_id=paused.id,
    input=[
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": "Start research",
        }
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

background = InMemoryBackgroundBackend(registry)
app = FastAPI(lifespan=background.lifespan)
server = LanggraphOpenaiServe(registry=registry, app=app, background=background)
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
backend = HatchetBackgroundBackend(
    create_hatchet_task(hatchet), hatchet.runs, hatchet.streams
)
server = LanggraphOpenaiServe(
    registry=registry,
    app=app,
    checkpoint_scope=authenticated_scope,
    background=backend,
)
server.bind_openai_api()
```

The worker process runs the task. Its Hatchet
[lifespan](https://docs.hatchet.run/reference/python/lifespans) yields a
`GraphRegistry` holding the background models and, when any of them declares
interrupts, the shared run coordinator, so database pools open inside it:

```python
async def lifespan():
    async with open_resources() as resources:
        yield resources.background_registry


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

### Enable Durable Streams

Streaming stores each event in a
[Hatchet durable stream](https://docs.hatchet.run/reference/python/feature-clients/streams)
topic named after the run. This needs a Hatchet engine release with durable
streams. A self-hosted engine leaves them off for each tenant until you turn them
on:

=== "New tenants"

    Set this on the engine. It applies when a tenant is created:

    ```bash
    SERVER_DEFAULT_TENANT_ENTITLEMENT_DURABLE_STREAMS=true
    ```

=== "An existing tenant"

    Update the engine's database, then restart the engine, which caches the
    value for up to five minutes. `default` is the slug of the tenant Hatchet
    creates by itself:

    ```sql
    INSERT INTO tenant_entitlement (tenant_id, durable_streams)
    SELECT id, true FROM "Tenant" WHERE slug = 'default'
    ON CONFLICT (tenant_id) DO UPDATE SET durable_streams = true;
    ```

Hatchet does not document this setting; it comes from the engine's source. Without
it, runs still complete: the worker and API log `background.events_not_published`
and `background.events_not_read`, and a stream delivers only the accepting events
and the final Response. The same happens once the engine refuses an event: the
first failure ends publishing for that run, and a stream never delivers an event
past a gap.

Each event of a streaming job is one row in Hatchet's database, kept for the
tenant's data retention; a Response created without `stream=True` writes none. Text
deltas share publish requests, about ten per second per run, because the engine
limits requests per API token (`SERVER_GRPC_RATE_LIMIT`, 1000 per second by
default). Its default limits are 100,000 topics and 10 million messages per 24
hours; see its
[configuration options](https://docs.hatchet.run/self-hosting/configuration-options).

## Bring Another Engine

`BackgroundBackend` is the contract behind the background API:

```python
class BackgroundBackend(Protocol):
    async def submit(self, job: BackgroundJob) -> BackgroundRun: ...
    async def get(self, run_id: str) -> BackgroundRun | None: ...
    async def cancel(self, run_id: str) -> None: ...
    def events(self, run_id: str) -> AsyncGenerator[str, None]: ...
```

An engine must:

- give each run a UUID; the public Response ID embeds it;
- start at most one run per `job.idempotency_key` and return that run again for
  a repeated key;
- execute a run with
  `execute_background_job(job, engine_run_id, registry, publish)`, passing the
  engine's run ID and, for a job created with `stream=true`, a `publish` that stores
  each event, and keep the returned Response JSON as the run's result;
- yield the stored events, encoded as JSON, from `events`, in order, from the first,
  and keep following new ones. `events` may end early when events cannot be read,
  and must never mix the events of two attempts;
- report `queued`, `in_progress`, `completed`, `failed`, or `cancelled`.

## Gateway Compatibility

Retrieve and cancel must reach an LGOS instance that uses the same engine;
every such instance answers for any Response ID, so gateway routing mistakes do
not lose Responses. Treat Response IDs as opaque:

- LiteLLM encodes the creating deployment in the Response ID it returns and
  routes retrieve and cancel back to it. A replayed create may get a different
  proxy alias for the same LGOS Response.
- Bifrost sends retrieve and cancel to its default `openai` provider unless
  the request carries a `provider` query parameter. Send the create model's
  provider prefix there. Provider-prefixed models make create placement
  predictable and avoid automatic fallbacks to other instances; Bifrost does
  not read `x-model-provider` on Responses routes.

Both gateways forward a background stream's events live, with these differences:

| Behavior | LiteLLM | Bifrost |
| --- | --- | --- |
| Resume with `stream` and `starting_after` | Works with the `homeserver-litellm` image; upstream LiteLLM drops both parameters, so the stream delivers no events and no error | Works |
| A cancelled run's `response.failed` | Arrives with status `failed`; retrieve reports `cancelled` | SDK clients get `APIError`; retrieve reports `cancelled` |
| A failed run's `response.failed` | Arrives as sent | SDK clients get `APIError` with its message |
| The opening `response.created` and `response.queued` | Arrive at once with the `homeserver-litellm` image; upstream LiteLLM holds them until the run's first output event | Held until the run's first output text |

Until the opening events arrive, or LGOS's first keepalive 15 seconds into a
quiet wait, a client behind Bifrost or upstream LiteLLM has no Response ID, so
it can neither show the queued state nor cancel the run. A graph that publishes
a status as soon as it starts keeps that wait to the time a worker takes to pick
up the run.

Both forward the `keepalive` event, so a client's read timeout only has to outlast
its 15-second interval. Treat any stream that breaks the same way: retrieve the
Response, and while it is active resume from the last `sequence_number`. The
maintained UIs do exactly this.

See the [OpenAI-compatible proxy guide](openai-proxies.md) and the
[background mock demo](../demo/graphs/background-mock.md).
