# Configure Persistence And Coordination

LGOS is embedded in an application that owns its graphs and infrastructure.
Choose and construct components in application code, then inject them into
LGOS. Installing an optional integration does not create connections, select a
backend, or initialize a schema.

## Choose Components By Responsibility

| Component | Owns | Configuration point |
| --- | --- | --- |
| LangGraph checkpointer | Recoverable execution state for one thread | `builder.compile(checkpointer=...)` |
| LangGraph Store | Application data and memory across threads | `builder.compile(store=...)` |
| LGOS `ResponseStore` | Background response snapshots, owner scope, retention, and checkpoint-cleanup intent | `BackgroundWorker(store=...)` and `HatchetBackgroundBackend(store=...)` |
| LGOS `RunCoordinator` | Coordination of execution and cleanup for a checkpoint thread | `GraphConfig(run_coordinator=...)` |
| LGOS `BackgroundBackend` | Submission and cancellation behind the polling API | `LanggraphOpenaiServe(background=...)` |

Ordinary foreground serving requires none of the LGOS persistence components.
Interrupt-enabled graphs require a checkpointer and coordinator.
Background execution additionally requires a background backend and, when
using the supplied worker, a response store. A LangGraph Store is needed only
when the application's graph uses one.

Checkpointers and LangGraph Stores use
[LangGraph's native interfaces](https://docs.langchain.com/oss/python/langgraph/persistence).
LGOS does not wrap or select their providers. For interrupts and background
execution, checkpointers must implement asynchronous `aget_tuple`, `aput`,
`aput_writes`, and `adelete_thread`. Validation checks those methods, not a
database brand; it cannot establish deployment durability.

## Supplied Integrations

Optional adapters are grouped by the LGOS capability they implement.
`langgraph_openai_serve.integrations.background` contains background-backend
and Response-store adapters, while
`langgraph_openai_serve.integrations.coordination` contains run coordinators.
There is deliberately no LGOS checkpoint integration package: applications
use a native LangGraph checkpointer directly.

| Deployment | Components |
| --- | --- |
| Single-process development | LangGraph `InMemorySaver`, LGOS `InMemoryRunCoordinator`, and `InMemoryBackgroundBackend` |
| Maintained durable example | LangGraph `AsyncPostgresSaver`, LGOS `PostgresRunCoordinator`, `PostgresResponseStore`, and `HatchetBackgroundBackend` |
| Application-selected infrastructure | A compatible LangGraph checkpointer, implementations of `RunCoordinator` and `ResponseStore`, and Hatchet or a custom `BackgroundBackend` |

Install the supplied durable integrations with:

```bash
uv add "langgraph-openai-serve[postgres,hatchet]"
```

SQLite and Redis response-store/coordinator adapters are not supplied by LGOS.
Using a LangGraph SQLite or Redis checkpointer does not supply the other roles.
Implementing `ResponseStore` is enough to replace response persistence while
keeping the Hatchet backend and LGOS worker.

!!! warning "Match the deployment scope"

    An in-memory coordinator protects only one process, even when checkpoints
    are stored in a shared database. Separate API and worker processes must
    share the appropriate response records, checkpoints, and coordination
    namespace. Do not use in-memory persistence for durable background work.

PostgreSQL can serve every persistence role in the example, but sharing one
database or connection pool is optional. The demo's `PostgresRuntime` is
application wiring, not a package requirement. Hatchet's own internal database
is separate from the LGOS response-store interface.

## Compose Application-Owned Resources

Construct and open clients in each process's application lifespan. This
composition assumes the application has already opened `checkpointer`,
`response_store`, and `hatchet`, constructed `coordinator`, and supplied its
graph `builder`, FastAPI `app`, and trusted `resolve_scope`:

```python
from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
)
from langgraph_openai_serve.integrations.background.hatchet import (
    HatchetBackgroundBackend,
    create_hatchet_workflows,
)

graph = builder.compile(checkpointer=checkpointer)
graphs = GraphRegistry(
    registry={
        "report": GraphConfig(
            graph=graph,
            description="Prepare a report.",
            background_version="v1",
            run_coordinator=coordinator,
        )
    }
)
workflows = create_hatchet_workflows(hatchet)
background = HatchetBackgroundBackend(
    workflow=workflows.response,
    runs=hatchet.runs,
    store=response_store,
)
LanggraphOpenaiServe(
    app=app,
    graphs=graphs,
    checkpoint_scope=resolve_scope,
    background=background,
).bind_openai_api()
```

The worker process registers `workflows.registrations` with Hatchet and yields
a `BackgroundWorker`, built from its own connections to the same logical
resources, from the Hatchet worker lifespan. Use consistent graph versions and
background settings in both processes. See [Background Responses](background-responses.md) for worker setup.

The application owns credentials, pool sizing, startup, and shutdown. Run
adapter setup, such as `PostgresResponseStore.setup()` migrations, before
accepting work. Stop and drain workers
before closing their clients. An injected pool remains application-owned.
When sharing a PostgreSQL pool with the coordinator, reserve connections for
checkpoint and response I/O; see
[PostgreSQL coordination](../reference.md#postgresql-coordination).

## Implement A Response Store

Implement the public `ResponseStore` protocol from `langgraph_openai_serve`.
It operates on `NewRun` and `StoredRun`, with no SQL or client types.
`ResponseStatus` is public in `langgraph_openai_serve.background.store`.

| Operation | Required behavior |
| --- | --- |
| `create` | Persist a new `queued` run. When another stored run holds the same `idempotency_digest`, return that run instead; concurrent creates with one digest must yield one run. The digest is released when its run is deleted. |
| `get` | Return the run, including an expired one not yet removed. LGOS checks owner scope and expiry with `StoredRun.visible_to()`. |
| `mark_in_progress` | Move active work in progress without reviving a terminal record; return `None` for a missing or terminal record. |
| `finish` | Atomically commit the first terminal Response with its retention deadline and cleanup intent. Return the winner, which is the existing run when another outcome already won, or `None` when the run is missing. |
| `claim_queued` | Return a bounded batch of runs still `queued` that were created before a cutoff, touching `updated_at` so a queued backlog rotates. Maintenance resubmits them. |
| `claim_cleanup_ready` | Return a bounded batch of runs with pending checkpoint cleanup and touch `updated_at` so unfinished work rotates fairly. Claims are not exclusive leases. |
| `finish_cleanup` | Clear cleanup intent after deletion, or when deletion cannot be performed. |
| `expire` | Delete a bounded batch of expired runs whose cleanup is no longer pending. Never expire active work. |

Every transition must be atomic across all clients in the deployment, so plain
key-value `get`/`set` is not enough; use conditional updates or transactions.

## Implement A Run Coordinator

Import `RunCoordinator`, `RunLease`, `RunBusyError`, and
`RunLeaseLostError` from `langgraph_openai_serve` or
`langgraph_openai_serve.graph.coordination`. Coordination applies to both
interrupts and background execution.

```python
async with coordinator(checkpoint_thread_id) as lease:
    lease.ensure_owned()
    # Read, execute, or clean up the protected checkpoint thread.
```

Implementations must:

1. Reject an occupied key with `RunBusyError`; do not queue behind its owner.
   Different keys may proceed concurrently, subject to configured capacity.
2. Yield a fresh `RunLease` for every acquisition. Release only that acquisition
   on all exit paths, including cancellation.
3. When ownership is lost or uncertain, set `lease.lost = True` before
   cancelling the owning task, and propagate failure when the context exits.
   Never reset a lost lease or silently reacquire within the same context.
4. Use the same coordination namespace for all execution and cleanup touching
   the same checkpoint state.

LGOS checks ownership at execution boundaries, before publishing background
outcomes, and before checkpoint cleanup. Lost background ownership is retryable;
it must not become a terminal graph failure. Interrupt execution preserves
checkpoints when ownership is lost. Callers must propagate cancellation.

!!! warning "Cooperative ownership is not fencing"

    A lease flag and task cancellation cannot stop a paused process, an
    already-issued database write, or an external side effect. A TTL lock can
    expire while its old owner is still executing. Document the coordinator's
    failure assumptions. Strong stale-writer rejection requires fencing
    enforced by the protected storage; merely returning a token does not
    provide it. LGOS's native checkpointer interface does not add fencing.

The PostgreSQL coordinator monitors its exact advisory-lock session and
cancels the owning task on a detected session failure. It also has a detection
interval and cooperative cancellation limits. Consult the
[Redis distributed-lock guidance](https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/#disclaimer-about-consistency)
before implementing an expiring distributed lease.

## Recovery Across Components

Separate stores and the execution engine do not share an atomic transaction.
LGOS recovers the gaps in stages:

1. Commit the accepted Response before submitting its ID to the engine. There
   is no outbox: after a failed submission, or a process crash between the
   commit and the submission, maintenance resubmits the still-queued Response.
   The engine must treat a duplicate submission as a no-op.
2. Persist graph progress and final output in checkpoints. If publication fails,
   a retry reconstructs the Response from the completed checkpoint.
3. Commit the terminal Response before deleting checkpoints. Retain cleanup
   intent until deletion succeeds or cannot be performed.

Keep checkpoints for the whole active, retry, and recovery period. Application
tools must tolerate replay of unfinished work; neither coordination nor
checkpoint recovery promises exactly-once external effects.

## Test A Custom Adapter

The repository contains fixture-driven contract suites:

- `tests/background/test_store.py` uses `response_store_pair`: two clients
  sharing a fresh, empty response namespace. `response_store` selects the first.
- `tests/graph/test_coordination.py` uses `coordinator_pair`: two clients
  sharing a fresh coordination namespace with capacity for two distinct keys.

To reuse them in an adapter project, copy the test modules and provide these
fixtures in its `conftest.py`. Enable AnyIO's asyncio test backend and own
connections with yield fixtures. The default LGOS fixtures use the same
in-memory instance twice; durable adapters should use independently opened
clients against the same isolated storage.

The suites exercise idempotent creation, terminal races, retention, cleanup
rotation, contention, and release after failure or cancellation. Add
adapter-specific tests for process restart, connection loss, lease revocation,
and concurrent processes; in-process contracts alone do not establish
distributed safety.
