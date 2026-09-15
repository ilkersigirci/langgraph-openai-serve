# 03 — Run Ownership And Cancellation

- Status: **Complete**
- Priority: **P0**
- Dependencies: **02**

## Objective

Give one object or context one clear ownership path for a prepared graph run,
its coordinator lease, checkpoint disposition, stream producer, and cleanup.
Keep deterministic cancellation and primary-error precedence.

## Problem

Ownership currently spans four places:

- `GraphRun` stores execution data, usage aggregation, checkpoint identity,
  `should_execute`, and a manually entered async lease.
- `prepare_run()` duplicates ordinary and interrupt construction, enters the
  lease through `__aenter__`, and recreates context-manager exception handling.
- `invoke_run()` and `stream_run()` each mutate an
  `"unknown" | "preserve" | "delete"` disposition before calling
  `finalize_run()`.
- `_StreamOwner` owns an asyncio producer task, both ends of an AnyIO channel,
  and the prepared run because Starlette owns response iteration.

Every safeguard has a reason, but the reader must reconstruct who closes what
on preparation failure, graph failure, client disconnect, immediate close,
interrupt, and normal completion.

## Files In Scope

- `src/langgraph_openai_serve/graph/utils.py`
- `src/langgraph_openai_serve/graph/runner.py`
- `src/langgraph_openai_serve/api/streaming.py`
- Chat and Responses services/views that transfer run ownership
- cancellation, concurrency, interrupt-cleanup, and PostgreSQL coordinator tests

Do not change the OpenAI payload, interrupt continuation identity, or response
event state machine in this unit.

## Implementation Steps

### 1. Consolidate prepared-run resources

1. Use the standard library's `AsyncExitStack` or one equally direct async
   context owner for the coordinator lease. Avoid storing a context manager that
   was entered elsewhere and later invoking its dunder exit method manually.
2. Build common request, graph, callback, input, and context values once. Keep
   the interrupt-only state read and checkpoint identifiers in a small branch.
3. Make close idempotent and make ownership transfer explicit at the route,
   service, runner, and stream boundary. Exactly one layer should own cleanup at
   a time. Do not require every caller to invoke both a runner finalizer and a
   separate resource close in a particular order.
4. Represent checkpoint retention with the smallest state that expresses the
   lifecycle table below. A batch is *committed* when the runner has validated
   and returned or yielded it; this does not claim that the network peer
   received the bytes, which the server cannot know. Cleanup failure may replace
   a successful result, but it must not mask the original graph or cancellation
   failure.
5. Keep shielding limited to cleanup sections that must finish after request
   cancellation. Do not shield graph execution.

### Checkpoint and lease lifecycle

| Situation | Checkpoint disposition | Lease |
| --- | --- | --- |
| Preparation or resume validation fails before execution | Leave any pre-existing thread untouched | Release |
| A retry re-emits an already pending batch | Preserve | Release after the batch is committed |
| Execution returns or yields a validated interrupt batch | Preserve | Release after the batch is committed |
| Execution completes normally without an interrupt | Delete the temporary thread | Release after deletion |
| Execution, output rendering, or streaming fails or is cancelled after execution starts and before a batch is committed | Delete incomplete state best-effort | Release |
| Cleanup itself fails while another failure is active | Keep the original failure primary and log cleanup failure | Attempt release exactly once |

For non-interrupt graphs, checkpoint disposition is not applicable but the same
single-owner and primary-error rules still apply.

### 2. Simplify the HTTP stream owner conservatively

Retain a dedicated producer owner by default: repository history and the real
TCP regression tests show that it protects cancellation of nested graph and
provider generators. After unit 02, reduce it to the task and resources it must
actually own, and make its lifecycle an async context rather than a collection
of externally ordered `start()`/`aclose()` mutations where that is genuinely
simpler. The zero-buffer AnyIO channel remains valid backpressure.

Deleting the owner is optional, not a required experiment. Do so only if a
disposable direct-StreamingResponse probe passes the real TCP disconnect,
immediate-close, provider-finalizer, graph-finalizer, and lease-release cases.
An in-process ASGI test is insufficient. Do not retain the probe or a second
streaming path when it fails.

### Deferred API migration

Do not evaluate or adopt LangGraph v3 in this unit. The locked implementation is
experimental, and combining an execution-API migration with lifecycle cleanup
would make failures and cancellation regressions harder to attribute. Unit 00
records the future replacement gate.

## Required Behavior

- Preparation errors remain ordinary OpenAI HTTP errors before SSE headers.
- A client disconnect promptly cancels graph and provider work and completes
  their async finalizers.
- Closing before the source starts releases the prepared run.
- Invalid or stale resume preparation releases the lease without deleting the
  valid pending checkpoint that the caller may still resume correctly.
- A completed interrupt releases its coordinator lease but preserves its
  checkpoint thread.
- Success without an interrupt deletes the temporary checkpoint thread.
- Graph failure, response failure, cancellation, and cleanup failure preserve
  current error precedence.
- `PostgresRunCoordinator` holds its advisory lock on one pool connection until
  the run exits and discards indeterminate sessions.

## Validation

```bash
just check
just test tests/api/test_chat_cancellation.py tests/api/interrupt tests/api/responses tests/integrations/test_postgres.py
just test
cd demo && just test --editable
```

Run the PostgreSQL integration recipe when that service is available. An
in-process `ASGITransport` run is not a substitute for the real TCP disconnect
case.

## Outcome

Completed on 2026-09-15.

- `GraphRun` is now the idempotent async owner for one prepared execution. Its
  `AsyncExitStack` acquires and releases the coordinator lease, and one private
  `"untouched" | "delete" | "preserve"` state implements the checkpoint
  lifecycle: preparation leaves state untouched, execution makes incomplete
  state cleanup-eligible, and committing a validated interrupt batch preserves
  it.
- Preparation builds the graph, usage callback, run identity, runnable config,
  input, and context through one common path. The interrupt branch alone
  acquires the lease, reads state, and resolves retry or resume input. Failed or
  cancelled preparation unwinds the resource stack under a cleanup shield
  without deleting pre-existing checkpoint state.
- Direct runner wrappers and the Chat and Responses services now express
  ownership with `async with run`. Lower-level `invoke_run()` and `stream_run()`
  only advance execution and interrupt-commit state; the separate
  `finalize_run()` function and manually entered lease were removed.
- The dedicated HTTP producer owner and zero-buffer AnyIO channel were retained.
  The owner is itself an async context, cancels and awaits the asyncio producer,
  and provides idempotent fallback cleanup when a response closes before its
  service source starts. Once a service starts, its prepared-run context closes
  graph/provider generators and releases the run before terminal protocol
  rendering.
- Cleanup remains shielded and ordered: checkpoint deletion precedes lease
  release, a cleanup error may replace a successful result, and cleanup errors
  are logged instead of replacing an active graph, rendering, or cancellation
  failure. Focused tests cover successful invoke/stream cleanup failure,
  combined checkpoint-and-lease cleanup failure, exact-once release, immediate
  close, active request-error precedence at the stream owner, and real TCP
  graph/provider cancellation.
- Product documentation now describes prepared-run ownership and the HTTP
  streaming handoff. No OpenAI payload, interrupt identity, Responses event
  state machine, dependency, or lockfile changed.

Locked-version verification used `uv.lock`, `uv run --locked`, and the installed
sources for AnyIO 4.14.2, FastAPI 0.139.2, Starlette 1.3.1, and Psycopg Pool
3.3.1. The implementation was checked against these primary sources:

- [Python 3.11 `AsyncExitStack`](https://docs.python.org/3.11/library/contextlib.html#contextlib.AsyncExitStack)
- [AnyIO 4.14.2 cancellation and shielding](https://github.com/agronholm/anyio/blob/4.14.2/docs/cancellation.rst)
- [FastAPI 0.139.2 request dependency lifetime](https://github.com/fastapi/fastapi/blob/0.139.2/fastapi/routing.py)
- [Starlette 1.3.1 `StreamingResponse`](https://github.com/Kludex/starlette/blob/1.3.1/starlette/responses.py)
- [Psycopg Pool 3.3.1 async connection lifecycle](https://github.com/psycopg/psycopg/blob/3.3.1/psycopg_pool/psycopg_pool/pool_async.py)
- [PostgreSQL session advisory locks](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS)

| Validation | Result |
| --- | --- |
| `just check` | Pass |
| `just test tests/api/test_chat_cancellation.py tests/api/interrupt tests/api/responses tests/integrations/test_postgres.py` | 161 passed |
| `just test` | 393 passed |
| `cd demo && just test --editable` | API 111 passed, Files 12 passed, Chainlit 116 passed, Open WebUI 120 passed |
| `cd demo && just test-postgres --editable` | API PostgreSQL 2 passed, Chainlit PostgreSQL 10 passed |
| `just docs` | Strict build passed |
