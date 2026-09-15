# 03 — Run Ownership And Cancellation

- Status: **Waiting for 02**
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
   a time.
4. Represent checkpoint retention with the smallest state that expresses the
   rule: preserve only a batch exposed for resume; delete every terminal or
   unclassified interrupt thread. Cleanup failure may replace a successful
   response, but it must not mask the original graph or cancellation failure.
5. Keep shielding limited to cleanup sections that must finish after request
   cancellation. Do not shield graph execution.

### 2. Evaluate the HTTP stream owner

First make a local branch that streams the protocol generator directly through
the locked Starlette response and relies on the prepared-run context for
cleanup. Run the real TCP cancellation tests, including immediate close. Keep
the simpler path only if all graph, provider, nested generator, and lease
finalizers complete deterministically.

If native Starlette consumption still fails, retain a dedicated producer owner.
Reduce it to the task and resources it must actually own after unit 02, and make
its lifecycle an async context rather than a collection of externally ordered
`start()`/`aclose()` mutations where practical. The zero-buffer AnyIO channel is
valid if it remains necessary for backpressure.

### 3. Evaluate LangGraph v3 separately

Locked LangGraph's `AsyncGraphRunStream` owns a caller-driven pump, backpressure,
and an `abort()` path that cancels an in-flight `__anext__`. Build a disposable
spike using `astream_events(version="v3")` and compare it with the stable v2
runner for:

- root and nested message filtering by node and `nostream` tag;
- custom status events and root server-tool updates in arrival order;
- final output and complete direct/parallel/nested interrupts;
- usage aggregation;
- `durability="exit"` and checkpoint cleanup;
- real TCP disconnect and immediate close.

Adopt v3 only if the locked experimental API passes all cases and removes the
custom owner or a material amount of runner code. Do not ship both paths, add a
feature flag, or update LangGraph merely to make the spike pass. Otherwise
delete the spike and retain stable v2.

## Required Behavior

- Preparation errors remain ordinary OpenAI HTTP errors before SSE headers.
- A client disconnect promptly cancels graph and provider work and completes
  their async finalizers.
- Closing before the source starts releases the prepared run.
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

Not started.
