# 04 — Interrupt Continuation Identity

- Status: **Waiting for 02**
- Priority: **P1**
- Dependencies: **02**

## Objective

Contain interrupt resume validation in one module, isolate the one dependency
on LangGraph checkpoint internals, and preserve exact stale, parallel, nested,
restart, and tenant-scope behavior. This is a containment audit, not a mandate
to rewrite an algorithm that is already isolated and covered.

## Why This Cannot Be A Deletion Refactor

`graph/interrupt/state.py::checkpoint_state_token()` scans checkpoint history,
selects the latest checkpoint in each namespace, includes counts from the
`__resume__` pending-write channel, and hashes the result. This is the least
comfortable part of the interrupt design because it couples LGOS to checkpoint
tuple details and scales with thread history.

The coupling currently carries required information. Against locked LangGraph,
two sequential interrupts in the same task reused both their raw interrupt ID
and checkpoint ID. The same happened when that graph was invoked indirectly
inside a parent node. Only the recorded resume writes distinguished the second
generation. Root `StateSnapshot` traversal also cannot recover every indirectly
invoked subgraph namespace.

The following replacements are insufficient:

| Candidate | Why it fails |
| --- | --- |
| Interrupt ID | Reused by sequential calls in one task |
| Root checkpoint ID | May not advance for a nested or sequential pause |
| Pair of interrupt and checkpoint IDs | Both values can be reused together |
| Random Response ID | Cannot be reproduced after retry or process restart without a response store |
| In-memory generation counter | Fails across workers and restart |
| Graph lifecycle callback | Useful while executing, but not durable for a later retry after restart |
| Root state snapshot only | Does not expose every indirectly invoked subgraph checkpoint |

## Files In Scope

- `src/langgraph_openai_serve/graph/interrupt/state.py`
- `src/langgraph_openai_serve/graph/interrupt/models.py`
- `src/langgraph_openai_serve/graph/interrupt/validation.py`
- interrupt codec only where token construction or parsing changes
- graph registration capability checks only if the resulting implementation
  needs a smaller checkpointer surface
- interrupt behavior and upgrade tests

Do not add a response database, new request fields, a custom route, or a
user-configurable continuation provider in this unit.

## Implementation Steps

1. Keep run-ID normalization, checkpoint-scope derivation, current-state
   conflict checks, pending-set validation, and generation fingerprinting as
   separate named stages. A reader should be able to follow new request, retry,
   resume, and terminal paths without stepping through unrelated UUID helpers.
2. After unit 02, build durable batches from native interrupts and the existing
   runnable configuration. Remove the old post-execution snapshot helper when
   it has no remaining caller.
3. Give the fingerprint a name that describes its role as continuation
   generation, not merely “latest checkpoint.” Keep its algorithm and format in
   one location with a nearby comment explaining the sequential-ID evidence.
4. Limit raw `CheckpointTuple.pending_writes` and `RESUME` channel knowledge to
   that function. The rest of interrupt code should handle an opaque token.
5. Validate and document the exact `BaseCheckpointSaver` methods LGOS and
   LangGraph actually require. Do not loosen registration by deleting a method
   check until a real interrupt graph works with a saver lacking that method.
6. Keep the fingerprint deterministic across process restart and saver
   implementations. Preserve the existing lowercase 64-hex wire value and the
   versioned domain separator used to compute it. A future wire-format change
   needs an explicit codec migration; this unit does not add a visible prefix.
7. Preserve complete-batch resume through `Command(resume={id: value, ...})`.
   Do not convert parallel answers to positional lists.
8. Keep the recorded scan measurements as the baseline. Re-measure with a
   representative persistent saver only if this unit changes query behavior or
   new evidence suggests a regression. Optimize only if measured history sizes
   justify it and the public saver API can return every namespace head
   correctly. Do not add a cache that becomes another source of truth.

## Required Behavior

- A retried pending run re-emits the same semantic interrupt batch.
- A previous generation is rejected with HTTP 409 after a sequential graph
  advances, even if LangGraph reused its raw ID.
- Missing, duplicate, fabricated, mixed-generation, and partial batches fail at
  the existing 400 or 409 boundary.
- Direct, nested, indirectly nested, and parallel batches resume after a process
  and graph restart.
- Checkpoint scope, model, and run ID remain part of the private storage key.
- This unit does not change checkpoint retention or lease ownership; unit 03
  owns those rules.

## Tests To Retain

- `tests/api/interrupt/test_state.py`, especially sequential reused-ID and
  server-scope cases
- `tests/api/interrupt/test_contract.py` parallel and nested batches
- `tests/graph/runner/test_runner_interrupts.py` restart and capability cases
- PostgreSQL restart/runtime integration when available

Prefer these end-to-end signals over tests that reproduce the hash line by line.
If a small pure token encoder remains, one fixed-vector test is enough to protect
cross-restart stability.

If unit 02 leaves the checkpoint-internal knowledge already contained in one
well-named function and no simpler public saver API exists, record a no-change
outcome rather than renaming stable helpers for cosmetic reasons.

## Validation

```bash
just check
just test tests/api/interrupt tests/graph/runner tests/integrations/test_postgres.py
just test
cd demo && just test-postgres --editable --uri "$DEMO_API_TEST_POSTGRES_URI"
```

The final command is conditional on an available PostgreSQL test service.

## Future Replacement Gate

Replace the history scan when the locked or intentionally upgraded LangGraph
version exposes a documented durable generation/revision that changes for
sequential and indirectly nested interrupt pauses and can be recovered after
restart. Verify that claim with the same direct and nested probe before coding.

## Outcome

Not started.
