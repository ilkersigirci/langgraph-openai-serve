# 02 — Native LangGraph Execution Results

- Status: **Ready**
- Priority: **P0**
- Dependencies: **None**

## Objective

Consume LangGraph 1.2.9's stable v2 result and stream types directly, remove the
post-execution `aget_state()` read, and pass the narrowest honest event types to
the protocol adapters.

## Problem

`graph/runner.py` requests `version="v2"` but then:

- casts `ainvoke()` to `GraphOutput[Any]`;
- casts `astream()` to `AsyncGenerator[dict[str, Any], None]`;
- reimplements discrimination with `.get("type")`;
- types message events as a bare `dict`;
- calls `graph.aget_state()` after both invoke and stream to rediscover
  interrupts already present in native results.

Locked LangGraph exposes `GraphOutput.interrupts`, `StreamPart`,
`MessagesStreamPart`, and `ValuesStreamPart.interrupts`. Using those values
reduces local type assertions and one checkpoint read per executed interrupt
request.

## Files In Scope

- `src/langgraph_openai_serve/graph/runner.py`
- `src/langgraph_openai_serve/graph/utils.py`
- `src/langgraph_openai_serve/graph/interrupt/state.py`
- focused runner and interrupt tests under `tests/graph/runner/` and
  `tests/api/interrupt/`
- small protocol-adapter type updates required by the new runner event union

Do not redesign run resource ownership, the state-token algorithm, Responses
event ordering, or graph registration in this unit.

## Implementation Steps

1. Pass `version="v2"` as a literal visible at the `ainvoke()` call site so the
   locked overload resolves to native `GraphOutput`. Do not hide the literal in
   a `dict[str, Any]` expanded through `**kwargs`, which erases the useful return
   type. Render `result.value`, and build an interrupt batch from
   `result.interrupts` when present.
2. Apply the same rule at `astream()`: keep `version="v2"` visible to the type
   checker, consume LangGraph's `StreamPart` union, narrow on `part["type"]`,
   and give message extraction a `MessagesStreamPart` rather than a bare
   dictionary. Invocation option helpers may remain only for values that do not
   erase overload selection.
3. Treat root `ValuesStreamPart` values (`ns == ()`) as the durable final output.
   Capture `part["data"]` and accumulate its interrupts in first-seen order.
   Repeated IDs with the same native interrupt may be de-duplicated; the same ID
   paired with conflicting data is invalid durable state and must fail loudly.
4. Do not assume the final root values part contains the complete parallel
   interrupt set. A local locked-version probe emitted parallel nested
   interrupts across multiple root values parts.
5. Change interrupt-batch construction to accept native interrupts plus the
   prepared runnable configuration and run ID. The state token may still scan
   checkpoint tuples, but batch construction must no longer fetch a new
   `StateSnapshot` after execution.
6. For a retry that re-emits an already pending batch without execution, reuse
   the validated interrupts from the snapshot obtained during preparation.
   Carry only that minimal prepared data to the runner; do not retain the whole
   snapshot or make a second state read.
7. Delete casts and `Any` made obsolete by native discriminated types. Keep
   `Any` at graph input/output adapter boundaries where graph schemas are truly
   user-defined.

## Required Behavior

- Non-streaming and streaming requests return the same final assistant output,
  interrupt batch, usage, custom events, and server-tool updates as before.
- Direct, sequential, nested, indirectly nested, and parallel interrupts keep
  their complete durable batch semantics.
- A retry with the same run ID re-emits pending calls without executing graph
  nodes.
- Interrupt-enabled runs still use `durability="exit"`.
- The initial state read remains for new/retry/resume conflict checks. Only the
  redundant post-execution read is removed.
- Terminal checkpoint deletion and lease release remain unchanged until unit 03.

## Tests

Retain the existing public behavior tests. Add or adjust focused runner cases
for native v2 inputs only where they distinguish an incomplete implementation:

- `GraphOutput.interrupts` on invoke;
- multiple root values parts forming one parallel batch, including rejection of
  a conflicting duplicate ID;
- nested and indirectly nested interrupt streaming;
- final root value selection with subgraph values present;
- custom, updates, and message-part narrowing;
- interrupted output never passed to `output_to_message`.

Do not add a mock assertion whose only purpose is counting `aget_state()` calls.
The removal should be evident in the runner and protected by native-result
behavior tests.

## Validation

```bash
just check
just test tests/graph/runner tests/api/interrupt tests/api/responses tests/api/test_chat_cancellation.py
just test
```

## Deferred Native Option

LangGraph's v3 `astream_events()` has useful typed projections, native abort,
and aggregated interrupts, but locked 1.2.9 labels it experimental. It is not
part of this refactor. Reconsider it only as a separate migration against a
locked stable release and the full behavior matrix recorded in unit 00.

## Outcome

Not started.
