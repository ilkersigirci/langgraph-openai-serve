# 05 — Responses Event Assembly

- Status: **Waiting for 02 and 03**
- Priority: **P1**
- Dependencies: **02, 03**

## Objective

Keep one explicit OpenAI Responses state machine, give graph orchestration and
wire assembly clear boundaries, and remove only complexity that compensates for
coarse runner events.

## Current Assessment

`api/responses/streaming.py` is the largest package module, but most of its state
is required by the Responses contract:

- response creation and in-progress events;
- monotonically increasing sequence numbers;
- output item and content part add/delta/done lifecycles;
- stable output indexes;
- commentary and final-answer phases;
- streamed text reconciliation with the final `AIMessage`;
- refusals, valid and invalid function calls, custom tools, web search, and
  interrupts;
- completed, incomplete, and failed terminal Responses;
- usage and citation annotations.

It already constructs official OpenAI SDK event and output models. That is the
correct package-native boundary. The SDK's client decoder and stream manager do
not provide a public server-side lifecycle builder.

The avoidable issue is that the same file also owns graph execution,
exception-to-event handling, SSE encoding, and collection of non-streaming
responses. `ServerToolTracker` then consumes coarse root update dictionaries.

## Files In Scope

- `src/langgraph_openai_serve/api/responses/streaming.py`
- `src/langgraph_openai_serve/api/responses/service.py`
- `src/langgraph_openai_serve/api/responses/server_tools.py`
- Responses views only for ownership and naming changes
- Responses event fixtures and behavior tests

Do not expand the supported Responses subset or change interrupt, file, client
tool, custom tool, or web-search ownership in this unit.

## Implementation Steps

1. Update the graph-to-Responses adapter to consume unit 02's native runner
   events. Remove casts and broad unions that no longer describe possible
   values, while keeping `Any` or mapping validation where LangGraph graph data
   is genuinely user-defined.
2. Keep one stateful accumulator for streaming and non-streaming output. Its
   public methods should accept protocol facts: final-text delta, status
   commentary, server-tool item/update, final assistant message, interrupt
   batch, or failure.
3. Keep mutable state local to that accumulator. Sequence number, output list,
   active final text item, and server-tool call/result correlation are clear
   local mutation and do not need immutable copies after every event.
4. Separate graph/I/O orchestration from event lifecycle construction only if
   doing so produces one clean module boundary without import cycles. A split
   such as builder/events versus stream orchestration is reasonable; splitting
   every event family into a file is not. If the existing single module remains
   clearer after the type changes, keep it and record that decision.
5. Make terminal ownership explicit. Exactly one completed, incomplete, or
   failed terminal event must be produced, and the non-streaming collector must
   obtain the same final `Response` from that path.
6. Keep server-tool correlation separate from generic text-item state. Continue
   accepting LangGraph's native `UpdatesStreamPart`; validate only the
   graph-authored message shape the tracker consumes and retain its pending-call
   completeness check. Do not invent an exhaustive type for arbitrary graph
   state updates.
7. Simplify large branch methods such as `_tool_item()` only when named helpers
   correspond to real SDK output families. Do not replace an exhaustive type
   branch with reflection or dynamic model lookup.
8. Keep `encode_event()` as minimal standard SSE framing. Do not import private
   `openai-python` client streaming internals.
9. Review `ResponseContext` for a small immutable data object, but leave it
   alone if converting it only moves constructor arguments between files.

## Required Behavior

- Streaming event order and sequence numbers remain byte-equivalent after
  normalization for existing fixtures.
- Non-streaming and streaming Responses contain the same ordered output items,
  text, citations, calls, results, statuses, and usage.
- Commentary never enters `response.output_text`; final text excludes
  commentary in both modes.
- A final message can reconcile provider text not emitted as token deltas.
- Server custom tools return call and output pairs; web search returns its
  standard call and citations; client functions remain pending for the caller.
- Interrupts finish as a completed Response containing resumable function calls.
- Midstream failure emits the existing OpenAI error event and one failed
  terminal Response, then releases the run through unit 03.

## Tests

The existing stream fixtures are valuable contract tests. Keep them and the
focused tests for text before interrupt, refusals, malformed tool calls, server
tools, citation indexes, terminal outcomes, cancellation, and usage.

Do not rewrite fixtures from the refactored implementation. Expected event
payloads must remain independently reviewed wire examples. Existing normalized
fixtures, rather than raw UUIDs or timestamps, define event parity.

This unit must end with either one demonstrably cleaner orchestration/builder
boundary or a recorded no-split decision. Do not churn event-family methods
merely to make the largest file shorter.

## Validation

```bash
just check
just test tests/api/responses tests/api/interrupt tests/api/test_chat_cancellation.py
just test
cd demo && just test --editable
```

If event payloads intentionally change to match the upstream OpenAI contract,
document the source and update the compatibility document in the same unit.

## Outcome

Not started.
