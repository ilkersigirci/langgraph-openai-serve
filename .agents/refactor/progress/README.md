# Refactor Progress

Analysis completed on 2026-09-15. No production code, tests, dependencies, or
product documentation were changed during this review. This directory is the
handoff for later implementation work.

The repository is healthy, but complexity is concentrated in a few boundaries.
The first changes fix nested OpenAI request validation and consume LangGraph's
native typed results. Interrupt durability and Responses event assembly contain
substantial necessary complexity; their work units narrow and contain that
complexity instead of deleting safeguards.

Units 01–06 are the core refactor. After unit 06, re-baseline before touching
the demo clients. Units 07–09 are evidence-gated follow-through: they may record
that the final core API requires no further change, and they must not become a
general rewrite of otherwise healthy demo code.

## Rules For Every Work Unit

- Preserve the documented OpenAI `/v1` wire contract. Internal Python APIs may
  break without a compatibility layer.
- Recheck the locked dependency's public API before coding. Do not copy code
  from a newer package version or depend on an undocumented implementation.
- Make the smallest complete change. Do not combine adjacent cleanup merely
  because the same file is open.
- Update affected callers, focused behavior tests, and product documentation in
  the same implementation change.
- Assert behavior through HTTP, the official OpenAI SDK, or another stable
  public boundary. Retain interaction tests only where the interaction is the
  contract, such as cancellation and PostgreSQL session-lock ownership.
- Do not use line count as a success metric. Remove code, casts, state, or I/O
  only when the resulting responsibility is clearer.
- Before starting a unit, change its status both here and in the unit file to
  **In progress**. On completion, keep the two statuses synchronized and record
  the result and validation in that unit's Outcome section.
- Keep dependency experiments disposable. Do not ship an experimental API,
  feature flag, or dual implementation unless the unit's acceptance criteria
  explicitly require it.

## Ordered Work Units

Priority labels describe refactor sequencing, not incident severity; the
repository is passing its baseline.

| ID | Work unit | Priority | Depends on | Status |
| --- | --- | --- | --- | --- |
| 00 | [Baseline and decisions](00-baseline-and-decisions.md) | Reference | — | Analysis complete |
| 01 | [Strict OpenAI request boundaries](01-strict-openai-request-boundaries.md) | P0 | — | Complete |
| 02 | [Native LangGraph execution results](02-native-langgraph-execution.md) | P0 | — | Ready |
| 03 | [Run ownership and cancellation](03-run-ownership-and-cancellation.md) | P0 | 02 | Waiting for 02 |
| 04 | [Interrupt continuation identity](04-interrupt-continuation-identity.md) | P1 | 02 | Waiting for 02 |
| 05 | [Responses event assembly](05-responses-event-assembly.md) | P1 | 02, 03 | Waiting for 02 and 03 |
| 06 | [Graph registration](06-graph-registration.md) | P1 | 02, 03, 04 | Waiting for core execution work |
| 07 | [Chainlit interrupt ledger](07-chainlit-client.md) | P2 | 05, 06 | Waiting for core work |
| 08 | [Open WebUI Function runtime](08-openwebui-client.md) | P2 | 05, 06 | Waiting for core work |
| 09 | [Demo registration migration and audit](09-demo-graphs-and-services.md) | P2 | 06 | Waiting for 06 |
| 10 | [Closeout audit](10-closeout-audit.md) | P2 | 01–09 | Waiting for implementation |

Units 01 and 02 are independent and may be implemented in either order. After
02, units 03 and 04 are independent: one owns execution lifetime and the other
owns continuation identity. Unit 05 follows 03. Unit 06 follows both 03 and 04
so its checkpointer capability checks reflect the final execution and interrupt
paths. Keep 04 and 05 as separate changes because interrupt identity and
Responses assembly have different invariants.

After 06, run the package suite and the editable demo suite, then reread the
recorded outcomes before starting 07–09. A client or demo unit is complete when
its named boundary is clean and validated; a reviewed **no change required**
outcome is preferable to speculative cleanup.

## Main Findings

1. Nested unknown fields currently pass validation even though the documented
   Responses and Chat subsets reject unknown request fields. This is a contract
   bug, not an aesthetic cleanup.
2. The runner requests LangGraph v2 results but casts them back to coarse
   dictionaries and performs another state read after execution. Native
   `GraphOutput` and `StreamPart` types expose the output and interrupts already.
3. `GraphRun`, `prepare_run`, `finalize_run`, and `_StreamOwner` divide ownership
   of one run across several mutable states. Cleanup guarantees are required,
   but their ownership can be made easier to follow.
4. The interrupt state token reads checkpoint history and the `__resume__`
   pending-write channel. A local probe confirmed that sequential interrupts can
   reuse both the raw interrupt ID and checkpoint ID, including inside a nested
   invocation. Exact stale-resume rejection therefore needs this generation
   signal until LangGraph publishes a durable alternative.
5. `ResponsesStreamBuilder` is large because the standard event grammar is
   large. It already uses official OpenAI SDK output and event models. Refactor
   its orchestration boundary and types; do not replace it with hand-built
   dictionaries or separate streaming and non-streaming implementations.
6. `GraphConfig` performs static registration checks during every resolution,
   while `GraphRegistry` uses Pydantic validation and serialization machinery
   for a small mapping. This can be simplified after runner ownership settles.
7. The Chainlit interrupt ledger and Open WebUI pipe each mix host callbacks,
   durable continuation data, and rendering. They need client-specific cleanup;
   their platform contracts are different and should not be forced behind one
   shared UI abstraction.

## Intentionally Retained Designs

- `ClientSettings` keeps its strict frozen model, cached default construction,
  JSON round trip, and non-finite-number checks. Existing tests demonstrate
  independent registration and request-boundary guarantees.
- `PostgresRunCoordinator` keeps a PostgreSQL session advisory lock on one
  pooled connection and discards a connection after an indeterminate acquire or
  release. PostgreSQL session-lock semantics make this necessary across workers.
- Responses output and streaming events continue to use official OpenAI SDK
  models. There is no public server-side SDK encoder that replaces the local SSE
  framing.
- Stateless Responses replay validation remains protocol-specific. LangChain's
  general message conversion helpers target chat-style role messages and do not
  enforce LGOS's item-ID and call/output causality rules.
- Real TCP cancellation tests, interrupt restart and stale-generation tests, and
  Responses stream fixtures remain contract tests even when they touch difficult
  execution paths.

## Completion Definition

The refactor is complete when units 01–09 have recorded outcomes (including any
evidence-backed no-change outcomes), unit 10 passes, the OpenAI contract
documentation matches observed behavior, no temporary compatibility layer
remains, and every retained exception to the simple design has a nearby
explanation of the invariant it protects.
