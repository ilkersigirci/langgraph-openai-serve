# Refactor Progress

Initial analysis completed on 2026-09-15, and implementation units 01–10
completed on 2026-09-16. This directory records the decisions, sequencing,
validation, and retained invariants for the finished refactor.

The refactor fixed nested OpenAI request validation, consumes LangGraph's native
typed results, and gives prepared runs one explicit owner. Interrupt durability
and Responses event assembly retain substantial necessary complexity; their
work units narrow and contain it instead of deleting safeguards.

Units 01–06 were the core refactor. Units 07–09 were evidence-gated
follow-through: they could record that the final core API required no further
change and did not become a general rewrite of otherwise healthy demo code.

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
| 02 | [Native LangGraph execution results](02-native-langgraph-execution.md) | P0 | — | Complete |
| 03 | [Run ownership and cancellation](03-run-ownership-and-cancellation.md) | P0 | 02 | Complete |
| 04 | [Interrupt continuation identity](04-interrupt-continuation-identity.md) | P1 | 02 | Complete |
| 05 | [Responses event assembly](05-responses-event-assembly.md) | P1 | 02, 03 | Complete |
| 06 | [Graph registration](06-graph-registration.md) | P1 | 02, 03, 04 | Complete |
| 07 | [Chainlit interrupt ledger](07-chainlit-client.md) | P2 | 05, 06 | Complete |
| 08 | [Open WebUI Function runtime](08-openwebui-client.md) | P2 | 05, 06 | Complete |
| 09 | [Demo registration migration and audit](09-demo-graphs-and-services.md) | P2 | 06 | Complete |
| 10 | [Closeout audit](10-closeout-audit.md) | P2 | 01–09 | Complete |

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

## Initial Findings And Disposition

1. Nested unknown fields passed validation even though the documented Responses
   and Chat subsets reject them. Unit 01 replaced SDK output validators with
   strict local request models.
2. The runner cast LangGraph v2 results back to coarse dictionaries and read
   state after execution. Unit 02 consumes native `GraphOutput` and `StreamPart`
   values directly and removed the post-run read.
3. `GraphRun`, `prepare_run`, `finalize_run`, and `_StreamOwner` divided one run's
   ownership across several mutable states. Unit 03 made `GraphRun` the resource
   owner, removed `finalize_run()`, and retained the HTTP producer owner only for
   ASGI cancellation.
4. The interrupt state token reads checkpoint history and the `__resume__`
   pending-write channel. A local probe confirmed that sequential interrupts can
   reuse both the raw interrupt ID and checkpoint ID, including inside a nested
   invocation. Exact stale-resume rejection therefore needs this generation
   signal until LangGraph publishes a durable alternative.
5. `ResponsesStreamBuilder` was large because it mixed the standard event
   grammar with execution and I/O. Unit 05 separated orchestration and renamed
   the one shared streaming/non-streaming accumulator `ResponsesEventBuilder`.
6. `GraphConfig` repeated static checks during resolution, while
   `GraphRegistry` used Pydantic serialization machinery for a small mapping.
   Unit 06 made config values frozen and the registry a focused mapping owner.
7. The Chainlit interrupt ledger and Open WebUI pipe mixed host callbacks,
   durable continuation data, and rendering. Units 07 and 08 isolated strict
   continuation codecs while keeping their different host contracts separate.

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
