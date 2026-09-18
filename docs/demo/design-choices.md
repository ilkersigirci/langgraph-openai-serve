# Demo Design Choices

## Chainlit human review is event-driven

### Decision

A pending review is a persisted `cl.Message` with a `cl.CustomElement`.
`on_message` publishes it and returns. The element submits the complete batch
through `callAction`; the callback validates it and performs one Responses
continuation. A chained interrupt updates the same form, while a terminal
response completes the ledger and removes it.

Message metadata is authoritative for the model, Response, function calls,
revision, and element ID. The browser sends only opaque references and answers.

### Why not the alternatives?

- `Ask*Message` keeps the handler waiting on its WebSocket. Navigation then
  needs task, reconnect, and cancellation bookkeeping. Our former `on_chat_end`
  cancellation could also stop an unrelated stream and persist partial output.
- Plain `cl.Action` controls use native callbacks but are not persisted thread
  elements in Chainlit 2.12, so history hydration cannot restore them.

A custom element provides the durable form, while `callAction` keeps the native
Chainlit endpoint and callback registry. No custom API route is needed.

### Pros

- Navigation and reload restore reviews through normal history hydration.
- No waiter, resume task, timer, or HITL session cache is needed.
- Batch validation, persisted revisions, and a per-step lock reject incomplete,
  stale, or duplicate submissions.
- Ordinary responses can finish and persist after a thread switch.

### Cons

- The form needs custom JSX; drafts and decisions are not a separate audit log.
- The lock is process-local; cross-worker submission needs durable coordination.
- Continuation and persistence are not transactional. Completing the ledger
  first favors replay safety over guaranteed final rendering.
- Running responses are not rebound to another WebSocket. Returning early may
  require a refresh to hydrate the completed message.

### Chainlit 2.12 timestamp workaround

The official PostgreSQL layer hydrates `createdAt` in a format rejected by its
own update path. `HitlWorkflow` normalizes it before updating a restored ledger.
Otherwise the UI can look complete while PostgreSQL keeps a ghost `pending`
ledger that blocks the next turn. Recheck this workaround after upgrading
Chainlit.
