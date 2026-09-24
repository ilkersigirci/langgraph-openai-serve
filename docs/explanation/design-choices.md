# Design Choices

Why LGOS works the way it does. Add a row for each significant decision; when
one changes, update its row.

| Choice | Why | Cost | Revisit when |
| --- | --- | --- | --- |
| Background runs live only in Hatchet; LGOS keeps no Response table. | One source of truth: no outbox, resubmission, cleanup, or migrations. | Polling depends on Hatchet; retention is Hatchet's (30 days by default when self-hosted), not `store`; an unknown ID takes about 17 seconds to return 404. | Responses must outlive Hatchet or its retention. |
| The background worker runs the foreground Responses path, with no retries or checkpoint resume. | One execution path; resuming needed a second runner and recovery machinery. | A run that raises fails. A run reassigned after its worker dies starts over, and an interrupt answer that had already applied then fails as stale. | Runs are too long or expensive to repeat. |
| Background creates accept an `Idempotency-Key` header. | SDKs and gateways retry creates, which would otherwise start duplicate runs. | It is not an OpenAI header, and gateways must forward it. | OpenAI offers a retry-safe create. |
| A node calls `interrupt()` at most once per invocation. | LangGraph's interrupt IDs then identify each pause without scanning checkpoints. | A node cannot ask two questions. | LangGraph gives each `interrupt()` call a unique ID. |
