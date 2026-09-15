# 10 — Closeout Audit

- Status: **Waiting for implementation**
- Priority: **P2**
- Dependencies: **01–09**

## Objective

Verify the completed refactor as one system, remove temporary migration debris,
and record which remaining complexity protects an external contract or durable
runtime invariant.

This unit is an audit. It must not become a last-minute bucket for unrelated
rewrites.

## Audit Steps

1. Read the Outcome section of units 01–09. Resolve any unfinished item or
   explicitly move it to a new, bounded work unit with evidence; do not silently
   declare it complete.
2. Compare public OpenAPI schemas and representative official-SDK requests with
   the pre-refactor contract. Confirm that supported `/v1/models`,
   `/v1/chat/completions`, `/v1/responses`, Files demo, streaming SSE, error
   envelopes, metadata, and interrupt continuation behavior remain documented.
3. Search the changed core paths for remaining `Any`, casts, bare dictionaries,
   broad exception handling, manual lifecycle calls, and mutable state. Review
   each occurrence in context. Remove it only when a precise public or local
   type and a simpler ownership path exist.
4. Search for compatibility aliases, deprecated constructors, duplicated old
   and new paths, unused helpers, stale comments, and TODOs introduced during
   the refactor. Remove them; early development does not need a transition
   layer.
5. Review module boundaries against `.agents/CODE_STYLE.md`. Merge one-use files
   that created navigation without a responsibility. Split any remaining
   function only when the extracted unit has a clear input, output, and name.
6. Audit changed tests. Delete interaction assertions that merely preserve the
   former implementation, but retain tests for externally observable behavior
   and required ownership interactions. In particular, keep:
   - real-TCP cancellation and async finalization;
   - interrupt retry, stale generation, parallel batch, concurrency, and
     restart behavior;
   - normalized Responses stream fixtures and terminal status behavior;
   - PostgreSQL session-lock connection ownership and cancellation recovery;
   - client reconnect and persisted continuation behavior.
7. Review comments in complex retained code. Each comment should explain a host
   limitation, protocol ordering rule, durability invariant, or rejected native
   alternative. Remove narration and obsolete historical notes.
8. Review dependencies and lockfiles. No dependency or lockfile should change
   unless a completed unit records why its behavior required that update.
9. Build all documentation and check links from product docs and this progress
   directory. Make examples use only the final public API.
10. Record final file/line inventory as context, not a score. The meaningful
    result is fewer duplicated paths, state transitions, casts, post-run reads,
    and incidental test couplings.

## Full Validation

```bash
just check
just test
just docs
cd demo
just check --editable
cd ..
```

Also validate the deployed configurations and integrations affected by the
refactor:

```bash
cd demo
just compose-config
just test-postgres --editable --uri "$DEMO_API_TEST_POSTGRES_URI"
```

The PostgreSQL command requires a disposable database. Run the dedicated live
direct, LiteLLM, or Bifrost tests only when their services and credentials are
available. Record each skipped external check in the Outcome instead of
claiming it passed.

For the two browser clients, follow `demo/.agents/skills/browser-checks/SKILL.md`
and verify at least ordinary streaming, one client tool, file handling, and an
interrupt reconnect for the paths changed in units 07 and 08.

## Final Review Questions

- Can a reader trace one non-streaming run, one streaming run, and one interrupt
  resume without following duplicated state machines?
- Does every untyped external value become a validated type at its first owned
  boundary?
- Does package-native functionality replace local code only where the locked
  package offers the required stable guarantee?
- Is every extra database read, checkpoint scan, background task, or pooled
  connection tied to a documented invariant and regression test?
- Do streaming and non-streaming Responses still share one event/output state
  model?
- Are the Chainlit and Open WebUI adapters simple within their different host
  persistence models, without a shared UI framework?
- Can all internal API migrations be understood from the final code and docs
  without compatibility glue?

## Completion Record

When this audit passes, update the root progress table to mark every completed
unit, summarize the final design in `00-baseline-and-decisions.md`, and add:

- the final validation commands and results;
- external checks that were skipped and why;
- a short list of intentionally retained complex mechanisms;
- any new bounded follow-up work that is outside this refactor.

Do not delete this directory. It records the evidence and sequencing behind the
refactor and prevents a later cleanup from undoing durability safeguards.

## Outcome

Not started.
