# 07 — Chainlit Client

- Status: **Waiting for core work**
- Priority: **P2**
- Dependencies: **01, 05, 06**

## Objective

Make the Chainlit adapters easier to reason about by separating durable
interrupt data from Chainlit lifecycle and rendering calls. Keep the client on
the standard Responses API and preserve recovery after process or browser
restarts.

## Assessment

`lgos_chainlit/hitl.py` currently owns several different concerns in one module:

- Chainlit hook registration and session/task lifecycle;
- creation and replay of OpenAI Responses requests;
- a versioned, persisted interrupt ledger;
- conversion of restored `ThreadDict` data back into typed continuation state;
- prompt rendering, custom element props, and user decisions;
- Chainlit hydration and persisted-step workarounds.

The ledger is necessary. It persists the complete function-call batch before
showing a prompt, allowing a reconnect to resume the same standard
`previous_response_id` request without private Chainlit data-layer access. Its
completed marker also prevents a finished batch from being reopened. The
problem is that pure ledger validation and host actions are interleaved, making
both harder to test.

The ordinary `simple.py` adapter has a smaller version of the same orchestration
problem: one response turn, streamed presentation, client-owned function tools,
and file handling meet in `_response_message()`. Its existing Responses helpers
already provide the right shared protocol boundary.

This design is grounded in Chainlit's public host contract: `on_chat_resume`
receives a `ThreadDict`, and `Message.metadata` is explicitly persisted by the
data layer. Recheck those public APIs and the locked 2.12.0 implementation
before moving any host operation.

## Files In Scope

- `demo/ui/chainlit_ui/src/lgos_chainlit/hitl.py`
- `demo/ui/chainlit_ui/src/lgos_chainlit/simple.py`
- `demo/ui/chainlit_ui/src/lgos_chainlit/utils/responses.py`
- a single new ledger module if extraction makes `hitl.py` materially smaller
- focused Chainlit tests and README text affected by the final structure

Keep reusable general Chainlit behavior in the separately released
`chainlit-utils` project as required by `demo/AGENTS.md`. The LGOS Responses
ledger and interrupt protocol remain in this repository because they are
application-specific.

## Implementation Steps

1. Complete units 01 and 05 first, then use their final strict input models and
   Responses helpers. Do not preserve an interim client shape with an adapter.
2. Extract a small pure ledger codec from `hitl.py`. It should own the schema
   version, pending/completed values, strict metadata validation, selection of
   the newest valid pending entry, and conversion to immutable continuation
   data. It must not import Chainlit UI components or access `cl.user_session`.
3. Keep the persisted Chainlit message reference beside the decoded value in a
   small host-side value object. Persistence, message updates, element removal,
   and session storage remain explicit Chainlit operations.
4. Keep the public metadata route. Do not add a private database query or a
   storage abstraction solely to avoid parsing `ThreadDict`.
5. Leave the hydration scheduler, reused-step identity, timestamp handling, and
   persisted-element cleanup next to the Chainlit calls that require them. Add
   short comments explaining the host invariant where the code alone cannot.
6. Turn `handle_message()` and `resolve_interrupts()` into a clear sequence:
   build one request, persist every returned interrupt batch, prompt for every
   item, mark that batch completed, and submit one standard resume request.
   Prefer small functions for those stages over a coordinator class.
7. Keep the rule that all calls in a parallel interrupt batch are answered
   together. Cancellation while a prompt is open must leave the persisted
   pending ledger intact for the next resume.
8. In `simple.py`, separate result classification from host rendering only if a
   pure helper is shared by the streamed and non-streamed paths. Continue using
   the official SDK stream iterator and the existing file, MCP, and response
   helpers; do not create another response accumulator.
9. Apply strict validation to client-owned persisted values at decode time.
   Invalid or future ledger versions should fail closed and produce the current
   visible recovery error without sending a guessed resume.
10. Replace tests that mock a chain of internal helpers with direct pure-codec
    cases plus the smallest hook-level tests needed to prove persistence order,
    reconnect behavior, rendering, and cancellation.

## Required Behavior

- The complete interrupt batch is durably written before its first prompt is
  displayed.
- A completed ledger is written before terminal output makes the run appear
  finished.
- Reconnect finds the newest pending ledger, removes stale persisted controls,
  and opens exactly one live prompt flow.
- A user cannot start a second request while a pending batch is unresolved.
- Prompt cancellation leaves enough durable state to recover later.
- Every resume uses standard Responses input and `previous_response_id`; no
  Chainlit-only field reaches `/v1/responses`.
- Ordinary chat retains streaming text, tool execution, MCP, citations, file
  upload/display, settings, and incomplete/refusal behavior.

## Validation

```bash
cd demo/ui/chainlit_ui
uv run --locked pytest tests/test_chainlit_hitl.py tests/test_thread_resume.py
uv run --locked pytest tests/test_chainlit_responses.py tests/test_chainlit_files.py tests/test_chainlit_mcp.py
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
uv run --locked ty check src
cd ../../..
cd demo && just test --editable
```

Use the browser-check skill from `demo/AGENTS.md` for a final live reconnect and
parallel-interrupt check when the demo stack is available. The unit tests do not
fully reproduce Chainlit hydration timing.

Primary host references are [Chainlit `on_chat_resume`](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-resume),
[persisted message metadata](https://docs.chainlit.io/api-reference/message),
and the [2.12.0 release](https://github.com/Chainlit/chainlit/releases/tag/2.12.0).

## Stop Conditions

- If an extraction needs to wrap most Chainlit types in duplicate local types,
  keep the host operation in `hitl.py` and extract only the JSON-safe ledger
  value and codec.
- If moving behavior to `chainlit-utils` is useful, treat its publication and
  version update as a separate change. Do not add a local path dependency or
  edit this demo's lockfile incidentally.

## Reviewed And Retained

`utils/chat_settings.py` is a linear host flow: publish safe default widgets,
retrieve model metadata, add supported dynamic widgets through
`chainlit-utils`, and store the defaults needed for later request metadata. Its
branches reflect degraded model discovery and distinct UI capabilities. Keep it
unless the final registration metadata removes a branch.

The authentication modules are outside this refactor. Their explicit PKCE/OIDC
callback, browser-local session claim, delegated token storage, revocation, and
request/socket credential selection are security boundaries with their own
tests. Do not fold them into the chat or interrupt cleanup.

## Outcome

Not started.
