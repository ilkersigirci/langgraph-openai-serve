# 07 — Chainlit Interrupt Ledger

- Status: **Complete**
- Priority: **P2**
- Dependencies: **05, 06**

## Objective

Separate durable interrupt-ledger data from Chainlit lifecycle and rendering
calls. Keep the client on the standard Responses API and preserve recovery after
process or browser restarts. Ordinary chat cleanup is not part of this unit
unless the final core contract forces a caller change.

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

The ordinary `simple.py` adapter combines one response turn, streamed
presentation, client-owned function tools, and file handling in
`_response_message()`, but its existing Responses helpers already provide the
right shared protocol boundary. Review it for required caller changes only; its
size alone is not evidence for another extraction.

This design is grounded in Chainlit's public host contract: `on_chat_resume`
receives a `ThreadDict`, and `Message.metadata` is explicitly persisted by the
data layer. Recheck those public APIs and the locked 2.12.0 implementation
before moving any host operation.

## Files In Scope

- `demo/ui/chainlit_ui/src/lgos_chainlit/hitl.py`
- `demo/ui/chainlit_ui/src/lgos_chainlit/simple.py` and
  `utils/responses.py` only for changes required by the final core wire contract
- a single new ledger module if extraction makes `hitl.py` materially smaller
- focused Chainlit tests and README text affected by the final structure

Keep reusable general Chainlit behavior in the separately released
`chainlit-utils` project as required by `demo/AGENTS.md`. The LGOS Responses
ledger and interrupt protocol remain in this repository because they are
application-specific.

## Implementation Steps

1. Complete units 05 and 06 first, then verify the client-generated requests
   against the final documented Responses subset. The independently deployed UI
   must continue using official OpenAI SDK request/output types; do not import
   LGOS's server-internal Pydantic request models.
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
8. Leave `simple.py` structurally unchanged unless a required wire migration
   reveals a pure result-classification helper shared by streamed and
   non-streamed paths. Continue using the official SDK stream iterator and the
   existing file, MCP, and response helpers; do not create another response
   accumulator.
9. Apply strict validation to client-owned persisted values at decode time.
   Invalid or future ledger versions should fail closed and produce the current
   visible recovery error without sending a guessed resume. Treat Chainlit's
   surrounding `ThreadDict` as an extensible host object: read and validate only
   consumed fields rather than rejecting unrelated host-added fields.
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

Completed on 2026-09-16.

- The versioned JSON-safe ledger codec now lives in
  `lgos_chainlit/interrupt_ledger.py`. It owns pending and completed values,
  strict decode-time validation, newest-ledger selection, and frozen, tuple-
  backed continuation values built from the official OpenAI SDK output type.
  It has no Chainlit UI or session dependency.
- `hitl.py` retains the host boundary: public message-metadata persistence,
  `Message.from_dict()` restoration, reused step identity, timestamp repair,
  stale-element removal, hydration scheduling, and user-session state. A small
  host value pairs the restored Chainlit message with its decoded continuation.
- The interrupt flow writes the complete function-call batch before the first
  prompt, collects every decision in a parallel batch, sends one standard
  `function_call_output` request with `previous_response_id`, and writes the
  completed marker before terminal output. Prompt cancellation or failure
  leaves the pending marker recoverable, and a pending batch blocks a second
  request.
- Invalid, unknown, or future LGOS-owned ledger fields and duplicate call IDs
  fail closed. A resumed thread schedules the existing visible recovery error
  after Chainlit hydration and never guesses a continuation. Unrelated fields
  in Chainlit's surrounding host step and metadata remain accepted.
- `simple.py` and the shared Responses helpers required no caller or wire
  change after units 05 and 06. Their streaming, files, MCP, citations,
  incomplete, and refusal behavior remained covered by the ordinary-client
  regression suite. Existing product documentation already describes the
  retained public behavior, so no product-documentation change was required.

Locked-version verification used the Chainlit UI's `uv.lock`,
`uv tree --locked`, and installed source. The resolved relevant versions were
Chainlit 2.12.0, chainlit-utils 0.1.0, OpenAI Python 2.46.0, Pydantic 2.13.4,
and AnyIO 4.14.2. The implementation was checked against these primary
references:

- [Chainlit `on_chat_resume`](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-resume)
- [Chainlit persisted message metadata](https://docs.chainlit.io/api-reference/message)
- [Chainlit 2.12.0 message implementation](https://github.com/Chainlit/chainlit/blob/2.12.0/backend/chainlit/message.py)
- [Chainlit 2.12.0 resume ordering](https://github.com/Chainlit/chainlit/blob/2.12.0/backend/chainlit/socket.py)
- [OpenAI Python 2.46.0 function-call output type](https://github.com/openai/openai-python/blob/v2.46.0/src/openai/types/responses/response_function_tool_call.py)
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)

| Validation | Result |
| --- | --- |
| `uv run --locked pytest tests/test_chainlit_hitl.py tests/test_thread_resume.py` | 24 passed |
| `uv run --locked pytest tests/test_chainlit_responses.py tests/test_chainlit_files.py tests/test_chainlit_mcp.py` | 26 passed |
| Chainlit Ruff check and format check | Pass |
| `uv run --locked ty check src` | Pass |
| `cd demo && just test --editable` | API 111 passed, Files 12 passed, Chainlit 126 passed, Open WebUI 120 passed |
| `just demo/check --editable` | All demo tests, lint, formatting, type checks, and Compose config checks passed |
| Live Chainlit browser check | Reconnect reopened one pending flow; two parallel decisions were submitted together in one resume request; terminal output completed the ledger |

The live check used an isolated browser session and a disposable standard
Responses mock because the checked-in demo graph emits only one interrupt at a
time. The mock rejected partial parallel resumes and recorded both outputs in
the accepted request. The running demo was restored to its original `simple`
configuration afterward.
