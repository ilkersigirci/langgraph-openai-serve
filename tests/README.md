# Test Suite Guide

Keep test setup explicit and assertions focused on observable behavior.

## Test Roots

- `tests/` owns the installed package's API, graph runner, and utility tests.
- `tests/api/interrupt/` keeps the interrupt codec, response, HTTP contract,
  durable-state, and concurrency coverage together.
- Each project under `demo/` owns its tests and lockfile. Run all of them with
  `make test-demo`, or use `make test-demo-local` to overlay the current LGOS
  checkout into the demo API test run.
- Live demo integration tests are excluded from default pytest runs. Start the
  required services and use the dedicated root target, such as
  `make test-bifrost`, to select the `integration` marker explicitly.
- `tests/integration/test_demo_*` guards copied wire declarations and the
  distribution boundary without making demo runtime code import the parent
  package checkout.
- Fixtures stay in the nearest test-root or subdirectory `conftest.py`.
- Do not import from a `conftest.py`; request fixtures by name.

## Shared Graphs

- Put reusable graph schemas and graph factories under `tests/graph/support/`.
- Use `tests/graph/support/schemas.py` for shared state, input, and output
  schemas.
- Put graph factories in modules named for their behavior, such as
  `message.py` or `interrupt.py`.
- Import helpers from the concrete module, for example
  `tests.graph.support.message` or `tests.graph.support.interrupt`.
- Use factory functions instead of module-level compiled graph instances.
- Compile stateful or interruptible graphs inside the factory so every test gets
  a fresh graph and checkpointer.
- Keep one-off graph shapes inline when the graph behavior is the subject of the
  test and reusing it would hide the assertion intent.

## Fixtures

- Reserve `conftest.py` for fixtures and pytest hooks or configuration. Put
  importable builders, data, and assertion helpers in support modules.
- `tests/conftest.py` owns app, client, and fresh in-memory SQLite saver
  fixtures shared by package tests.
- Subdirectory `conftest.py` files may add local fixtures. Package graph tests
  import reusable builders from `tests.graph.support` modules.
- Prefer explicit fixture arguments over autouse fixtures. Reserve autouse for
  test-root invariants such as the isolated Chainlit application directory.
- Give each test only the fixtures it needs. Keep each state-changing resource
  creation and its cleanup together in a yield fixture or context manager.

## Async Tests

- Each standalone test root enables AnyIO's automatic test discovery and pins
  AnyIO 4.14 or newer. Do not add per-test or module-level AnyIO markers.
- Each standalone test root selects the `asyncio` backend in its top-level
  `conftest.py`. Keep the fixture function-scoped so async resources cannot
  leak between tests.
- Own async clients, savers, pools, and servers with yield fixtures or async
  context managers. Keep creation and teardown together.
- Use events plus bounded timeout scopes for concurrency assertions. Use
  `sleep_forever()` only when a task is intentionally waiting for cancellation;
  do not add timing sleeps to coordinate tests.

### Restricted Sandbox Stalls

Some coding-agent sandboxes deny `send()` on asyncio's internal Unix socketpair
with `EPERM`. Executor and AnyIO worker completions then cannot wake the event
loop, so LangChain callback or Chainlit ASGI tests may appear to hang in
`selector.select()` while their worker thread is idle. If
`pytest -o faulthandler_timeout=5` shows that pattern, stop the run and request
user approval to rerun the exact test command through the agent tool's
unsandboxed or escalated-execution mechanism. Explain that the sandbox blocks
asyncio worker-thread wakeups. If approval is denied, report the environment
limitation. Do not add sleeps, heartbeats, or production changes; an incidental
timer only masks the environment failure.

## Test Shape

- Name tests for the behavior and expected outcome.
- Keep arrange, act, and assert phases visible; hide only repeated plumbing.
- Parametrize repeated input/output cases. Add explicit IDs when pytest's
  generated IDs would be unclear.
- Assert public results directly instead of duplicating implementation details.
- Keep integration coverage for graph wiring, but unit-test edge cases at the
  narrowest stable boundary.
- Enable live logs only while diagnosing with `--log-cli-level=INFO`; normal
  runs rely on pytest's failure-time log capture.

## Runner And API Tests

- Runner tests should exercise graph execution behavior directly through
  `run_langgraph` or `run_langgraph_stream`.
- API tests should exercise HTTP/OpenAI-client behavior through the FastAPI or
  OpenAI client fixtures.
- Use `AsyncOpenAI` over HTTPX's ASGI transport for OpenAI contract tests. Use
  the raw HTTP client only for wire-format and host-application assertions.
- If the same graph shape is needed in both layers, define it once in
  `tests/graph/support/` and call the factory from each test.

## Stateful LangGraph Tests

- Graphs with `features={GraphFeature.INTERRUPTS}` must use a fresh
  `AsyncSqliteSaver.from_conn_string(":memory:")` checkpointer and a fresh
  `InMemoryRunCoordinator` per test. Register the coordinator on
  `GraphConfig.run_coordinator`.
- The checkpointer used by an interrupt graph must implement asynchronous state
  reads, checkpoint writes, pending writes, and `adelete_thread`;
  configuration-error tests should make whichever
  missing capability they exercise explicit.
- Persistence tests must use a `tmp_path` SQLite file, close the first
  checkpointer, and recreate the graph with a reopened checkpointer before
  resuming.
- Initial interrupt requests need no metadata. Tests for caller-owned
  idempotency should pass a non-nil UUID as
  `metadata.langgraph_run_id`; invalid or reused UUID cases should remain
  separate assertions.
- Resume helpers must copy the complete assistant message with all original
  `tool_calls`, then append exactly one JSON `{"resume": ...}` tool result for
  every call. Parallel interrupts must be answered together and matched by
  `tool_call_id`; never synthesize only the visible payload or select the first
  call.
- Cover the durable lifecycle at the API boundary: an initial retry with the
  same caller run UUID re-emits the pending batch, stale or repeated resumes
  return a conflict without re-executing work, concurrent resumes are
  single-flight, and terminal completion deletes the checkpoint lineage.
- Tests that intentionally verify missing checkpointer behavior should use an
  uncheckpointed graph factory so the failure setup is obvious.

Real PostgreSQL tests use unique persistence scopes, close and recreate the
runtime, and delete their exact checkpoints or Store documents in teardown.
Keep them excluded from default runs and invoke them through
`make -C demo test-postgres` so ordinary and parallel unit runs never share an
external database.

## Client Event Tests

- Graphs that emit client events must declare
  `features={GraphFeature.CLIENT_EVENTS}`.
- Streaming requests must also opt in with
  `metadata.langgraph_stream_events="v1"`; test the feature declaration and
  request opt-in as independent gates.
