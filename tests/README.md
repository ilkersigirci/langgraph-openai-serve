# Test Suite Guide

Keep test setup explicit and assertions focused on observable behavior.

## Test Roots

- `tests/` owns the installed package's API, graph runner, and utility tests.
- `demo/tests/` owns demo applications, graphs, and UI adapter tests.
- The two test roots keep separate fixtures, even when small fixtures are
  intentionally duplicated.
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

- Use `conftest.py` for pytest fixtures only.
- `tests/conftest.py` owns app/client fixtures shared by package tests.
- `demo/tests/conftest.py` owns fixtures used only by demo tests, when needed.
- Subdirectory `conftest.py` files may add local fixtures. Package graph tests
  import reusable builders from `tests.graph.support` modules.
- Prefer explicit fixture arguments over autouse fixtures.

## Test Shape

- Name tests for the behavior and expected outcome.
- Keep arrange, act, and assert phases visible; hide only repeated plumbing.
- Parametrize repeated input/output cases and give each case a meaningful ID.
- Assert public results directly instead of duplicating implementation details.
- Keep integration coverage for graph wiring, but unit-test edge cases at the
  narrowest stable boundary.
- Enable live logs only while diagnosing with `--log-cli-level=INFO`; normal
  runs rely on pytest's failure-time log capture.
- Defer Chainlit imports until the test redirects its application root to
  `tmp_path`, and keep Chainlit pointed away from the repository `.env`.

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

- Interrupt-enabled graphs must use a fresh
  `AsyncSqliteSaver.from_conn_string(":memory:")` checkpointer per test.
- Persistence tests must use a `tmp_path` SQLite file, close the first
  checkpointer, and recreate the graph with a reopened checkpointer before
  resuming.
- Requests for interrupt-enabled graphs must include
  `metadata.langgraph_thread_id`.
- Tests that intentionally verify missing checkpointer behavior should use an
  uncheckpointed graph factory so the failure setup is obvious.
