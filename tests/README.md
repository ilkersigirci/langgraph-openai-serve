# Test Suite Guide

Keep test setup explicit, but do not duplicate reusable LangGraph construction.

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
- Root `tests/conftest.py` owns app/client fixtures shared across the suite.
- Subdirectory `conftest.py` files may add local fixtures, but should import
  graph builders from `tests.graph.support.message`.
- Prefer explicit fixture arguments over autouse fixtures.

## Runner And API Tests

- Runner tests should exercise graph execution behavior directly through
  `run_langgraph` or `run_langgraph_stream`.
- API tests should exercise HTTP/OpenAI-client behavior through the FastAPI or
  OpenAI client fixtures.
- If the same graph shape is needed in both layers, define it once in
  `tests/graph/support/` and call the factory from each test.

## Stateful LangGraph Tests

- Interrupt-enabled graphs must use a fresh
  `AsyncSqliteSaver.from_conn_string(":memory:")` checkpointer per test.
- Persistence tests must use a `tmp_path` SQLite file and reopen the saver or
  interpreter before resuming.
- Requests for interrupt-enabled graphs must include
  `metadata.langgraph_thread_id`.
- Tests that intentionally verify missing checkpointer behavior should use an
  uncheckpointed graph factory so the failure setup is obvious.
