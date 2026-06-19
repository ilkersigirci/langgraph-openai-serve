Guidance for coding agents working in this repository. Keep this file short; use
human docs for explanations and examples.

## Start Here

- Read `README.md` for the project purpose and the shortest runnable example.
- Read `docs/index.md` to navigate the full documentation.
- Use `docs/explanation/architecture.md` for request flow, graph execution,
  routing, or schema changes.

## Repository Map

- `src/langgraph_openai_serve/` contains the package.
- `src/langgraph_openai_serve/openai_server.py` wires FastAPI routers to a graph
  registry.
- `src/langgraph_openai_serve/api/` contains OpenAI-compatible HTTP endpoints,
  schemas, and services.
- `src/langgraph_openai_serve/graph/` contains graph registration, input/context
  adaptation, and LangGraph execution helpers.
- `src/langgraph_openai_serve/utils/` contains shared conversion utilities.
- `demo/` contains runnable API and UI examples.
- `tests/` mirrors package behavior with pytest coverage.

## Local Setup

This project uses Python 3.11+ and `uv`.

```bash
uv sync --frozen
```

Use the Makefile wrappers when possible because they match the documented
workflow:

```bash
make help
make run-demo-api
make -s test
make -s lint
```

The demo API serves OpenAI-compatible routes at `http://localhost:8000/v1`.

## Common Tasks

- Run all tests: `make -s test`
- Run one test file: `uv run --module pytest tests/path/to/test_file.py`
- Run one test by node id: `uv run --module pytest tests/path/to/test_file.py::test_name`
- Lint without modifying files: `make -s lint`
- Format package code: `make -s format`

## Working Rules

- Keep changes scoped to the package, demo, tests, or docs area affected by the
  request.
- Do not update dependencies or regenerate `uv.lock` unless the task requires it.
- Do not commit secrets. Treat `.env` as local state and use `.env.example` for
  documented variables.
- Prefer adding or updating focused tests for behavior changes.
- For API behavior, check route/service code and the graph runner path.
- For graph adapters, check the demo before changing public APIs.
