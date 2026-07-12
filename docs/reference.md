# Reference

## OpenAI-Compatible API

Default prefix: `/v1`. Change it with `LGOS_OPENAI_API_PREFIX` or
`bind_openai_api(prefix=...)`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | List registered graph model names. |
| `POST` | `/v1/chat/completions` | Run a graph through OpenAI chat completions. |
| `GET` | `/v1/health` | Health check. |

FastAPI docs for the mounted OpenAI app are disabled by default. Set
`LGOS_OPENAI_API_DOCS_ENABLED=true` to expose `{prefix}/docs`, `{prefix}/redoc`,
and `{prefix}/openapi.json`.

## Settings

Package settings:

| Setting | Default | Notes |
| --- | --- | --- |
| `LGOS_OPENAI_API_PREFIX` | `/v1` | Must start with `/`; trailing slash is normalized. |
| `LGOS_OPENAI_API_DOCS_ENABLED` | `false` | Enables docs only for the mounted OpenAI app. |

Demo-only settings (read by `demo/api/settings.py`):

| Setting | Default | Notes |
| --- | --- | --- |
| `DEMO_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Upstream OpenAI-compatible model base URL for simple demo graphs. |
| `DEMO_OPENAI_API_KEY` | `DUMMY` | Upstream API key for simple demo graphs. |
| `DEMO_OPENAI_MODEL` | `gpt-5.4-mini` | Upstream model name for simple demo graphs. |
| `DEMO_POSTGRES_URI` | `postgresql://lgos:lgos@localhost:5432/lgos` | Checkpoint database used by the interruptible demo graph. |
| `DEMO_CHAINLIT_OPENAI_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible demo API used by Chainlit. |
| `DEMO_CHAINLIT_HITL_MODEL` | `interruptible-approval` | Interrupt-enabled model selected by the Chainlit HITL demo. |

## Public API

Use `LanggraphOpenaiServe` to bind OpenAI-compatible routes to a FastAPI app.
Use `GraphRegistry` to map OpenAI `model` names to `GraphConfig` values.
The registry must contain at least one graph. A missing or empty registry raises
`GraphRegistryError` during application configuration.

`GraphConfig` accepts:

- `graph`: compiled graph, sync factory, or async factory.
- `streamable_node_names`: node names whose streamed `AIMessageChunk` values are
  forwarded to clients.
- `request_to_input(request, messages)`: custom OpenAI request to graph input.
- `context_factory(request)`: custom LangGraph runtime context.
- `output_to_text(output)`: custom graph output to assistant text.
- `interrupts_enabled`: enables OpenAI tool-call based LangGraph resume flow.

Interrupt-enabled graphs must be compiled with a LangGraph checkpointer and
requests must include `metadata={"langgraph_thread_id": "<client-chat-id>"}`.
Use a durable checkpointer in production.

## Demo Models

`make run-demo-api` registers:

- `simple-graph-with-history`
- `simple-graph-no-history`
- `custom-input-output-context`
- `advanced-mcp-tools`
- `complex-subgraphs`
- `interruptible-approval`

## Local Commands

```bash
uv sync --frozen
make help
make run-demo-api
make run-demo-ui-chainlit
make run-demo-ui-chainlit-hitl
make -s test
uv run --module pytest tests/path/to/test_file.py
uv run --module pytest tests/path/to/test_file.py::test_name
make -s lint
make -s format
make doc-build
```

::: langgraph_openai_serve
