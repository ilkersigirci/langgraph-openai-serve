# Reference

## OpenAI-Compatible API

Default prefix: `/v1`. Change it with `LGOS_OPENAI_API_PREFIX` or
`bind_openai_api(prefix=...)`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | List registered graph models and feature metadata. |
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
| `DEMO_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Upstream OpenAI-compatible base URL for LLM-backed demo graphs. |
| `DEMO_OPENAI_API_KEY` | `DUMMY` | Upstream API key for LLM-backed demo graphs. |
| `DEMO_OPENAI_MODEL` | `gpt-5.4-mini` | Upstream generation model for LLM-backed demo graphs. |
| `DEMO_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model used by the `lgos-rag` demo graph. |
| `DEMO_POSTGRES_URI` | `postgresql://lgos:lgos@localhost:5432/lgos` | Checkpoint database used by the interruptible demo graph. |
| `DEMO_CHAINLIT_OPENAI_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible demo API used by Chainlit. |
| `DEMO_CHAINLIT_HITL_MODEL` | `interruptible-approval` | Interrupt-enabled model selected by the Chainlit HITL demo. |
| `DEMO_CHAINLIT_UI_FILE` | `simple` | Chainlit target module; accepts `simple` or `hitl`. |
| `DEMO_CHAINLIT_LOGIN_TYPE` | `mock` | Browser login callback; accepts `mock` or `oauth`. |

Native Chainlit settings used by the demo:

| Setting | Default | Notes |
| --- | --- | --- |
| `DATABASE_URL` | required | PostgreSQL URL validated by the demo and used by Chainlit's official asyncpg data layer and migration command. |
| `CHAINLIT_AUTH_SECRET` | required | Secret used by Chainlit to sign login tokens; generate it with `uv run chainlit create-secret`. |
| `CHAINLIT_APP_ROOT` | `demo/ui/chainlit_ui` in `.env.example` | Directory containing the tracked Chainlit configuration and welcome Markdown. |
| `CHAINLIT_URL` | request origin | OAuth-only external origin used for callback URLs behind a reverse proxy. |
| `OAUTH_GENERIC_CLIENT_ID` | required for `oauth` | Client ID for Chainlit's generic OAuth provider. |
| `OAUTH_GENERIC_CLIENT_SECRET` | required for `oauth` | OAuth client secret; keep it outside source control. |
| `OAUTH_GENERIC_AUTH_URL` | required for `oauth` | OIDC authorization endpoint. |
| `OAUTH_GENERIC_TOKEN_URL` | required for `oauth` | OIDC token endpoint. |
| `OAUTH_GENERIC_USER_INFO_URL` | required for `oauth` | OIDC user-info endpoint. |
| `OAUTH_GENERIC_SCOPES` | required for `oauth` | Space-separated scopes requested by Chainlit. |
| `OAUTH_GENERIC_NAME` | `generic` | Optional provider ID shown by Chainlit and used in the callback path; the example uses `PocketID`. |
| `OAUTH_GENERIC_USER_IDENTIFIER` | `email` | Optional identifier claim; the PocketID example uses its stable `sub` claim. |

The generic variable names map to
[`GenericOAuthProvider`](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/chainlit/oauth_providers.py)
in the minimum supported Chainlit release.
The six settings marked as required for `oauth` match that provider's `env`
list; its name and user-identifier settings have framework defaults. No OAuth
setting is required in the default `mock` mode.
The demo `Settings` model validates the login selection. The separate
`ChainlitSettings` model validates the PostgreSQL URL, signing secret, and
mode-dependent generic-provider settings before the UI mounts. Chainlit
consumes the same native environment values at runtime.
The demo requires Chainlit 2.11.1 or newer. Because the official PostgreSQL
schema requires release-specific migrations, review Chainlit's migration
guidance whenever updating the lockfile and update the tracked `config.toml` and
SQL migrations when required.

## Public API

Use `LanggraphOpenaiServe` to bind OpenAI-compatible routes to a FastAPI app.
Use `GraphRegistry` to map OpenAI `model` names to `GraphConfig` values.
The registry must contain at least one graph. A missing or empty registry raises
`GraphRegistryError` during application configuration.

`GraphConfig` accepts:

- `graph`: compiled graph, sync factory, or async factory.
- `streamable_node_names`: node names whose streamed `AIMessageChunk` values are
  forwarded to clients.
- `features`: `GraphFeature` values that enable optional server behavior.
- `runtime_callbacks`: callbacks included in the LangGraph `RunnableConfig`.
- `request_to_input(request, messages)`: custom OpenAI request to graph input.
- `context_factory(request)`: build typed LangGraph runtime context passed via
  the graph's `context` argument.
- `output_to_text(output)`: custom graph output to assistant text.

Graphs that use `context_factory` should declare a compatible
`StateGraph(..., context_schema=Context)` and access the value from an injected
`Runtime[Context]`. Runtime context is separate from `RunnableConfig`:

| Value | LGOS/LangGraph path | Intended use |
| --- | --- | --- |
| Graph input | `graph.astream(input, ...)` | Messages and mutable workflow state. |
| Runtime context | `context_factory` → `context=` → `Runtime.context` | Immutable per-run application values and dependencies. |
| Runnable config | `config=` | Callbacks, tags, tracing, and other execution controls. |
| Checkpoint thread | `metadata.langgraph_thread_id` → `config["configurable"]["thread_id"]` | Load, save, interrupt, and resume checkpoint state. |

LGOS assembles runnable config from `runtime_callbacks` and, when present, the
checkpoint thread ID. There is intentionally no adapter for placing arbitrary
OpenAI request fields into `config["configurable"]`; use typed runtime context
for values consumed by nodes.

The same `features` set drives runtime behavior and the
versioned `langgraph_openai_serve.features` extension returned by `/v1/models`.
`GraphFeature.INTERRUPTS` enables and advertises the interrupt/resume flow.

Interrupt-enabled graphs must be compiled with a LangGraph checkpointer and
requests must include `metadata={"langgraph_thread_id": "<client-chat-id>"}`.
Use a durable checkpointer in production.

## Citation Events

Inside a graph node or tool, emit a citation with LangGraph's stream writer:

```python
from langgraph.config import get_stream_writer
from langgraph_openai_serve import citation_event

get_stream_writer()(
    citation_event(
        url="https://example.com/source",
        title="Example source",
        span=(10, 14),
    )
)
```

`span` uses Python's half-open convention, so `text[10:14]` returns the cited
text. LGOS converts it to OpenAI's inclusive `end_index` at the event boundary.
Use `citation_slice(annotation, text)` to validate received indices and convert
them back to a Python slice. Citation events must refer to the final rendered
assistant text.

See [Citation ownership and UI rendering](explanation/openai-compatibility.md#citation-ownership-and-ui-rendering)
for transport and client behavior.

The graph runner preserves LangGraph's native `CustomStreamPart` values,
including their subgraph namespace. Other event types remain available to direct
runner consumers through `langgraph_openai_serve.graph.runner`.

## Demo Models

`make run-demo-api` registers:

- `simple-graph-with-history` (runtime context enables conversation history and
  `metadata.system_prompt` overrides the default system prompt)
- `citation-events` (structured URL citations alongside portable Markdown)
- `simple-graph-no-history` (runtime context selects only the latest message and
  supports the same per-request system prompt)
- `lgos-rag`
- `custom-input-output-context`
- `advanced-mcp-tools`
- `complex-subgraphs`
- `interruptible-approval`

## Local Commands

=== "Setup"

    ```bash
    uv sync --frozen
    make help
    ```

=== "Run demos"

    ```bash
    make run-demo-api
    make run-demo-ui-chainlit
    make run-demo-ui-chainlit-hitl
    ```

=== "Test and lint"

    ```bash
    make -s test
    uv run --module pytest tests/path/to/test_file.py
    uv run --module pytest tests/path/to/test_file.py::test_name
    make -s lint
    make -s format
    ```

=== "Documentation"

    ```bash
    make doc-build
    make doc-serve
    ```

    `make doc-serve` serves the live preview on `http://localhost:7999` and
    watches for documentation changes.

::: langgraph_openai_serve
