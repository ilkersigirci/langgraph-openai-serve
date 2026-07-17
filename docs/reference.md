# Reference

## OpenAI-Compatible API

Default prefix: `/v1`. Change it with `LGOS_OPENAI_API_PREFIX` or
`bind_openai_api(prefix=...)`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | List standard summaries for registered graph models. |
| `GET` | `/v1/models/{model}` | Retrieve one model with optional LGOS discovery metadata. |
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
| `DEMO_CHAINLIT_INFERENCE__BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible inference API used by Chainlit. |
| `DEMO_CHAINLIT_INFERENCE__API_KEY` | `DUMMY` | Inference API or gateway key used by Chainlit. |
| `DEMO_CHAINLIT_INFERENCE_MODEL_PREFIX` | empty | Optional proxy namespace prepended to inference model IDs, such as Bifrost's `openai/`. |
| `DEMO_CHAINLIT_DISCOVERY__BASE_URL` | unset | Optional direct LGOS or documented pass-through URL used for model listing and retrieval; an omitted discovery endpoint reuses inference. |
| `DEMO_CHAINLIT_DISCOVERY__API_KEY` | unset | Required whenever an explicit discovery base URL is configured. |
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
The registry must contain at least one graph. Pydantic rejects empty registries
and model IDs that cannot be addressed as one URL path segment. Registry keys
are read-only after validation; use `registry.register(model_id, config)` to add
or replace a graph.

`GraphConfig` accepts:

- `graph`: compiled graph, sync factory, or async factory.
- `streamable_node_names`: node names whose streamed `AIMessageChunk` values are
  forwarded to clients.
- `features`: `GraphFeature` values that enable optional server behavior.
- `client_settings`: explicit public `ClientSettings` model class advertised by
  model retrieval.
- `runtime_callbacks`: callbacks included in the LangGraph `RunnableConfig`.
- `request_to_input(request, messages)`: custom OpenAI request to graph input.
- `context_factory(request, client_settings)`: compose the final typed LangGraph
  runtime context from server-owned values and optional validated public settings.
- `output_to_text(output)`: custom graph output to assistant text.

When both are configured, LGOS validates the public settings first and passes
them to `context_factory`. Without a factory, the validated settings instance is
the runtime context. When the resolved graph declares a `context_schema`, LGOS
validates every non-null final context against it. Graphs should access context
from an injected `Runtime[Context]`. Runtime context is separate from
`RunnableConfig`:

| Value | LGOS/LangGraph path | Intended use |
| --- | --- | --- |
| Graph input | `graph.astream(input, ...)` | Messages and mutable workflow state. |
| Runtime context | public settings → optional `context_factory` → `context=` → `Runtime.context` | Immutable per-run application values and dependencies. |
| Runnable config | `config=` | Callbacks, tags, tracing, and other execution controls. |
| Checkpoint thread | `metadata.langgraph_thread_id` → `config["configurable"]["thread_id"]` | Load, save, interrupt, and resume checkpoint state. |

LGOS assembles runnable config from `runtime_callbacks` and, when present, the
checkpoint thread ID. There is intentionally no adapter for placing arbitrary
OpenAI request fields into `config["configurable"]`; use typed runtime context
for values consumed by nodes.

The same `features` set drives runtime behavior and the versioned
`langgraph_openai_serve.features` extension returned by
`GET /v1/models/{model}`. `GraphFeature.INTERRUPTS` enables and advertises the
interrupt/resume flow.

### Runtime Settings

Subclass `ClientSettings` to publish only fields deliberately selected by the
server author. LGOS never inspects or publishes the LangGraph context schema:

```python title="Public settings model"
from pydantic import Field

from langgraph_openai_serve import ClientSettings


class PublicSettings(ClientSettings):
    use_history: bool = Field(default=True, title="Use conversation history")
```

Pass this model as `GraphConfig.client_settings` and use it as the graph's context
schema when it is the complete runtime context. Every public field must have a
default and should keep the inherited `ClientSettings` validation behavior.

All public fields travel together as compact JSON text in the
`metadata.langgraph_runtime_settings` string. Clients omit values equal to the advertised
defaults. System instructions remain ordinary OpenAI messages and are
independent of `ClientSettings`; native Chat Completions fields keep their
standard request semantics.

LGOS validates defaults when the graph is registered and validates settings on
every request. Without `context_factory`, the settings become `Runtime.context`.
A factory can instead combine them with server-derived identity, authorization,
database clients, and other dependencies.

The serialized descriptor appears only on model retrieval as
`langgraph_openai_serve.client_settings`, with independent `schema_version`,
`json_schema`, and `defaults` fields. All client settings use the fixed
`metadata.langgraph_runtime_settings` envelope.

See [Configure LangGraph Runtime Settings](how-to-guides/langgraph-runtime-settings.md)
for the runtime settings flow, and
[Runtime Settings](explanation/openai-compatibility.md#runtime-settings)
for the request lifecycle and integration boundaries.

The demo Chainlit adapter deliberately renders only direct top-level scalar
properties with concrete defaults: booleans become switches, inline string
enums become selects, and strings become text inputs. The server remains the
validation authority for constraints that Chainlit widgets cannot express.

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

- `simple-graph` (Chainlit can change conversation history and intended audience
  through the discovered runtime settings)
- `citation-events` (structured URL citations alongside portable Markdown)
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
    make test-bifrost
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
