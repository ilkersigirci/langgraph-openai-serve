# Getting Started

This tutorial uses the repository demo app. It serves several LangGraph graphs
through the OpenAI-compatible `/v1` interface.

## Prerequisites

- Python 3.11 or newer
- `uv`
- PostgreSQL (the included Compose service requires Docker)
- An OpenAI-compatible upstream model only if you call the LLM-backed graphs

!!! tip "Start without an upstream model"

    The custom adapter, citation, nested subgraph, and mock MCP demo graphs do
    not require real API keys.

## Run The Demo API

```bash title="Start PostgreSQL and the API"
docker compose up -d lgos-postgres
make run-demo-api
```

??? info "Demo environment settings"

    The demo reads `DEMO_POSTGRES_URI` and defaults to
    `postgresql://lgos:lgos@localhost:5432/lgos`, which matches the Compose
    service.

    LLM-backed graphs additionally read `DEMO_OPENAI_BASE_URL`,
    `DEMO_OPENAI_API_KEY`, and `DEMO_OPENAI_MODEL`. The `lgos-rag` graph also
    reads `DEMO_OPENAI_EMBEDDING_MODEL`. These settings and their LangChain
    agent dependencies belong to the demo and are not installed as part of the
    library's runtime dependencies.

The base URL is `http://localhost:8000/v1`.

Inspect registered graphs:

```bash
curl http://localhost:8000/v1/models
```

The demo model names are listed in [Reference](../reference.md#demo-models).

## Call A Graph

```python title="Call a registered graph"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

response = client.chat.completions.create(
    model="custom-input-output-context",
    messages=[{"role": "user", "content": "Show me the custom adapter."}],
    user="demo-user",
)

print(response.choices[0].message.content)
```

Try the citation custom-event graph:

```python
response = client.chat.completions.create(
    model="citation-events",
    messages=[{"role": "user", "content": "Show me a cited answer."}],
)

print(response.choices[0].message.content)
print(response.choices[0].message.annotations)
```

The deterministic answer combines portable Markdown resources with structured
citations. See [Citation Events](../reference.md#citation-events) for the graph
helper and
[Citation ownership and UI rendering](../explanation/openai-compatibility.md#citation-ownership-and-ui-rendering)
for transport and client behavior.

Ask the RAG graph about this project's Markdown documentation with real-time
token streaming:

```python
stream = client.chat.completions.create(
    model="lgos-rag",
    messages=[{"role": "user", "content": "How does LGOS streaming work?"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

`lgos-rag` follows an agentic RAG loop: it decides when retrieval is needed,
searches chunked documentation through a tool, grades relevance, and rewrites
once when retrieval misses. Social and conversation-history turns skip
retrieval. Grounded answers use exact source-backed Markdown links rather than
citation annotations; source-provided image Markdown is preserved, while audio
and video remain ordinary links.

Try the async mock MCP graph:

```python
response = client.chat.completions.create(
    model="advanced-mcp-tools",
    messages=[{"role": "user", "content": "What is the weather in Istanbul?"}],
)
```

Try the deterministic nested subgraph graph:

```python
response = client.chat.completions.create(
    model="complex-subgraphs",
    messages=[
        {
            "role": "user",
            "content": "Show OpenAI adapter streaming with nested subgraphs.",
        }
    ],
    user="demo-user",
)

print(response.choices[0].message.content)
```

## Demo Files

- `demo/api/app.py`: registers graph names as OpenAI model names.
- `demo/api/graphs/simple.py`: a single-node message graph whose runtime context
  controls conversation history and intended audience. It publishes those safe
  fields explicitly through model retrieval and keeps arbitrary system prompt
  text out of runtime configuration.
- `demo/api/graphs/lgos_rag.py`: agentic RAG over every Markdown file in
  `docs/`, with relevance grading, bounded query rewriting, streamed generation,
  and grounded answers.
- `demo/api/graphs/custom_io.py`: input, output, and context adapters.
- `demo/api/graphs/advanced_mcp.py`: async factory with mock MCP-style tools.
- `demo/api/graphs/complex_subgraphs.py` and `demo/api/graphs/subgraphs/`:
  router-selected subgraphs with streamed fake chat model output.
- `demo/api/graphs/interruptible.py`: interrupt and resume graph persisted in
  PostgreSQL by the demo application.
- `demo/api/graphs/citations.py`: custom citation event and OpenAI annotation
  demo.
- `demo/api/settings.py`: separate Pydantic settings models for demo application
  values and native Chainlit persistence/authentication configuration.
- `demo/ui/chainlit_ui/database.py` and `migrations/`: versioned Chainlit
  PostgreSQL schema setup.
- `demo/ui/chainlit_ui/hitl.py`: Chainlit interrupt approval demo.
- `demo/ui/openwebui/openwebui_pipe.py`: Open WebUI manifold Pipe that discovers
  registered graph models and bridges interrupt approval.

## Run The Chainlit UI

Chainlit remains an ordinary OpenAI client of the demo API. Its UI state is
separate: the official Chainlit data layer stores authenticated users, threads,
steps, and feedback in PostgreSQL. A shared mock user handles browser login by
default; PocketID OAuth is optional.

Create the local environment file once:

```bash
cp .env.example .env
uv run chainlit create-secret
```

Put the generated value in `CHAINLIT_AUTH_SECRET`; it is used only to sign
Chainlit's session tokens. `DATABASE_URL` configures the persistent Chainlit
data layer and migration command.

=== "Mock login (default)"

    Keep `DEMO_CHAINLIT_LOGIN_TYPE=mock`. No OAuth variables or OAuth client are
    needed. The login form accepts any non-empty username and password and maps
    every session to the shared `demo-user` identity.

=== "PocketID OAuth"

    Set `DEMO_CHAINLIT_LOGIN_TYPE=oauth`, uncomment the generic OAuth variables
    in `.env`, and fill in the client ID, client secret, and PocketID endpoint
    URLs. `OAUTH_GENERIC_USER_IDENTIFIER=sub` uses PocketID's stable subject as
    the Chainlit user identifier.

    Configure the PocketID OIDC client callback as
    `http://localhost:5000/auth/oauth/PocketID/callback` for the local demo. For
    a deployed UI, use
    `${CHAINLIT_URL}/auth/oauth/${OAUTH_GENERIC_NAME}/callback` and set
    `CHAINLIT_URL` to the external HTTPS origin behind a reverse proxy.

=== "Local processes"

    Start PostgreSQL, then run the API and UI in separate terminals:

    ```bash
    docker compose up -d lgos-postgres
    make run-demo-api
    ```

    ```bash
    make run-demo-ui-chainlit
    ```

=== "Compose"

    ```bash
    docker compose up -d lgos-chainlit
    ```

Open `http://localhost:5000` and use the selected login. Both UI run targets
apply pending schema migrations before starting. Opening a stored thread
restores its native Chainlit role/content transcript and continues with the same
login identity. Structured tool-call state is assembled separately when the
HITL client sends an interrupt-resume request.
After a profile is selected, Chainlit retrieves that model, converts the
advertised JSON Schema into Chat Settings, and merges only saved values that
remain valid under the current schema. If detailed LGOS metadata is unavailable,
the chat continues with server defaults. Proxy-specific inference and discovery
settings are covered in
[Configure an OpenAI Proxy](../how-to-guides/openai-proxy.md).

!!! warning "Production credentials and origins"

    Mock mode gives every visitor the same identity and is not authentication.
    Before deployment, select OAuth, replace the demo PostgreSQL credentials,
    keep the OAuth client secret and Chainlit signing secret outside source
    control, and update `allow_origins` in the tracked Chainlit `config.toml` to
    the deployed HTTPS origin. File uploads remain disabled until an S3, GCS, or
    Azure storage provider is configured.

See Chainlit's current guidance for
[password callbacks](https://docs.chainlit.io/authentication/password),
[OAuth](https://docs.chainlit.io/authentication/oauth),
[PostgreSQL persistence](https://docs.chainlit.io/data-layers/official), and
[deployment](https://docs.chainlit.io/deploy/overview). PocketID documents the
same [authorization, token, and user-info endpoints](https://pocket-id.org/docs/client-examples/linkding)
and [`openid email profile groups` scopes](https://pocket-id.org/docs/client-examples/blinko).

## Human In The Loop Demo

The `interruptible-approval` model showcases LangGraph `interrupt()` and resume.

=== "Chainlit"

    ```bash
    make run-demo-ui-chainlit-hitl
    ```

    The Chainlit UI opens the interrupt approval flow and uses the same
    persistent thread store and selected login mode.

=== "Open WebUI"

    ```bash
    docker compose up -d open-webui
    ```

    Import the generic Pipe, then select `interruptible-approval`.

See [Docker](../how-to-guides/docker.md) for the Open WebUI Function setup and
[OpenAI compatibility](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for the interrupt tool-call protocol.

## Next Steps

- [Register custom graphs in a FastAPI app](custom-graphs.md#register-and-bind)
- [Connect OpenAI clients](openai-clients.md)
- [Run with Docker](../how-to-guides/docker.md)
