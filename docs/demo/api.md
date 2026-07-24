# Run The Demo API

This tutorial uses the self-contained project under `demo/api`. It serves
several LangGraph graphs through the OpenAI-compatible `/v1` interface.

## Prerequisites

- Python 3.11 or newer
- `uv`
- PostgreSQL (the included Compose service requires Docker)
- An OpenAI-compatible upstream model only if you call the LLM-backed graphs

!!! tip "Start without an upstream model"

    The custom adapter, citation, nested subgraph, and mock MCP demo graphs do
    not require real API keys.

## Start PostgreSQL And The API

```bash title="Prepare the demo"
cd demo
cp .env.example .env
docker compose -f compose.yaml up -d postgres
```

=== "Test this checkout"

    Overlay the parent LGOS checkout without changing the demo lockfile:

    ```bash
    uv run --directory api --env-file ../.env \
      --locked --with-editable ../.. lgos-demo-api-setup
    uv run --directory api --env-file ../.env \
      --locked --with-editable ../.. lgos-demo-api
    ```

=== "Use the locked PyPI release"

    Run the API as a completely independent project:

    ```bash
    make run-api
    ```

??? info "Demo environment settings"

    The API reads `DEMO_API_POSTGRES_URI` and defaults to
    `postgresql://lgos:lgos@localhost:5432/lgos`, which matches the Compose
    service.

    LLM-backed graphs additionally read `DEMO_API_OPENAI_BASE_URL`,
    `DEMO_API_OPENAI_API_KEY`, and `DEMO_API_OPENAI_MODEL`. The
    `lgos-rag` graph also reads `DEMO_API_OPENAI_EMBEDDING_MODEL`. Its corpus is
    packaged with the API. These settings and dependencies belong to the API
    project and are not installed with the library.

The base URL is `http://localhost:8000/v1`.

Inspect registered graphs:

```bash
curl http://localhost:8000/v1/models
```

The complete model and requirement matrix is in [Example Graphs](graphs.md).

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
[Citation ownership](../explanation/openai-compatibility.md#citation-ownership)
for transport and client behavior.

Ask the RAG graph about the packaged LGOS overview and demo documentation with
real-time token streaming:

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

Try the deterministic status-event showcase:

```python
stream = client.chat.completions.create(
    model="status-events",
    messages=[
        {
            "role": "user",
            "content": "Prepare the media workflow.",
        }
    ],
    stream=True,
    user="demo-user",
    metadata={"langgraph_stream_events": "v1"},
)

for chunk in stream:
    extension = (chunk.model_extra or {}).get("langgraph_openai_serve")
    if isinstance(extension, dict):
        print("Event:", extension["event"])

    if text := chunk.choices[0].delta.content:
        print(text, end="", flush=True)
```

The graph emits `Generating audio`, `Calculating embeddings`, and a final
`Media ready` status with `done=True`. The separate `custom-event-showcase`
graph demonstrates application-defined `progress` and `artifact` events
interleaved with assistant text.

## Try A Demo Client

The demo includes optional [Chainlit](chainlit.md) and
[Open WebUI](open-webui.md) clients. Route either client through the bundled
[Bifrost gateway](bifrost.md) to exercise normal and pass-through proxy paths.

## Next Steps

- [Run the complete stack with Docker Compose](docker.md)
- [Register custom graphs in your own FastAPI app](../tutorials/custom-graphs.md#register-and-bind)
- [Connect OpenAI clients](../tutorials/openai-clients.md)
