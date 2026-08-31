# Run The Demo API

This tutorial uses the self-contained project under `demo/api`. It serves
several LangGraph graphs through the OpenAI-compatible `/v1` interface.

## Prerequisites

- Python 3.11 or newer
- `uv`
- PostgreSQL (the included Compose service requires Docker)
- An OpenAI-compatible upstream model only if you call the LLM-backed graphs

!!! tip "Start without an upstream model"

    Several deterministic graphs do not require provider credentials. Use the
    [graph matrix](graphs/index.md) to choose one and see its other dependencies.

## Start PostgreSQL And The API

```bash title="Prepare the demo"
cd demo
cp .env.example .env
docker compose -f docker/compose/demo.yml up -d lgos-db
```

=== "Test this checkout"

    Overlay the parent LGOS checkout without changing the demo lockfile:

    ```bash
    make run-api-local
    ```

=== "Use the published image"

    Run the published API container and its PostgreSQL dependency:

    ```bash
    make run-api
    ```

??? info "Demo environment settings"

    The API reads `DEMO_API_POSTGRES_URI` and defaults to
    `postgresql://lgos:lgos@localhost:3001/lgos`, which matches the Compose
    service.

    LLM-backed graphs additionally read `DEMO_API_OPENAI_BASE_URL`,
    `DEMO_API_OPENAI_API_KEY`, and `DEMO_API_OPENAI_MODEL`. The
    `lgos-rag` graph also reads `DEMO_API_OPENAI_EMBEDDING_MODEL`. Its corpus is
    packaged with the API. These settings and dependencies belong to the API
    project and are not installed with the library.

The direct `lgos-a` base URL is `http://localhost:3004/v1`. Compose also runs
the same image as independently addressable `lgos-b` on port 3005; the two
services expose the same graph set under separate provider identities.

Inspect registered graphs:

```bash
curl http://localhost:3004/v1/models
```

Each demo graph publishes its API-owned description in the lightweight
`langgraph_openai_serve` list extension.

The complete model and requirement matrix is in [Example Graphs](graphs/index.md).

## Call A Graph

```python title="Call a registered graph"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3004/v1", api_key="DUMMY")

response = client.chat.completions.create(
    model="custom-input-output-context",
    messages=[{"role": "user", "content": "Show me the custom adapter."}],
    user="demo-user",
)

print(response.choices[0].message.content)
```

Try the citation graph:

```python
response = client.chat.completions.create(
    model="citation-events",
    messages=[{"role": "user", "content": "Show me a cited answer."}],
)

print(response.choices[0].message.content)
print(response.choices[0].message.annotations)
```

See [Events And Citations](graphs/events-and-citations.md) for this graph's
output and
[Citation ownership](../explanation/openai-compatibility.md#citation-ownership)
for the normative transport boundary.

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

See [LGOS RAG](graphs/lgos-rag.md) for its retrieval flow, bounded rewrite, and
process-local index lifetime.

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

See [Events And Citations](graphs/events-and-citations.md) for the status and
custom-event flows and their client behavior.

## Try A Demo Client

The demo includes optional [Chainlit](chainlit.md) and
[Open WebUI](open-webui.md) clients. The Compose stack routes both through the
bundled [Bifrost gateway](bifrost.md). See
[Demo Architecture](architecture.md) for the shared request and ownership
flows, then use each client guide for its adapter-specific behavior.

## Next Steps

- [Run the complete stack with Docker Compose](docker.md)
- [Register custom graphs in your own FastAPI app](../tutorials/custom-graphs.md#register-and-bind)
- [Connect OpenAI clients](../tutorials/openai-clients.md)
