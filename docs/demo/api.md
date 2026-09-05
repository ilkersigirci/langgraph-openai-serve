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
The separate `lgos-files-api` project and image serve the central S3-backed
Files API on port 3006. It is not mounted into either graph API; see its
[run guide](files-api.md) and [settings](reference.md#files-api-settings).

Inspect registered graphs:

```bash
curl http://localhost:3004/v1/models
```

Each demo graph publishes its API-owned description and feature names in the
lightweight `langgraph_openai_serve` list extension.

The complete model and requirement matrix is in [Example Graphs](graphs/index.md).

## Call A Graph

```python title="Call a registered graph"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3004/v1", api_key="DUMMY")

response = client.responses.create(
    model="custom-input-output-context",
    input="Show me the custom adapter.",
    store=False,
    user="demo-user",
)

print(response.output_text)
```

Try the citation graph:

```python
response = client.responses.create(
    model="citation-events",
    input="Show me a cited answer.",
    store=False,
)

print(response.output_text)
print(response.output[0].content[0].annotations)
```

See [Events And Citations](graphs/events-and-citations.md) for this graph's
output and
[Citation ownership](../explanation/openai-compatibility.md#citation-ownership)
for the normative transport boundary.

Ask the RAG graph about the packaged LGOS overview and demo documentation with
real-time token streaming:

```python
stream = client.responses.create(
    model="lgos-rag",
    input="How does LGOS streaming work?",
    store=False,
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

See [LGOS RAG](graphs/lgos-rag.md) for its retrieval flow, bounded rewrite, and
process-local index lifetime.

Try the async mock MCP graph:

```python
response = client.responses.create(
    model="advanced-mcp-tools",
    input="What is the weather in Istanbul?",
    store=False,
)
```

Try the deterministic status-event showcase:

```python
stream = client.responses.create(
    model="status-events",
    input="Prepare the media workflow.",
    store=False,
    stream=True,
    user="demo-user",
)

phases = {}
for event in stream:
    if event.type == "response.output_item.added" and event.item.type == "message":
        phases[event.output_index] = event.item.phase
    elif event.type == "response.output_text.done":
        print(f"{phases[event.output_index]}: {event.text}")
```

See [Events And Citations](graphs/events-and-citations.md) for the status and
custom-event flows and their client behavior.

## Try A Demo Client

The demo includes optional [Chainlit](chainlit.md) and
[Open WebUI](open-webui.md) clients. The Compose stack routes both through the
LiteLLM or [Bifrost gateway](bifrost.md) selected by
`OPENAI_GATEWAY_TYPE`, never directly to an API or Files container. See
[Demo Architecture](architecture.md) for the shared request and ownership
flows, then use each client guide for its adapter-specific behavior.

## Next Steps

- [Run the complete stack with Docker Compose](docker.md)
- [Register custom graphs in your own FastAPI app](../tutorials/custom-graphs.md#register-and-bind)
- [Connect OpenAI clients](../tutorials/openai-clients.md)
