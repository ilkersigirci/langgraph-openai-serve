# Run The Demo API

This tutorial uses the self-contained project under `demo/api`. It serves
several LangGraph graphs through the OpenAI-compatible `/v1` interface.

## Prerequisites

- Python 3.11 or newer
- `uv`
- Bash and Just 1.58.0 or newer
- PostgreSQL (the included Compose service requires Docker)
- An OpenAI-compatible upstream model only if you call the LLM-backed graphs
- A Hatchet deployment and client token only for background Responses

!!! tip "Start without an upstream model"

    Several deterministic graphs do not require provider credentials. Use the
    [graph matrix](graphs/index.md) to choose one and see its other dependencies.

## Start PostgreSQL And The API

```bash title="Prepare the demo"
cp demo/.env.example demo/.env
just demo/up lgos-db --wait
```

=== "Test this checkout"

    Overlay the parent LGOS checkout without changing the demo lockfile:

    ```bash
    just demo/api --editable
    ```

=== "Use the published image"

    Run the published API container and its PostgreSQL dependency:

    ```bash
    just demo/up lgos-demo-api-a
    ```

??? info "Demo environment settings"

    The API reads `DEMO_API_POSTGRES_URI` from the demo environment. Use
    [`.env.example`](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/demo/.env.example)
    for the supplied connection settings.

    LLM-backed graphs additionally read `DEMO_API_OPENAI_BASE_URL`,
    `DEMO_API_OPENAI_API_KEY`, and `DEMO_API_OPENAI_MODEL`. The
    `lgos-rag` graph also reads `DEMO_API_OPENAI_EMBEDDING_MODEL`. Its corpus is
    packaged with the API. The `server-tool` graph reads
    `DEMO_API_WEB_SEARCH_BACKEND` and `DEMO_API_WEB_SEARCH_URL` to choose its
    web-search execution backend. These settings and dependencies belong to the
    API project and are not installed with the library.

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
lightweight `lgos` list extension.

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

Start with the dependency-free MCP-shaped example:

```python
response = client.responses.create(
    model="mcp-mock",
    input="What is the weather in Istanbul?",
    store=False,
)

print(response.output_text)
```

`mcp-mock` uses a stand-in client and deterministic fake model, so it teaches
async tool discovery and the agent tool loop without requiring an MCP server,
gateway, database, or provider credential. See
[Core Graph Patterns](graphs/core-patterns.md#mcp-mock).

The real `mcp-postgres` graph expects its OpenAI client to discover and execute
tools through the selected gateway's MCP endpoint. Use the maintained Chainlit
or Open WebUI client for the complete native tool loop; see
[PostgreSQL Through Native MCP](graphs/mcp-postgres.md#try-it).
`advanced-graph` uses the same client-owned loop for any tools authorized by the
gateway; `mcp-postgres` remains the narrower database-focused example.

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

Try the deterministic response-outcome showcase:

```python
refusal = client.responses.create(
    model="response-outcomes",
    input="refusal",
    store=False,
)
print(refusal.status, refusal.output[0].content[0].refusal)

stream = client.responses.create(
    model="response-outcomes",
    input="incomplete",
    store=False,
    stream=True,
)
for event in stream:
    if event.type == "response.incomplete":
        print(event.response.incomplete_details.reason)
```

See [Core Graph Patterns](graphs/core-patterns.md#response-outcomes) for when a
refusal differs from an incomplete response and which terminal events clients
must handle.

For background Responses, enable `DEMO_API_BACKGROUND_ENABLED`, start the
independent `just demo/background-worker` process, or run the complete UI path
with the `background` Compose profile and `just demo/compose`.
`advanced-graph`, `background-mock`, and `background-interrupt` support it.
Chainlit and Open WebUI expose polling through either bundled gateway. See
[Background Mock](graphs/background-mock.md) for basic execution and
[Background Interrupt](graphs/background-interrupt.md) for approval and resumption.

### Background Python Client

The background graph examples share this direct API client and polling helper.
Run this setup first with the API and background worker running, then choose a
[create/poll/cancel example](graphs/background-mock.md#python-sdk) or the
[review/resume example](graphs/background-interrupt.md#python-sdk).

```python
import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:3004/v1", api_key="DUMMY")


def poll(response):
    deadline = time.monotonic() + 120
    while response.status in {"queued", "in_progress"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Still running: {response.id}")
        time.sleep(1)
        response = client.responses.retrieve(response.id)
    if response.status != "completed":
        raise RuntimeError(f"{response.status}: {response.error}")
    return response
```

`poll` returns a completed Response or raises with its terminal status and error.
The timeout stops local polling; it does not cancel the background work. The
examples close the client with `with client:`. Rerun the setup before running
another example.

For gateway URLs, model naming, and credentials, use the
[OpenAI proxy guide](../how-to-guides/openai-proxies.md).

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
