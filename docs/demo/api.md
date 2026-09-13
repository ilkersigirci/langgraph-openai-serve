# Run The Demo API

This tutorial uses the self-contained project under `demo/api`. It serves
several LangGraph graphs through the OpenAI-compatible `/v1` interface.

## Prerequisites

- Python 3.11 or newer
- `uv`
- Bash and Just 1.58.0 or newer
- PostgreSQL (the included Compose service requires Docker)
- An OpenAI-compatible upstream model only if you call the LLM-backed graphs

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
    API project and are not installed with the library. `advanced-graph` can
    connect separately to an OpenAI-compatible vector service; document search
    and note saving need `DEMO_API_VECTOR_STORE_ID`.

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

## Advanced Research And Note Review

The [advanced graph](graphs/advanced-graph.md) is Responses-only. It can use a
real web-search backend and an optional private knowledge base. Configure the
knowledge base independently from the model provider:

```dotenv
DEMO_API_VECTOR_STORE_BASE_URL=https://api.openai.com/v1
DEMO_API_VECTOR_STORE_API_KEY=...
DEMO_API_VECTOR_STORE_BIFROST_KEY_NAME=
DEMO_API_VECTOR_STORE_ID=vs_...
```

Omit the vector base URL to reuse the model endpoint; its key also falls back to
the model key. An explicit vector base URL never inherits the model key: set its
own key, or leave it blank to use `DUMMY` for an unauthenticated local service.
The service must expose compatible Files and vector-store upload, polling, and
search endpoints. This can point to OpenAI today or a future LGOS vector
service. Without a vector-store ID, plain answers and web search still work,
while `save_note=true` is rejected.

If a Bifrost passthrough has multiple OpenAI keys, set
`DEMO_API_VECTOR_STORE_BIFROST_KEY_NAME` so every stateful vector-store request
uses the same managed key. Leave it blank for direct OpenAI and other compatible
services.

```python title="Read-only research with live status and answer streaming"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3004/v1", api_key="DUMMY")
with client.responses.stream(
    model="advanced-graph",
    input="Explain durable LangGraph interrupts using our notes and current official docs.",
    tools=[{"type": "web_search"}],
    store=False,
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    result = stream.get_final_response()
print("\nOutcome:", result.status, result.incomplete_details)
```

Private document search is graph-internal and becomes available when the server
has a vector store configured. Clients do not send a new `file_search` tool:
LGOS's public Responses contract is unchanged. Add `web_search` to make public
research available, or use `tool_choice="none"` to disable all search. Status
text is emitted as commentary; rich clients can use each message's `phase` to
render it separately from the final answer.

```python title="Review exact note bytes before saving"
import json

settings = {"lgos_settings": '{"save_note":true}'}
result = client.responses.create(
    model="advanced-graph",
    input="Research LangGraph interrupt durability and prepare a note for our library.",
    tools=[{"type": "web_search"}],
    metadata=settings,
    store=False,
)
while calls := [
    item for item in result.output
    if item.type == "function_call" and item.name == "lgos_interrupt"
]:
    for call in calls:
        print(json.dumps(json.loads(call.arguments), indent=2))
    decision = input("approve, reject, or revision feedback: ")
    result = client.responses.create(
        model="advanced-graph",
        previous_response_id=result.id,
        input=[
            {"type": "function_call_output", "call_id": call.call_id, "output": decision}
            for call in calls
        ],
        tools=[{"type": "web_search"}],
        metadata=settings,
        store=False,
    )
print(result.status, result.output_text)
```

Review arguments include the full content and destination. No file is uploaded
on rejection, and revision feedback produces a new approval request. Send the
decision as a plain string, not a JSON-encoded string. Keep the current response
ID and resend settings/tools on resume, including after an API restart. Both
initial and resumed requests also support `stream=True`.

See [storage boundaries](graphs/advanced-graph.md#storage-boundaries) for
checkpoint, upload, and indexing behavior.

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
