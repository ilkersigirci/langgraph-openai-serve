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
[Citation ownership](../explanation/openai-compatibility.md#citation-ownership)
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

Try the deterministic custom-event showcase:

```python
stream = client.chat.completions.create(
    model="custom-event-showcase",
    messages=[
        {
            "role": "user",
            "content": "Build the compatibility report.",
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

The graph emits `status`, `progress`, and `artifact` events. Progress updates
are interleaved with assistant text, so clients can update passive UI without
changing the ordinary text stream.

## Demo Files

- `demo/api/app.py`: registers graph names as OpenAI model names.
- `demo/api/graphs/simple.py`: a single-node message graph whose runtime context
  controls conversation history and intended audience. It publishes those safe
  fields explicitly through model retrieval and keeps arbitrary system prompt
  text out of runtime settings.
- `demo/api/graphs/lgos_rag.py`: agentic RAG over every Markdown file in
  `docs/`, with relevance grading, bounded query rewriting, streamed generation,
  and grounded answers.
- `demo/api/graphs/custom_io.py`: input, output, and context adapters.
- `demo/api/graphs/advanced_mcp.py`: async factory with mock MCP-style tools.
- `demo/api/graphs/complex_subgraphs.py` and `demo/api/graphs/subgraphs/`:
  router-selected subgraphs with streamed fake chat model output.
- `demo/api/graphs/custom_events.py`: deterministic status, progress, and
  artifact events interleaved with assistant text.
- `demo/api/graphs/interruptible.py`: interrupt and resume graph persisted in
  PostgreSQL by the demo application.
- `demo/api/graphs/citations.py`: custom citation event and OpenAI annotation
  demo.
- `demo/api/settings.py`: Pydantic settings for the demo application.

## Try An Integration

The repository provides optional [Chainlit](../integrations/chainlit.md) and
[Open WebUI](../integrations/open-webui.md) clients. Gateway-specific setup is
under [OpenAI-compatible proxies](../integrations/openai-proxies.md).

## Next Steps

- [Register custom graphs in a FastAPI app](custom-graphs.md#register-and-bind)
- [Connect OpenAI clients](openai-clients.md)
- [Run with Docker](../how-to-guides/docker.md)
