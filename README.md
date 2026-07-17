# LangGraph OpenAI Serve

Serve LangGraph graphs through an OpenAI-compatible `/v1` API so existing
OpenAI SDKs, Chainlit, Open WebUI, and similar clients can call them without a
project-specific protocol.

## Install

```bash
uv add langgraph-openai-serve
# or
pip install langgraph-openai-serve
```

The package contains the OpenAI-compatible server integration, not a built-in
LLM graph. Applications register their own graphs; repository demo graph
dependencies are kept in the `demo` dependency group.

## Quick Demo

From this repository:

```bash
docker compose up -d lgos-postgres
make run-demo-api
```

Then call the demo with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

response = client.chat.completions.create(
    model="custom-input-output-context",
    messages=[{"role": "user", "content": "Show me custom schemas."}],
    user="demo-user",
)

print(response.choices[0].message.content)
```

Use `curl http://localhost:8000/v1/models` only as a diagnostic to inspect the
registered demo graph names.

The repository also includes a PostgreSQL-persistent Chainlit client. It uses a
shared mock login by default, with PocketID OAuth available as an opt-in mode.
See the [Chainlit integration](docs/integrations/chainlit.md).

## Use In FastAPI

```python
from fastapi import FastAPI
from langgraph_openai_serve import GraphConfig, GraphRegistry, LanggraphOpenaiServe
from your_graphs import my_graph

app = FastAPI()
graphs = GraphRegistry(
    registry={
        "my-graph": GraphConfig(
            graph=my_graph,
            streamable_node_names=["generate"],
        )
    }
)

LanggraphOpenaiServe(app=app, graphs=graphs).bind_openai_api()
```

The default base URL is `{host}/v1`. Register graph names become OpenAI `model`
values.

## Docs

- Start here: [docs/index.md](docs/index.md)
- Runnable demo: [docs/tutorials/getting-started.md](docs/tutorials/getting-started.md)
- OpenAI clients: [docs/tutorials/openai-clients.md](docs/tutorials/openai-clients.md)
- Custom graphs: [docs/tutorials/custom-graphs.md](docs/tutorials/custom-graphs.md)
- LangGraph runtime settings: [docs/how-to-guides/langgraph-runtime-settings.md](docs/how-to-guides/langgraph-runtime-settings.md)
- Integrations: [docs/integrations/index.md](docs/integrations/index.md)
- API and configuration: [docs/reference.md](docs/reference.md)
- Compatibility contract: [docs/explanation/openai-compatibility.md](docs/explanation/openai-compatibility.md)
