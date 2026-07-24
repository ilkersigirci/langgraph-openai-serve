# LangGraph OpenAI Serve

Serve LangGraph graphs through an OpenAI-compatible `/v1` API so existing
OpenAI SDKs, Chainlit, Open WebUI, and similar clients can call them without a
project-specific protocol.

## Install

Use `uv` for project dependency management:

```bash
uv add langgraph-openai-serve
```

The equivalent `pip` command is:

```bash
pip install langgraph-openai-serve
```

The package contains the OpenAI-compatible server integration, not a built-in
LLM graph. Applications register their own graphs. The `demo/` checkout keeps
each deployable application in an independent uv project with its own lockfile.
Its `lgos-rag` example indexes a small corpus packaged with the demo API, so the
entire directory can be copied and run without files from this repository.

## Quick Demo

From this repository, prepare the demo environment and PostgreSQL:

```bash
cd demo
cp .env.example .env
docker compose -f compose.yaml up -d lgos-db
uv run --directory api --env-file ../.env \
  --locked --with-editable ../.. lgos-demo-api-setup
uv run --directory api --env-file ../.env \
  --locked --with-editable ../.. lgos-demo-api
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

The optional editable overlay tests this checkout without changing the
self-contained demo project or its lockfile. The demo publishes independent API
and Chainlit images and uses official images for third-party services such as
Open WebUI. See the [demo Docker Compose guide](docs/demo/docker.md).

The demo also includes a PostgreSQL-persistent Chainlit client. It uses a
shared mock login by default, with PocketID OAuth available as an opt-in mode.
See the [Chainlit demo](docs/demo/chainlit.md).

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

The default base URL is `{host}/v1`. Registered graph names become OpenAI `model`
values.

## Docs

- Documentation home: [docs/index.md](docs/index.md)
- Package getting started: [docs/getting-started.md](docs/getting-started.md)
- Self-contained demo stack: [docs/demo/index.md](docs/demo/index.md)
- Runnable demo API: [docs/demo/api.md](docs/demo/api.md)
- Demo graph catalog: [docs/demo/graphs.md](docs/demo/graphs.md)
- OpenAI clients: [docs/tutorials/openai-clients.md](docs/tutorials/openai-clients.md)
- Custom graphs: [docs/tutorials/custom-graphs.md](docs/tutorials/custom-graphs.md)
- LangGraph runtime settings: [docs/how-to-guides/langgraph-runtime-settings.md](docs/how-to-guides/langgraph-runtime-settings.md)
- OpenAI-compatible proxies: [docs/how-to-guides/openai-proxies.md](docs/how-to-guides/openai-proxies.md)
- API and configuration: [docs/reference.md](docs/reference.md)
- Compatibility contract: [docs/explanation/openai-compatibility.md](docs/explanation/openai-compatibility.md)
