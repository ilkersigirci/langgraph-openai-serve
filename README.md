# LangGraph OpenAI Serve

A package that provides an OpenAI-compatible API for LangGraph instances.

## Features

- Expose your LangGraph instances through an OpenAI-compatible API
- Register multiple graphs and map them to different model names
- Use with any FastAPI application
- Support for both streaming and non-streaming completions

## Installation

```bash
# Using uv
uv add langgraph-openai-serve

# Using pip
pip install langgraph-openai-serve
```

## Quick Start

Here's a simple example of how to use LangGraph OpenAI Serve:

```python
from langgraph_openai_serve import LangchainOpenaiApiServe, GraphRegistry, GraphConfig

# Import your LangGraph instances
from your_graphs import simple_graph

async def advanced_graph():
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent

    tools = await MultiServerMCPClient().get_tools()
    graph = create_react_agent(model="openai:gpt-4.1", tools=tools)
    return graph

# You can configure your graphs with your desired configurations.
simple_graph_with_history = simple_graph.with_config(
    configurable={"use_history": True},
)
simple_graph_no_history = simple_graph.with_config(
    configurable={"use_history": False},
)

# Create a GraphRegistry
graph_registry = GraphRegistry(
    registry={
        "simple-graph-with-history": GraphConfig(
            graph=simple_graph_with_history, streamable_node_names=["generate"]
        ),
        "simple-graph-no-history": GraphConfig(
            graph=simple_graph_no_history, streamable_node_names=["generate"]
            ),
        "advanced_graph": GraphConfig(graph=advanced_graph, streamable_node_names=["generate"])
    }
)

graph_serve = LangchainOpenaiApiServe(
    graphs=graph_registry,
)

# Bind the OpenAI-compatible endpoints
graph_serve.bind_openai_chat_completion(prefix="/v1")

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(graph_serve.app, host="0.0.0.0", port=8000)
```

Usage with your own FastAPI app is also supported:

```python
from fastapi import FastAPI
from langgraph_openai_serve import LangchainOpenaiApiServe, GraphRegistry, GraphConfig

# Import your LangGraph instances
from your_graphs import simple_graph, advanced_graph

# Create a FastAPI app
app = FastAPI(
    title="LangGraph OpenAI API",
    version="1.0",
    description="OpenAI API exposing LangGraph agents",
)

# Create a GraphRegistry
graph_registry = GraphRegistry(
    registry={
        "simple_graph": GraphConfig(graph=simple_graph, streamable_node_names=["generate"]),
        "advanced_graph": GraphConfig(graph=advanced_graph, streamable_node_names=["generate"])
    }
)

graph_serve = LangchainOpenaiApiServe(
    app=app,
    graphs=graph_registry,
)

# Bind the OpenAI-compatible endpoints
graph_serve.bind_openai_chat_completion(prefix="/v1")

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(graph_serve.app, host="0.0.0.0", port=8000)
```

## Using with the OpenAI Client

Once your API is running, you can use any OpenAI-compatible client to interact with it:

```python
from openai import OpenAI

# Create a client pointing to your API
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any-value"  # API key is not verified
)

# Use a specific graph by specifying its name as the model
response = client.chat.completions.create(
    model="simple_graph_1",  # This maps to the graph name in your registry
    messages=[
        {"role": "user", "content": "Hello, how can you help me today?"}
    ]
)

print(response.choices[0].message.content)

# You can also use streaming
stream = client.chat.completions.create(
    model="advanced_graph",
    messages=[
        {"role": "user", "content": "Write a short poem about AI."}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Docker Usage

To run with Docker:

```bash
# Start the server
docker compose up -d langgraph-openai-serve-dev

# For a complete example with open-webui
docker compose up -d open-webui
```
