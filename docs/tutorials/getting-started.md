# Getting Started with LangGraph OpenAI Serve

This tutorial will guide you through the process of setting up and running your first LangGraph OpenAI compatible server.

## Prerequisites

Before you begin, make sure you have:

- Python 3.11 or higher installed
- Basic familiarity with [LangGraph](https://github.com/langchain-ai/langgraph)
- Basic understanding of FastAPI (optional)

## Installation

First, install the `langgraph-openai-serve` package:

```bash
# Using uv (recommended)
uv add langgraph-openai-serve

# Using pip
pip install langgraph-openai-serve
```

## Basic Usage

Here's a simple example to help you get started. In this example, we'll create a basic server that exposes a simple LangGraph workflow through an OpenAI-compatible API.

### 1. Create a Python file for your server

Create a new file called `server.py` with the following content:

```python
from langgraph_openai_serve import LangchainOpenaiApiServe, GraphRegistry, GraphConfig

# Import your LangGraph instances or use the default simple graph
# that comes with the package
from langgraph_openai_serve.graph.simple_graph import app as simple_graph

# Create a GraphRegistry
graph_registry = GraphRegistry(
    registry={
        "simple_graph": GraphConfig(graph=simple_graph, streamable_node_names=["generate"]),
    }
)

# Create a server instance with your graph(s)
graph_serve = LangchainOpenaiApiServe(
    graphs=graph_registry,
    configure_cors=True,  # Enable CORS for browser clients
)

# Bind the OpenAI-compatible endpoints
graph_serve.bind_openai_chat_completion(prefix="/v1")

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(graph_serve.app, host="0.0.0.0", port=8000)
```

### 2. Run the server

Start the server by running:

```bash
python server.py
```

Your server should now be running at http://localhost:8000

### 3. Test the API

You can test the API using curl:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple_graph",
    "messages": [
      {"role": "user", "content": "Hello, how can you help me today?"}
    ]
  }'
```

Alternatively, you can use the OpenAI Python client:

```python
from openai import OpenAI

# Create a client pointing to your API
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any-value"  # API key is not verified
)

# Use the graph by specifying its name as the model
response = client.chat.completions.create(
    model="simple_graph",  # This maps to the graph name in your registry
    messages=[
        {"role": "user", "content": "Hello, how can you help me today?"}
    ]
)

print(response.choices[0].message.content)
```

## Using with your own FastAPI application

If you already have a FastAPI application, you can integrate LangGraph OpenAI Serve with it:

```python
from fastapi import FastAPI
from langgraph_openai_serve import LangchainOpenaiApiServe, GraphRegistry, GraphConfig
from langgraph_openai_serve.graph.simple_graph import app as simple_graph

# Create a FastAPI app
app = FastAPI(
    title="My API with LangGraph",
    version="1.0",
    description="API that includes LangGraph capabilities",
)

# Create a GraphRegistry
graph_registry = GraphRegistry(
    registry={
        "simple_graph": GraphConfig(graph=simple_graph, streamable_node_names=["generate"]),
    }
)

# Create the LangchainOpenaiApiServe instance
graph_serve = LangchainOpenaiApiServe(
    app=app,  # Pass in your existing FastAPI app
    graphs=graph_registry,
)

# Bind the OpenAI-compatible endpoints
graph_serve.bind_openai_chat_completion(prefix="/v1")

# Add your other routes as needed
@app.get("/")
async def root():
    return {"message": "Welcome to my API with LangGraph integration!"}

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Next Steps

Now that you have a basic understanding of how to use LangGraph OpenAI Serve, you might want to:

1. [Create custom graphs](custom-graphs.md) to expose through your API
2. Learn more about [connecting with OpenAI clients](openai-clients.md)
3. Explore [Docker deployment](../how-to-guides/docker.md) for production use
