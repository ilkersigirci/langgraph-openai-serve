# Docker Deployment

This guide explains how to deploy your LangGraph OpenAI Serve API using Docker for production environments.

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker
- Docker Compose (optional, but recommended)

## Using the Provided Docker Setup

The LangGraph OpenAI Serve project comes with a ready-to-use Docker configuration in the `docker` directory. Here's how to use it:

### Building and Running the Docker Container

You can use Docker Compose to build and run the container:

```bash
# Start the server
docker compose up -d langgraph-openai-serve-dev
```

If you want to use the project with open-webui, a compatible UI for interacting with OpenAI-compatible APIs:

```bash
# For a complete example with open-webui
docker compose up -d open-webui
```

### Accessing the API

Once the container is running, you can access the API at:

- API: http://localhost:8000/v1
- OpenWebUI (if using): http://localhost:3000

## Creating a Custom Docker Deployment

If you need to create a custom Docker deployment for your specific LangGraph OpenAI Serve application, follow these steps:

### 1. Create a Dockerfile

Create a `Dockerfile` with the following content:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Create a requirements.txt File

Create a `requirements.txt` file with your dependencies:

```
langgraph-openai-serve
uvicorn
```

Add any additional dependencies your application needs.

### 3. Create Your Application

Create an `app.py` file with your custom LangGraph OpenAI Serve configuration:

```python
from fastapi import FastAPI
from langgraph_openai_serve import LangchainOpenaiApiServe, GraphRegistry, GraphConfig

# Import your custom graphs
from my_graphs import graph1, graph2

# Create a FastAPI app
app = FastAPI(
    title="My LangGraph API",
    description="Custom LangGraph API with OpenAI compatibility",
)

# Create a GraphRegistry
graph_registry = GraphRegistry(
    registry={
        "graph1": GraphConfig(graph=graph1, streamable_node_names=["generate"]),
        "graph2": GraphConfig(graph=graph2, streamable_node_names=["generate"]),
    }
)

# Initialize the LangGraph OpenAI Serve
graph_serve = LangchainOpenaiApiServe(
    app=app,
    graphs=graph_registry,
    configure_cors=True,
)

# Bind the OpenAI-compatible endpoints
graph_serve.bind_openai_chat_completion(prefix="/v1")
```

### 4. Create a Docker Compose File

For easier deployment, create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  langgraph-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### 5. Build and Run

Build and run your Docker container:

```bash
docker-compose up -d
```

## Environment Variables

You can configure your deployment using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | The port to run the server on | `8000` |
| `HOST` | The host to bind to | `0.0.0.0` |
| `LOG_LEVEL` | Logging level | `info` |

You can pass these variables in your Docker Compose file or directly to Docker.

## Production Best Practices

When deploying to production, consider the following best practices:

1. **Use a Production ASGI Server**: While Uvicorn is good for development, consider using Gunicorn with Uvicorn workers for production.

2. **Implement Authentication**: Add proper authentication to protect your API.

3. **Set Up HTTPS**: Use a reverse proxy like Nginx to handle HTTPS.

4. **Resource Constraints**: Set appropriate memory and CPU limits for your Docker container.

5. **Monitoring**: Implement monitoring for your service using tools like Prometheus and Grafana.

6. **Logging**: Configure proper logging to capture errors and performance metrics.

## Example Production Docker Compose

```yaml
version: '3'

services:
  langgraph-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=warning
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s

  nginx:
    image: nginx:latest
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl
    depends_on:
      - langgraph-api
    restart: unless-stopped
```

## Next Steps

After deploying your API with Docker, you might want to:

- [Implement authentication](authentication.md) for your API
