# Architecture Overview

This document provides a high-level overview of the LangGraph OpenAI Serve architecture, explaining how the different components work together to provide an OpenAI-compatible API for LangGraph workflows.

## System Architecture

LangGraph OpenAI Serve consists of several key components:

1. **FastAPI Application**: The web server that handles HTTP requests
2. **LangchainOpenaiApiServe**: The core class that bridges LangGraph and the API
3. **Graph Registry**: A registry that manages LangGraph instances
4. **API Routers**: FastAPI routers for different API endpoints
5. **Schema Models**: Pydantic models for data validation and serialization

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                HTTP Clients                             │
│    (OpenAI Python SDK, JavaScript SDK, curl, etc.)      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Application                    │
│                                                         │
│  ┌─────────────────┐       ┌─────────────────────────┐  │
│  │  Models Router  │       │  Chat Completions Router│  │
│  │   /v1/models    │       │  /v1/chat/completions   │  │
│  └────────┬────────┘       └──────────┬──────────────┘  │
│           │                           │                 │
│  ┌────────▼───────────────────────────▼──────────────┐  │
│  │             LangchainOpenaiApiServe               │  │
│  │                                                   │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │              Graph Registry                 │  │  │
│  │  │                                             │  │  │
│  │  │  ┌───────────┐  ┌───────────┐  ┌──────────┐ │  │  │
│  │  │  │ Graph 1   │  │ Graph 2   │  │ Graph N  │ │  │  │
│  │  │  └───────────┘  └───────────┘  └──────────┘ │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 LangGraph Workflows                     │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### FastAPI Application

The FastAPI application serves as the web server that handles HTTP requests. It can be:

1. Created automatically by LangchainOpenaiApiServe
2. Provided by the user when they want to integrate LangGraph OpenAI Serve with an existing FastAPI application

### LangchainOpenaiApiServe

This is the core class that connects LangGraph workflows with the OpenAI-compatible API. Its responsibilities include:

- Managing the FastAPI application
- Registering and managing LangGraph instances
- Providing routers for different API endpoints
- Handling CORS configuration when needed

### Graph Registry

The Graph Registry maintains a mapping between model names and LangGraph instances. When a request comes in for a specific model, the registry looks up the corresponding LangGraph workflow to execute. The registry allows:

- Registering multiple graphs with different names
- Retrieving graphs by name
- Listing available graphs

### API Routers

LangGraph OpenAI Serve provides several FastAPI routers:

1. **Models Router**: Handles `/v1/models` endpoint to list available LangGraph workflows
2. **Chat Completions Router**: Handles `/v1/chat/completions` endpoint for chat interactions
3. **Health Router**: Provides a health check endpoint at `/health`

### Schema Models

Pydantic models are used for data validation and serialization. These include:

1. **Request Models**: Define the structure of API requests
2. **Response Models**: Define the structure of API responses
3. **OpenAI Compatible Models**: Models that match OpenAI's API schema

## Request Flow

When a client makes a request to the API, the following sequence of events occurs:

1. **Client Request**: A client (like the OpenAI Python SDK) sends a request to the API
2. **FastAPI Router**: The appropriate router handles the request based on the endpoint
3. **Request Validation**: Pydantic models validate the request data
4. **Graph Selection**: The system looks up the requested LangGraph workflow in the registry
5. **Graph Execution**: The LangGraph workflow is executed with the provided messages
6. **Response Formatting**: The result is formatted according to the OpenAI API schema
7. **Client Response**: The response is sent back to the client

### Example Flow for Chat Completion

```
Client Request (POST /v1/chat/completions)
    │
    ▼
FastAPI Chat Router
    │
    ▼
Request Validation (ChatCompletionRequest)
    │
    ▼
Graph Selection (get_graph_for_model)
    │
    ▼
Message Conversion (convert_to_lc_messages)
    │
    ▼
Graph Execution (graph.ainvoke or graph.astream_events)
    │
    ▼
Response Formatting
    │
    ▼
Client Response
```

## Streaming vs. Non-Streaming

LangGraph OpenAI Serve supports both streaming and non-streaming responses:

### Non-Streaming Mode

In non-streaming mode:
1. The entire LangGraph workflow is executed
2. The final result is collected
3. A single response is returned to the client

### Streaming Mode

In streaming mode:
1. The LangGraph workflow is executed with streaming enabled
2. Events from the workflow are captured in real-time
3. Each chunk of generated content is immediately sent to the client
4. The client receives and processes chunks as they arrive

## Integration with LangGraph

LangGraph OpenAI Serve integrates with LangGraph by:

1. Accepting compiled LangGraph workflows (`graph.compile()`)
2. Converting between OpenAI message formats and LangChain message formats
3. Executing workflows with appropriate parameters (temperature, max_tokens, etc.)
4. Handling both streaming and non-streaming execution modes

## Next Steps

- Read about [integration with LangGraph](langgraph-integration.md) for more details on how LangGraph workflows are executed
- Learn about [OpenAI API compatibility](openai-compatibility.md) to understand how the API matches OpenAI's interface
