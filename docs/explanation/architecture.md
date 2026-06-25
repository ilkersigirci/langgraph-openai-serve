# Architecture Overview

This document provides a high-level overview of the LangGraph OpenAI Serve
architecture. For the public client contract, see
[OpenAI API Compatibility](openai-compatibility.md).

## System Architecture

LangGraph OpenAI Serve consists of several key components:

1. **FastAPI Host Application**: The web server that handles HTTP requests
2. **LangchainOpenaiApiServe**: The core class that bridges LangGraph and the API
3. **Graph Registry**: A registry that manages LangGraph instances
4. **OpenAI Sub-Application**: A mounted FastAPI app containing the OpenAI-compatible routes
5. **Schema Models**: Pydantic models for data validation and serialization

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│          OpenAI-Compatible Clients                      │
│    (OpenAI SDKs, Chainlit, Open WebUI, etc.)            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                FastAPI Host Application                 │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │          Mounted OpenAI FastAPI App               │  │
│  │    {prefix}/models, {prefix}/chat/completions     │  │
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

### FastAPI Host Application

The FastAPI application serves as the web server that handles HTTP requests. It can be:

1. Created automatically by LangchainOpenaiApiServe
2. Provided by the user when they want to integrate LangGraph OpenAI Serve with an existing FastAPI application

### LangchainOpenaiApiServe

This is the core class that connects LangGraph workflows with the OpenAI-compatible API. Its responsibilities include:

- Managing the host FastAPI application
- Registering and managing LangGraph instances
- Mounting an OpenAI-compatible FastAPI sub-application at the configured prefix
- Handling CORS configuration when needed

### Graph Registry

The Graph Registry maintains a mapping between model names and LangGraph instances. When a request comes in for a specific model, the registry looks up the corresponding LangGraph workflow to execute. The registry allows:

- Registering multiple graphs with different names
- Retrieving graphs by name
- Listing available graphs

### OpenAI Sub-Application

LangGraph OpenAI Serve mounts an OpenAI-compatible FastAPI sub-application on
the host app. The sub-application installs OpenAI-shaped error handlers and
contains several FastAPI routers:

1. **Models Router**: Handles `{prefix}/models` endpoint to list available LangGraph workflows
2. **Chat Completions Router**: Handles `{prefix}/chat/completions` endpoint for chat interactions
3. **Health Router**: Provides a health check endpoint at `{prefix}/health`

The default OpenAI-compatible prefix is `/v1`. Set `LGOS_OPENAI_API_PREFIX` or
pass `prefix=` to `bind_openai_chat_completion()` to mount the sub-application
elsewhere.
FastAPI docs for the mounted sub-application are disabled by default; set
`LGOS_OPENAI_API_DOCS_ENABLED=true` to expose `{prefix}/docs`,
`{prefix}/redoc`, and `{prefix}/openapi.json`.

### Schema Models

Pydantic models are used for data validation and serialization. These include:

1. **Request Models**: Define the structure of API requests
2. **Response Models**: Define the structure of API responses
3. **OpenAI Compatible Models**: Models that match OpenAI's API schema

## Request Flow

When an OpenAI-compatible client makes a request to the API, the following
sequence of events occurs:

1. **Client Request**: A client such as the OpenAI Python SDK, Chainlit, or
   Open WebUI sends an OpenAI-compatible request to the API
2. **Mounted OpenAI App**: The mounted sub-application receives requests under the configured prefix
3. **FastAPI Router**: The appropriate router handles the request based on the endpoint
4. **Request Validation**: Pydantic models validate the request data
5. **Graph Selection**: The system looks up the requested LangGraph workflow in the registry
6. **Graph Execution**: The LangGraph workflow is executed with the provided messages
7. **Response Formatting**: The result is formatted according to the OpenAI API schema
8. **Client Response**: The response is sent back to the client

### Example Flow for Chat Completion

```
Client Request (POST {prefix}/chat/completions)
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
Graph Execution (graph.ainvoke or graph.astream)
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
3. Adapting requests into native graph input, output, and runtime context schemas when configured
4. Handling both streaming and non-streaming execution modes
5. Passing `metadata.langgraph_thread_id` into LangGraph runnable config for
   graphs registered with `interrupts_enabled=True`

Interrupt-enabled graphs must be compiled with a checkpointer. This package
validates that requirement when the graph is resolved. Demo and test graphs may
use LangGraph's `InMemorySaver`, but production graphs should use a durable
checkpointer so pending interrupts survive restarts and work across workers.

## Next Steps

- Read about [integration with LangGraph](langgraph-integration.md) for more details on how LangGraph workflows are executed
- Learn about [OpenAI API compatibility](openai-compatibility.md) to understand how the API matches OpenAI's interface
