# Architecture

LangGraph OpenAI Serve mounts an OpenAI-compatible FastAPI sub-application on a
host FastAPI app and routes OpenAI chat requests to registered LangGraph graphs.

```mermaid
flowchart LR
  subgraph api["OpenAI API boundary"]
    direction TB
    A["OpenAI client"] --> B["POST /v1/chat/completions"]
    B --> C["Validate OpenAI request"]
    C --> D["GraphRegistry<br/>model to GraphConfig"]
  end

  subgraph execution["LGOS adapter and execution"]
    direction TB
    E["GraphConfig<br/>resolve graph and adapt input"]
    E --> F["Runner<br/>collect or stream events"]
    F -->|graph.astream| G["LangGraph graph"]
    G --> H["LGOS response rendering<br/>OpenAI completion or SSE chunks"]
  end

  D --> E
  H -.->|OpenAI-compatible response| A
```

## Components

`LanggraphOpenaiServe` is the boundary between your FastAPI app and the
OpenAI-compatible sub-application. It mounts the sub-application at the
configured prefix and can add CORS middleware when requested.

The mounted OpenAI app owns the public HTTP surface: model listing, chat
completions, health checks, request validation, response schemas, and
OpenAI-shaped error handling.

`GraphRegistry` maps each OpenAI `model` value to a `GraphConfig`. `GraphConfig`
then resolves the graph, applies custom input/context/output adapters when
present, and tells the runner whether streaming or interrupts are enabled.

The runner is the only layer that calls LangGraph. It converts the validated
OpenAI request into graph input and consumes `graph.astream` in both response
modes. For a normal response it collects the final graph value and custom
events; for a streaming response it forwards message, custom, and interrupt
events. The configured output adapter and chat service render those results as
OpenAI chat-completion objects or SSE chunks.

Endpoint paths and settings live in [Reference](../reference.md).

## Request Flow

1. An OpenAI-compatible client sends a chat completion request.
2. FastAPI validates the request schema.
3. The requested `model` is resolved from `GraphRegistry`.
4. OpenAI messages are converted to LangChain messages.
5. `GraphConfig` builds graph input, runnable config, and runtime context.
6. The runner consumes `graph.astream`, collecting values for a normal response
   or forwarding message and event streams for a streaming response.
7. LGOS renders the result as an OpenAI chat completion or SSE chunk sequence.

## Execution Modes

=== "Non-streaming"

    Requests collect the final graph result and return one chat completion
    response.

=== "Streaming"

    Requests consume LangGraph message and update streams. Text chunks are
    emitted only from configured streamable nodes; interrupt updates become
    OpenAI-compatible tool calls.

## Interrupts

Interrupt-enabled graphs pass `metadata.langgraph_thread_id` into LangGraph
runnable config. They must have a checkpointer so pending interrupts can resume.
The demo keeps an `AsyncPostgresSaver` and its connection pool open for the
application lifespan and stores checkpoints in the PostgreSQL service configured
by `DEMO_POSTGRES_URI`, so checkpoints survive requests and process restarts.
Schema initialization runs before API workers start—as a Compose `pre_start`
lifecycle hook in the demo—so multiple workers can safely use the durable
checkpointer without racing on startup migrations.
