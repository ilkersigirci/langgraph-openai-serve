# Architecture

LangGraph OpenAI Serve mounts an OpenAI-compatible FastAPI sub-application on a
host FastAPI app and routes OpenAI chat requests to registered LangGraph graphs.

```text
OpenAI client
  -> FastAPI host app
  -> mounted OpenAI app at {prefix}
  -> OpenAI routers, schemas, and error handlers
  -> GraphRegistry: model name -> GraphConfig
  -> GraphConfig: graph, adapters, streaming, interrupts
  -> graph runner: ainvoke or astream
  -> LangGraph graph
  -> OpenAI chat completion or streaming chunks
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
OpenAI request into graph input, runs `ainvoke` for normal responses or
`astream` for streaming responses, then renders the result back into OpenAI
chat-completion objects or streaming chunks.

Endpoint paths and settings live in [Reference](../reference.md).

## Request Flow

1. An OpenAI-compatible client sends a chat completion request.
2. FastAPI validates the request schema.
3. The requested `model` is resolved from `GraphRegistry`.
4. OpenAI messages are converted to LangChain messages.
5. `GraphConfig` builds graph input, runnable config, and runtime context.
6. The runner calls `graph.ainvoke` or `graph.astream`.
7. Output is rendered as an OpenAI chat completion or streaming chunk sequence.

## Execution Modes

Non-streaming requests collect the final graph result and return one chat
completion response.

Streaming requests consume LangGraph message and update streams. Text chunks are
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
