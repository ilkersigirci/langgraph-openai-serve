# OpenAI API Compatibility

This document explains how LangGraph OpenAI Serve achieves compatibility with
the OpenAI API format, allowing clients to interact with your LangGraph
workflows using standard OpenAI client libraries and OpenAI-compatible tools
such as Chainlit and Open WebUI.

## Compatibility Contract

LangGraph OpenAI Serve is an OpenAI-client compatibility layer, not a separate
LangGraph-specific HTTP API. Public chat and model behavior must remain
reachable through the configured OpenAI-compatible base URL used by official
OpenAI SDKs, Chainlit, Open WebUI, and similar clients. The default mount prefix
is `/v1`.

When adding graph features, adapt them into OpenAI-compatible request fields,
response objects, tool calls, streaming chunks, metadata, or error envelopes.
Do not require custom client-only payloads, headers, routes, or SSE event shapes
for core graph behavior unless those additions preserve the OpenAI client path.

## API Compatibility Layer

LangGraph OpenAI Serve implements a subset of the OpenAI API. With the default `LGOS_OPENAI_API_PREFIX=/v1`, those endpoints are:

- `/v1/models` - For listing available models (LangGraph instances)
- `/v1/chat/completions` - For chat interactions
- `/v1/health` - For health checks

The API is designed to be compatible with OpenAI client libraries in different
languages while providing access to your custom LangGraph workflows.

## Schema Compatibility

The compatibility is achieved through carefully designed Pydantic models that mirror the OpenAI API schema:

### Request Schema

```python
class ChatCompletionRequest(BaseModel):
    model: str  # Maps to your LangGraph workflow name
    messages: list[ChatCompletionRequestMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    tools: list[Tool] | None = None
    stop: list[str] | None = None
```

### Response Schema

```python
class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int  # Unix timestamp
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
```

## Message Format Conversion

One of the key aspects of OpenAI compatibility is converting between OpenAI message formats and LangChain message formats:

### OpenAI to LangChain Conversion

```python
def convert_to_lc_messages(messages: list[ChatCompletionRequestMessage]) -> list[BaseMessage]:
    """Convert OpenAI API messages to LangChain messages."""
    lc_messages = []
    for message in messages:
        if message.role == "user":
            lc_messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            lc_messages.append(AIMessage(content=message.content))
        elif message.role == "system":
            lc_messages.append(SystemMessage(content=message.content))
        # Handle more message types as needed
    return lc_messages
```

### LangChain to OpenAI Conversion

The reverse conversion happens when formatting responses:

```python
def create_chat_completion(
    model: str,
    response_content: str,
    token_usage: dict[str, int]
) -> ChatCompletion:
    """Create a ChatCompletion object from a response string."""
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    role="assistant",
                    content=response_content
                ),
                finish_reason="stop"
            )
        ],
        usage=ChatCompletionUsage(**token_usage)
    )
```

## Streaming Support

OpenAI's API supports streaming responses, which is particularly important for real-time interactions. LangGraph OpenAI Serve returns compatible Server-Sent Events (SSE) chunks and delegates content generation to `run_langgraph_stream`:

```python
async for event in run_langgraph_stream(
    body.model,
    body.messages,
    graph_registry,
    body,
):
    if isinstance(event, LangGraphInterrupt):
        yield interrupt_tool_call_chunk(event)
        continue

    yield {
        "object": "chat.completion.chunk",
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": event},
                "finish_reason": None,
            }
        ],
    }
```

Inside the runner, streaming uses LangGraph message streaming for text chunks and
LangGraph update streaming for interrupts:

```python
async for event in graph.astream(
    inputs,
    context=context,
    stream_mode=["messages", "updates"],
    subgraphs=True,
    version="v2",
):
    if event["type"] == "updates" and "__interrupt__" in event["data"]:
        yield extract_interrupt(event["data"])
        continue

    message, metadata = event["data"]
    if metadata.get("langgraph_node") in streamable_node_names:
        yield message.text
```

## Error Responses

OpenAI-compatible routes return errors in the OpenAI envelope:

```json
{
  "error": {
    "message": "Graph 'missing' not found in registry.",
    "type": "invalid_request_error",
    "param": "model",
    "code": null
  }
}
```

Route code that knows the OpenAI error metadata should raise
`OpenAIHTTPException` with `openai.types.shared.ErrorObject`. This keeps native
OpenAI clients on their normal error path, such as `BadRequestError` for HTTP
400 responses, with `type`, `param`, and `code` available on the SDK exception.
Generic FastAPI validation and HTTP errors are translated by the shared error
handlers registered when the OpenAI routers are bound.

## Function/Tool Calling Support

OpenAI's API supports tool calling, and LangGraph OpenAI Serve provides compatibility for this feature:

```python
class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class Tool(BaseModel):
    type: str = "function"
    function: FunctionDefinition
```

Tool definitions are accepted in the request schema for compatibility. Graphs
that need request-level tool information can read the full request in
`request_to_input` or use their own tool loading strategy. The demo
`advanced-mcp-tools` graph shows async MCP-style tool loading at graph
construction time.

Interrupt-enabled graphs also use normal OpenAI tool calls for human-in-the-loop
pauses. The reserved `langgraph_interrupt` tool carries a versioned argument
envelope:

```json
{
  "version": 1,
  "kind": "hitl.interrupt",
  "thread_id": "client-chat-id",
  "interrupt_id": "langgraph-interrupt-id",
  "payload": {}
}
```

Clients resume the graph by sending a follow-up `tool` role message with the
matching `tool_call_id` and JSON content like `{"resume": "approved"}`.

## Differences from OpenAI API

While LangGraph OpenAI Serve aims for high compatibility, there are some differences to be aware of:

1. **Model Selection**: Instead of predefined OpenAI models, you specify your registered LangGraph workflow names.

2. **Feature Support**: Not all OpenAI API features are supported. The focus is on chat completions with optional tool calling.

3. **Authentication**: By default, authentication is not enforced, though you can add it as shown in the [Authentication Guide](../how-to-guides/authentication.md).

4. **Token Counting**: Token usage statistics are approximated rather than using OpenAI's tokenizer.

## Client Compatibility

The API is compatible with OpenAI-client ingestion through:

- OpenAI Python SDK
- OpenAI Node.js SDK
- Chainlit configured with an OpenAI base URL
- Open WebUI configured with an OpenAI-compatible connection
- Most other OpenAI-compatible clients

Direct HTTP requests, such as `curl`, may be used to inspect endpoints during
development, but compatibility decisions should be validated through OpenAI
client behavior.

## Using with OpenAI Client Libraries

Here's a simple example of using the OpenAI Python client with LangGraph OpenAI Serve:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Your LangGraph OpenAI Serve URL
    api_key="any-value"  # API key is not verified by default
)

response = client.chat.completions.create(
    model="custom-input-output-context",
    messages=[
        {"role": "user", "content": "Show me the custom adapter."},
    ],
    user="demo-user",
)

print(response.choices[0].message.content)
```

## Next Steps

- Read more about [LangGraph integration](langgraph-integration.md) to understand how LangGraph workflows are executed.
- Explore the [API reference](../reference.md) for detailed endpoint documentation.
