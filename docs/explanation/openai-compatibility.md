# OpenAI API Compatibility

This document explains how LangGraph OpenAI Serve achieves compatibility with the OpenAI API format, allowing clients to interact with your LangGraph workflows using standard OpenAI client libraries.

## API Compatibility Layer

LangGraph OpenAI Serve implements a subset of the OpenAI API, focusing on the most commonly used endpoints:

- `/v1/models` - For listing available models (LangGraph instances)
- `/v1/chat/completions` - For chat interactions
- `/health` - For health checks

The API is designed to be compatible with OpenAI client libraries in different languages while providing access to your custom LangGraph workflows.

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

OpenAI's API supports streaming responses, which is particularly important for real-time interactions. LangGraph OpenAI Serve implements compatible streaming using Server-Sent Events (SSE):

```python
async def create_chat_completion_stream(
    request: Request,
    body: ChatCompletionRequest,
) -> StreamingResponse:
    """Stream chat completion responses."""
    async def stream_generator():
        unique_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = body.model

        # Initial response with role
        yield f"data: {json.dumps({
            'id': unique_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {
                    'role': 'assistant'
                },
                'finish_reason': None
            }]
        })}\n\n"

        # Generate content stream
        async for content_chunk, metrics in run_langgraph_stream(
            body.model,
            body.messages,
            body.temperature,
            body.max_tokens,
            body.tools if hasattr(body, 'tools') else None,
        ):
            yield f"data: {json.dumps({
                'id': unique_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model,
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': content_chunk
                    },
                    'finish_reason': None
                }]
            })}\n\n"

        # Final message
        yield f"data: {json.dumps({
            'id': unique_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {},
                'finish_reason': 'stop'
            }]
        })}\n\n"

        # End of stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )
```

## UI Event Mode

Clients that can render richer agent events can opt in per request with the
`x_langgraph_openai_serve` extension. The endpoint remains
`/v1/chat/completions`; without this extension, responses stay as ordinary
assistant text.

```json
{
  "model": "my_custom_graph",
  "messages": [{"role": "user", "content": "Summarize this"}],
  "stream": true,
  "x_langgraph_openai_serve": {
    "ui_events": {
      "enabled": true,
      "thread_id": "ui-chat-123"
    }
  }
}
```

In UI event mode, `message.content` for non-streaming responses and
`delta.content` for streaming responses contain newline-delimited AG-UI event
objects. Text is also represented as events, so clients parse one ordered event
log:

```jsonl
{"type":"RUN_STARTED","threadId":"ui-chat-123","runId":"run-..."}
{"type":"TEXT_MESSAGE_START","messageId":"msg-...","role":"assistant"}
{"type":"TEXT_MESSAGE_CONTENT","messageId":"msg-...","delta":"Hello"}
{"type":"TEXT_MESSAGE_END","messageId":"msg-..."}
{"type":"RUN_FINISHED","threadId":"ui-chat-123","runId":"run-...","result":{"finishReason":"stop"}}
```

The event payload uses AG-UI field names such as `threadId`, `runId`,
`messageId`, and `delta`. The request `thread_id` is passed to LangGraph as the
configurable `thread_id`. Streaming chunks only contain complete JSON lines; a
chunk may contain one or more complete lines, but never a partial JSON object.

Prior assistant messages are normalized defensively before graph input. If an
assistant message starts with an AG-UI `RUN_STARTED` event, only
`TEXT_MESSAGE_CONTENT.delta` content is reconstructed into model-visible text.
Render-only events such as citations, state, and custom UI data are dropped from
prompt context by default.

Citation helper events use `CUSTOM` events named `citation`. The backend
normalizes source IDs to run-local IDs such as `cite-1` and preserves original
IDs and provenance under `value.metadata`.

Response-required UI interactions such as human approvals use standard OpenAI
tool calls with the reserved function name `ui_event` and
`finish_reason="tool_calls"`. The compatibility layer validates basic JSON and
correlation fields for tool responses; domain-specific approval validation
belongs in graph code. Interruptible graphs should declare `hitl=True` in their
capabilities and require a UI-owned `thread_id`.

The server does not provide durable chat or interrupt persistence. The UI owns
conversation history and stable chat IDs. LangGraph thread state may be
ephemeral, so exact resume can fail after a backend restart unless your graph
uses durable checkpointing.

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

When tools are provided in the request, they are passed to the LangGraph workflow, which can use them to generate function calls in the response.

## Differences from OpenAI API

While LangGraph OpenAI Serve aims for high compatibility, there are some differences to be aware of:

1. **Model Selection**: Instead of predefined OpenAI models, you specify your registered LangGraph workflow names.

2. **Feature Support**: Not all OpenAI API features are supported. The focus is on chat completions with optional tool calling.

3. **Authentication**: By default, authentication is not enforced, though you can add it as shown in the [Authentication Guide](../how-to-guides/authentication.md).

4. **Token Counting**: Token usage statistics are approximated rather than using OpenAI's tokenizer.

## Client Compatibility

The API is compatible with:

- OpenAI Python SDK
- OpenAI Node.js SDK
- Most other OpenAI-compatible clients
- Direct HTTP requests (e.g., with curl)

Models also expose ignored-by-default capability metadata under
`x_langgraph_openai_serve` in `/v1/models`:

```json
{
  "id": "my_custom_graph",
  "x_langgraph_openai_serve": {
    "ui_event_protocol": {
      "name": "ag-ui",
      "version": "v1",
      "transport": "openai-chat-completions-content"
    },
    "capabilities": {
      "ui_events": true,
      "hitl": false,
      "citations": false,
      "state": false
    }
  }
}
```

## Using with OpenAI Client Libraries

Here's a simple example of using the OpenAI Python client with LangGraph OpenAI Serve:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Your LangGraph OpenAI Serve URL
    api_key="any-value"  # API key is not verified by default
)

response = client.chat.completions.create(
    model="my_custom_graph",  # The name of your registered LangGraph workflow
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
```

## Next Steps

- Read more about [LangGraph integration](langgraph-integration.md) to understand how LangGraph workflows are executed.
- Explore the [API reference](../reference.md) for detailed endpoint documentation.
