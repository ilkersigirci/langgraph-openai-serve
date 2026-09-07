import json

import pytest
from httpx import AsyncClient
from langchain_core.messages import BaseMessage
from openai import AsyncOpenAI, BadRequestError
from starlette import status

from langgraph_openai_serve import (
    ClientFunctionTool,
    GraphRegistry,
    GraphRequest,
)


async def test_non_streaming_completion_matches_openai_contract(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert response.object == "chat.completion"
    assert response.model == "test"
    choice = response.choices[0]
    assert choice.message.role == "assistant"
    assert choice.message.content == "hello"
    assert choice.finish_reason == "stop"

    assert response.usage is None


async def test_message_content_parts_are_accepted(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.chat.completions.create(
        model="test",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}],
            }
        ],
    )

    assert response.choices[0].message.content == "hello"


async def test_sdk_assistant_message_can_be_replayed_unchanged(
    openai_client: AsyncOpenAI,
) -> None:
    first = await openai_client.chat.completions.create(
        model="test", messages=[{"role": "user", "content": "Hello"}]
    )

    second = await openai_client.chat.completions.create(
        model="test",
        messages=[
            {"role": "user", "content": "Hello"},
            first.choices[0].message.model_dump(),
            {"role": "user", "content": "Continue"},
        ],
    )

    assert second.choices[0].message.content == "hello"


async def test_modern_function_tools_remain_supported(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    received: list[GraphRequest] = []

    def capture_request(
        request: GraphRequest,
        messages: list[BaseMessage],
    ) -> dict[str, list[BaseMessage]]:
        received.append(request)
        return {"messages": messages}

    graph_registry.get_graph("test").request_to_input = capture_request

    response = await openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "strict": True,
                },
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    assert response.choices[0].message.content == "hello"
    assert received == [
        GraphRequest(
            model="test",
            metadata={},
            user=None,
            tools=(
                ClientFunctionTool(
                    name="get_weather",
                    description="Get the weather for a city.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    strict=True,
                ),
            ),
            tool_choice="auto",
            parallel_tool_calls=False,
        )
    ]


async def test_streaming_completion_forwards_llm_chunks(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    chunks = [chunk async for chunk in stream]

    assert chunks[0].object == "chat.completion.chunk"
    assert chunks[0].model == "test"
    assert chunks[0].choices[0].delta.role == "assistant"
    streamed_content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert streamed_content == "hello"
    assert chunks[-1].choices[0].finish_reason == "stop"


async def test_stream_options_require_streaming(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError, match="stream_options"):
        await openai_client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            stream_options={"include_usage": True},
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("temperature", 0.2),
        ("top_p", 0.5),
        ("n", 2),
        ("stop", "END"),
        ("max_tokens", 10),
        ("presence_penalty", 1.0),
        ("frequency_penalty", 1.0),
        ("logit_bias", {"123": 1}),
    ],
)
async def test_unsupported_generation_controls_are_rejected(
    openai_client: AsyncOpenAI,
    parameter: str,
    value: object,
    stream: bool,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            stream=stream,
            extra_body={parameter: value},
        )

    assert exc_info.value.body["param"] == parameter


async def test_streaming_completion_uses_sse_wire_format(
    client: AsyncClient,
) -> None:
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"].startswith("text/event-stream")

        events = [line async for line in response.aiter_lines() if line]

    assert events
    assert all(event.startswith("data: ") for event in events)
    assert events[-1] == "data: [DONE]"
    for event in events[:-1]:
        assert isinstance(json.loads(event.removeprefix("data: ")), dict)


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(False, id="non-streaming"),
        pytest.param(True, id="streaming"),
    ],
)
async def test_unknown_model_raises_openai_bad_request(
    openai_client: AsyncOpenAI,
    stream: bool,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.chat.completions.create(
            model="missing",
            messages=[{"role": "user", "content": "Hi"}],
            stream=stream,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.response.json() == {
        "error": {
            "message": "Graph 'missing' not found in registry.",
            "type": "invalid_request_error",
            "param": "model",
            "code": None,
        }
    }


async def test_streaming_completion_ignores_custom_stream_events(
    openai_client: AsyncOpenAI,
    fastapi_app,
) -> None:
    from langchain_core.messages import AIMessage
    from langgraph.config import get_stream_writer
    from langgraph.graph import StateGraph

    from langgraph_openai_serve import GraphConfig, status_event
    from tests.graph.support.schemas import MessageState

    async def generate(_state: MessageState):
        writer = get_stream_writer()
        writer(status_event("Processing step 1"))
        writer(status_event("Processing step 2"))
        return {"messages": [AIMessage(content="done")]}

    graph = (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )
    fastapi_app.state.graph_registry.register(
        "custom-stream-test",
        GraphConfig(
            graph=graph,
            description="Test custom stream",
            streamable_node_names=["generate"],
        ),
    )

    stream = await openai_client.chat.completions.create(
        model="custom-stream-test",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )
    chunks = [chunk async for chunk in stream]

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "done"
    assert all("lgos" not in (chunk.model_extra or {}) for chunk in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop"
