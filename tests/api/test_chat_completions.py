import json

import pytest
from httpx2 import AsyncClient
from langchain_core.messages import AIMessage
from openai import AsyncOpenAI, InternalServerError, NotFoundError
from starlette import status

from langgraph_openai_serve import (
    ClientFunctionTool,
    GraphRegistry,
    GraphRequest,
)
from tests.graph.support.message import make_message_graph
from tests.graph.support.registration import (
    capture_graph_inputs,
    replace_graph_config,
)

TOOL_CALLS = [
    {"name": "weather", "args": {"city": "Istanbul"}, "id": "call_weather"},
    {"name": "translate", "args": {"text": "hello"}, "id": "call_translate"},
]


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
    inputs = capture_graph_inputs(graph_registry)

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
    assert inputs.requests == [
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


@pytest.mark.parametrize("stream", [False, True])
async def test_final_tool_calls_end_the_completion(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    stream: bool,
) -> None:
    replace_graph_config(
        graph_registry,
        "test",
        graph=make_message_graph(disable_streaming=True),
        output_to_message=lambda _output: AIMessage(content="", tool_calls=TOOL_CALLS),
    )
    request = {"model": "test", "messages": [{"role": "user", "content": "Hi"}]}

    if stream:
        chunks = [
            chunk
            async for chunk in await openai_client.chat.completions.create(
                **request, stream=True
            )
        ]
        tool_calls = [
            call for chunk in chunks for call in chunk.choices[0].delta.tool_calls or ()
        ]
        finish_reason = chunks[-1].choices[0].finish_reason
    else:
        completion = await openai_client.chat.completions.create(**request)
        tool_calls = completion.choices[0].message.tool_calls or []
        finish_reason = completion.choices[0].finish_reason

    assert finish_reason == "tool_calls"
    assert [
        (call.id, call.function.name, json.loads(call.function.arguments))
        for call in tool_calls
    ] == [(call["id"], call["name"], call["args"]) for call in TOOL_CALLS]


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "weather", "args": {"day": float("nan")}, "id": "call_1"}
                ],
            ),
            id="non-json-arguments",
        ),
        pytest.param(
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {"name": "weather", "args": "{", "id": "call_1", "error": None}
                ],
            ),
            id="invalid-call",
        ),
        pytest.param(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "weather", "args": {}, "id": "call_1"},
                    {"name": "translate", "args": {}, "id": "call_1"},
                ],
            ),
            id="repeated-call-id",
        ),
    ],
)
async def test_unservable_tool_calls_fail_on_both_apis(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    message: AIMessage,
) -> None:
    replace_graph_config(
        graph_registry,
        "test",
        graph=make_message_graph(disable_streaming=True),
        output_to_message=lambda _output: message,
    )

    with pytest.raises(InternalServerError):
        await openai_client.chat.completions.create(
            model="test", messages=[{"role": "user", "content": "Hi"}]
        )
    with pytest.raises(InternalServerError):
        await openai_client.responses.create(model="test", input="Hi")


@pytest.mark.parametrize(
    ("identity", "end_user"),
    [
        ({"user": "legacy-user"}, "legacy-user"),
        ({"safety_identifier": "user-123"}, "user-123"),
        ({"safety_identifier": "user-123", "user": "legacy-user"}, "user-123"),
    ],
    ids=["deprecated-user", "safety-identifier", "both"],
)
async def test_chat_safety_identifier_names_the_end_user(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    identity: dict[str, str],
    end_user: str,
) -> None:
    inputs = capture_graph_inputs(graph_registry)

    await openai_client.chat.completions.create(
        model="test", messages=[{"role": "user", "content": "hello"}], **identity
    )

    assert [request.user for request in inputs.requests] == [end_user]


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
async def test_unknown_model_is_not_found(
    openai_client: AsyncOpenAI,
    stream: bool,
) -> None:
    with pytest.raises(NotFoundError) as exc_info:
        await openai_client.chat.completions.create(
            model="missing",
            messages=[{"role": "user", "content": "Hi"}],
            stream=stream,
        )

    assert exc_info.value.code == "model_not_found"


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
    fastapi_app.state.graph_registry.graphs["custom-stream-test"] = GraphConfig(
        graph=graph,
        description="Test custom stream",
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
