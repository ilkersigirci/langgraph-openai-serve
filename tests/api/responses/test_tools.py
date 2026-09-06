import json

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from openai import AsyncOpenAI, BadRequestError

from langgraph_openai_serve import (
    ClientFunctionTool,
    GraphConfig,
    GraphRegistry,
    GraphRequest,
    NamedFunctionToolChoice,
)
from tests.graph.support.message import make_message_graph


async def test_function_tools_and_choices_reach_graph_adapter(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    received: list[GraphRequest] = []

    def capture(
        request: GraphRequest,
        messages: list[BaseMessage],
    ) -> dict[str, list[BaseMessage]]:
        received.append(request)
        return {"messages": messages}

    graph_registry.get_graph("test").request_to_input = capture

    response = await openai_client.responses.create(
        model="test",
        input="Weather?",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
                "strict": True,
            }
        ],
        tool_choice={"type": "function", "name": "get_weather"},
        parallel_tool_calls=False,
    )

    assert received == [
        GraphRequest(
            model="test",
            metadata={},
            user=None,
            tools=(
                ClientFunctionTool(
                    name="get_weather",
                    description="Get the weather.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                    strict=True,
                ),
            ),
            tool_choice=NamedFunctionToolChoice(name="get_weather"),
            parallel_tool_calls=False,
        )
    ]
    assert response.parallel_tool_calls is False
    assert response.tool_choice.type == "function"
    assert response.tool_choice.name == "get_weather"
    assert response.tools[0].type == "function"
    assert response.tools[0].name == "get_weather"


@pytest.mark.parametrize("tool_type", ["web_search_preview", "lgos_clock"])
async def test_unsupported_tool_types_are_rejected_explicitly(
    openai_client: AsyncOpenAI,
    tool_type: str,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model="test",
            input="Search.",
            tools=[{"type": tool_type}],
        )

    error = exc_info.value.response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "tools.0.type"


@pytest.mark.parametrize("invalid_arguments", ["{", '{"value":NaN}', '{"value":1e999}'])
async def test_function_calls_and_outputs_become_ordered_langchain_messages(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    invalid_arguments: str,
) -> None:
    received: list[list[BaseMessage]] = []

    def capture(
        _request: GraphRequest,
        messages: list[BaseMessage],
    ) -> dict[str, list[BaseMessage]]:
        received.append(messages)
        return {"messages": messages}

    graph_registry.get_graph("test").request_to_input = capture

    await openai_client.responses.create(
        model="test",
        input=[
            {
                "type": "function_call",
                "id": "fc_weather",
                "call_id": "call_weather",
                "name": "weather",
                "arguments": '{"city":"Istanbul"}',
                "status": "completed",
            },
            {
                "type": "function_call",
                "id": "fc_clock",
                "call_id": "call_clock",
                "name": "clock",
                "arguments": invalid_arguments,
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_weather",
                "output": "sunny",
            },
            {
                "type": "function_call_output",
                "call_id": "call_clock",
                "output": "noon",
            },
        ],
    )

    assert len(received) == 1
    messages = received[0]
    assert [type(message) for message in messages] == [
        AIMessage,
        ToolMessage,
        ToolMessage,
    ]
    assistant = messages[0]
    assert isinstance(assistant, AIMessage)
    assert assistant.tool_calls == [
        {
            "name": "weather",
            "args": {"city": "Istanbul"},
            "id": "call_weather",
            "type": "tool_call",
        }
    ]
    assert len(assistant.invalid_tool_calls) == 1
    assert assistant.invalid_tool_calls[0]["id"] == "call_clock"
    assert assistant.invalid_tool_calls[0]["args"] == invalid_arguments
    assert "not valid JSON" in (assistant.invalid_tool_calls[0]["error"] or "")
    assert [message.tool_call_id for message in messages[1:]] == [
        "call_weather",
        "call_clock",
    ]


@pytest.mark.parametrize(
    ("input_items", "message"),
    [
        pytest.param(
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_missing",
                    "output": "result",
                }
            ],
            "must match an earlier function call",
            id="unmatched-output",
        ),
        pytest.param(
            [
                {
                    "type": "function_call",
                    "call_id": "call_unanswered",
                    "name": "weather",
                    "arguments": "{}",
                },
            ],
            "missing outputs for call_unanswered",
            id="missing-output",
        ),
        pytest.param(
            [
                {
                    "type": "function_call",
                    "id": "fc_one",
                    "call_id": "call_duplicate",
                    "name": "one",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "id": "fc_two",
                    "call_id": "call_duplicate",
                    "name": "two",
                    "arguments": "{}",
                },
            ],
            "duplicate call_id",
            id="duplicate-call-id",
        ),
    ],
)
async def test_ambiguous_function_continuations_are_rejected(
    openai_client: AsyncOpenAI,
    input_items: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(model="test", input=input_items)

    error = exc_info.value.response.json()["error"]
    assert error["param"] == "input"
    assert message in error["message"]


@pytest.fixture
def tool_openai_client(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> AsyncOpenAI:
    tool_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "weather",
                "args": {"city": "Istanbul"},
                "id": "call_weather",
                "type": "tool_call",
            },
            {
                "name": "clock",
                "args": {"timezone": "Europe/Istanbul"},
                "id": "call_clock",
                "type": "tool_call",
            },
        ],
    )
    graph_registry.register(
        "tools",
        GraphConfig(
            graph=make_message_graph(),
            description="DUMMY",
            output_to_message=lambda _output: tool_message,
        ),
    )
    return openai_client


async def test_multiple_tool_calls_are_distinct_response_output_items(
    tool_openai_client: AsyncOpenAI,
) -> None:
    response = await tool_openai_client.responses.create(model="tools", input="Hi")

    assert [item.type for item in response.output] == [
        "function_call",
        "function_call",
    ]
    assert [item.call_id for item in response.output] == [
        "call_weather",
        "call_clock",
    ]
    assert [item.name for item in response.output] == ["weather", "clock"]
    assert [json.loads(item.arguments) for item in response.output] == [
        {"city": "Istanbul"},
        {"timezone": "Europe/Istanbul"},
    ]
    assert all(item.id.startswith("fc_") for item in response.output)


async def test_function_call_stream_has_complete_lifecycle(
    tool_openai_client: AsyncOpenAI,
) -> None:
    stream = await tool_openai_client.responses.create(
        model="tools",
        input="Hi",
        stream=True,
    )
    events = [event async for event in stream]
    one_call_event_types = [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event.type for event in events] == [
        *one_call_event_types[:-1],
        *one_call_event_types[2:],
    ]
    assert [event.sequence_number for event in events] == list(range(len(events)))

    added = [event for event in events if event.type == "response.output_item.added"]
    deltas = [
        event
        for event in events
        if event.type == "response.function_call_arguments.delta"
    ]
    done = [event for event in events if event.type == "response.output_item.done"]
    assert [event.output_index for event in added] == [0, 1]
    assert [event.item.call_id for event in added] == [
        "call_weather",
        "call_clock",
    ]
    assert [event.item.arguments for event in added] == ["", ""]
    assert [json.loads(event.delta) for event in deltas] == [
        {"city": "Istanbul"},
        {"timezone": "Europe/Istanbul"},
    ]
    assert [event.item.id for event in done] == [event.item.id for event in added]
    assert events[-1].type == "response.completed"
    assert [item.id for item in events[-1].response.output] == [
        event.item.id for event in done
    ]


@pytest.mark.parametrize("stream", [False, True])
async def test_hosted_selector_reaches_graph_without_a_function_schema(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    stream: bool,
) -> None:
    received: list[GraphRequest] = []

    def capture(
        request: GraphRequest, messages: list[BaseMessage]
    ) -> dict[str, list[BaseMessage]]:
        received.append(request)
        return {"messages": messages}

    config = graph_registry.get_graph("test")
    config.hosted_tools = {"lgos_clock"}
    config.request_to_input = capture
    response = await openai_client.responses.create(
        model="test",
        input="Time?",
        stream=stream,
        tools=[
            {"type": "custom", "name": "lgos_clock"},
            {"type": "function", "name": "client_tool"},
        ],
    )
    if stream:
        events = [event async for event in response]
        response = next(
            event.response for event in events if event.type == "response.completed"
        )
    assert received[0].hosted_tools == ("lgos_clock",)
    assert [tool.name for tool in received[0].tools] == ["client_tool"]
    assert response.tools[0].type == "custom"
    assert response.tools[0].name == "lgos_clock"
    assert response.tools[1].name == "client_tool"


@pytest.mark.parametrize("stream", [False, True])
async def test_unregistered_hosted_tool_is_rejected_before_execution(
    openai_client: AsyncOpenAI, stream: bool
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model="test",
            input="Time?",
            stream=stream,
            tools=[{"type": "custom", "name": "lgos_unknown"}],
        )
    assert exc_info.value.response.json()["error"]["param"] == "tools.0.name"


async def test_hosted_tool_rejects_client_supplied_parameters(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model="test",
            input="Time?",
            tools=[{"type": "custom", "name": "lgos_clock", "parameters": {}}],
        )
    assert exc_info.value.response.json()["error"]["param"] == "tools.0.parameters"


async def test_custom_tool_with_invalid_name_is_rejected(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model="test",
            input="Time?",
            tools=[{"type": "custom", "name": "not_an_lgos_tool"}],
        )
    assert exc_info.value.response.json()["error"]["param"] == "tools.0.name"
