import json

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from openai import AsyncOpenAI, BadRequestError

from langgraph_openai_serve import (
    ClientFunctionTool,
    GraphConfig,
    GraphRegistry,
    GraphRequest,
    NamedCustomToolChoice,
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


@pytest.mark.parametrize("tool_type", ["web_search_preview", "unsupported_tool"])
async def test_unsupported_tool_type_is_rejected_explicitly(
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
    assert error["param"] == "tools.0"


async def test_server_and_client_tools_are_separated_by_registration(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    received: list[GraphRequest] = []

    def capture(request: GraphRequest, messages: list[BaseMessage]):
        received.append(request)
        return {"messages": messages}

    config = graph_registry.get_graph("test")
    config.server_tools = {"clock", "web_search"}
    config.request_to_input = capture

    response = await openai_client.responses.create(
        model="test",
        input="What time is it and what is the weather?",
        tools=[
            {"type": "custom", "name": "clock"},
            {"type": "web_search"},
            {
                "type": "function",
                "name": "get_weather",
                "description": "Look up weather on the client.",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
        ],
        tool_choice={"type": "custom", "name": "clock"},
    )

    assert received[0].server_tools == ("clock", "web_search")
    assert received[0].tools == (
        ClientFunctionTool(
            name="get_weather",
            description="Look up weather on the client.",
            parameters={"type": "object", "properties": {}},
            strict=True,
        ),
    )
    assert received[0].tool_choice == NamedCustomToolChoice(name="clock")
    assert [tool.type for tool in response.tools] == [
        "custom",
        "web_search",
        "function",
    ]


@pytest.mark.parametrize("stream", [False, True])
async def test_required_web_search_choice_reaches_graph_adapter(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    stream: bool,
) -> None:
    received: list[GraphRequest] = []

    def capture(request: GraphRequest, messages: list[BaseMessage]):
        received.append(request)
        return {"messages": messages}

    config = graph_registry.get_graph("test")
    config.server_tools = {"web_search"}
    config.request_to_input = capture

    result = await openai_client.responses.create(
        model="test",
        input="Search for the latest news.",
        tools=[{"type": "web_search"}],
        tool_choice="required",
        stream=stream,
    )
    if stream:
        events = [event async for event in result]
        assert events[-1].type == "response.completed"
        response = events[-1].response
    else:
        response = result

    assert received[0].server_tools == ("web_search",)
    assert received[0].tool_choice == "required"
    assert response.tool_choice == "required"


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"tools": [{"type": "web_search"}]}, "tools.0.type"),
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "name": "clock",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    }
                ]
            },
            "tools.0.type",
        ),
        (
            {"tools": [{"type": "custom", "name": "missing"}]},
            "tools.0.name",
        ),
        (
            {"tools": [{"type": "custom", "name": "web_search"}]},
            "tools.0.type",
        ),
        ({"tool_choice": "required"}, "tool_choice"),
        ({"tools": [{"type": "custom", "name": "clock"}] * 2}, "tools"),
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    },
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    },
                ]
            },
            "tools",
        ),
    ],
)
async def test_invalid_server_tool_selection_fails_before_execution(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    fields,
    param,
) -> None:
    config = graph_registry.get_graph("test")
    config.server_tools = {"clock"}

    def unexpected_run(request: GraphRequest, messages: list[BaseMessage]):
        pytest.fail("Invalid tool selection must fail before graph preparation.")

    config.request_to_input = unexpected_run
    with pytest.raises(BadRequestError) as error:
        await openai_client.responses.create(model="test", input="Time?", **fields)

    assert error.value.response.json()["error"]["param"] == param


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
    assert [part["id"] for part in assistant.content] == ["fc_weather", "fc_clock"]
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


@pytest.mark.parametrize("stream", [False, True])
async def test_truncated_function_arguments_are_incomplete_not_server_errors(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    stream: bool,
) -> None:
    config = graph_registry.get_graph("test")
    config.streamable_node_names = []
    config.output_to_message = lambda _: AIMessage(
        content="",
        invalid_tool_calls=[
            {"id": "call_weather", "name": "weather", "args": '{"city":"Ista'}
        ],
        response_metadata={"finish_reason": "length"},
    )

    if stream:
        async with openai_client.responses.stream(
            model="test", input="Weather?"
        ) as response_stream:
            events = [event async for event in response_stream]
        assert events[-1].type == "response.incomplete"
        response = events[-1].response
        assert [
            event.delta
            for event in events
            if event.type == "response.function_call_arguments.delta"
        ] == ['{"city":"Ista']
    else:
        response = await openai_client.responses.create(model="test", input="Weather?")

    assert response.status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"
    assert len(response.output) == 1
    call = response.output[0]
    assert call.type == "function_call"
    assert call.status == "incomplete"
    assert call.call_id == "call_weather"
    assert call.name == "weather"
    assert call.arguments == '{"city":"Ista'


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
            "must match an earlier tool call",
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
