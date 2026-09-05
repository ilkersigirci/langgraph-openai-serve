import json
from pathlib import Path

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

FIXTURES = Path(__file__).with_name("fixtures")


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


async def test_hosted_tools_are_rejected_explicitly(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model="test",
            input="Search.",
            tools=[{"type": "web_search_preview"}],
        )

    error = exc_info.value.response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "tools.0.type"


async def test_function_calls_and_outputs_become_ordered_langchain_messages(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
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
                "arguments": "{",
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
    assert assistant.invalid_tool_calls[0]["args"] == "{"
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


async def test_function_call_stream_matches_golden_lifecycle(
    tool_openai_client: AsyncOpenAI,
) -> None:
    stream = await tool_openai_client.responses.create(
        model="tools",
        input="Hi",
        stream=True,
    )
    events = [event async for event in stream]
    with FIXTURES.joinpath("function_call_stream.json").open(
        encoding="utf-8"
    ) as fixture:
        expected = json.load(fixture)

    one_call_event_types = [payload["type"] for payload in expected]
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
