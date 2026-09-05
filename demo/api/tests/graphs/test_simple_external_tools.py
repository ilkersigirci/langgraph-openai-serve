from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph_openai_serve import GraphConfig, GraphRegistry
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.graph.runner import run_langgraph

from lgos_demo_api.graphs import simple_external_tools as graph_module

MODEL = "simple-graph-external-tools"
WEATHER_TOOL = {
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


class RecordingModel:
    """Small model double that records binding and invocation inputs."""

    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.bound_tools: Sequence[dict[str, Any]] | None = None
        self.bound_tool_choice: object = None
        self.bound_parallel_tool_calls: bool | None = None
        self.inputs: list[Sequence[BaseMessage]] = []

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any]],
        *,
        tool_choice: object = None,
        parallel_tool_calls: bool | None = None,
    ) -> "RecordingModel":
        self.bound_tools = tools
        self.bound_tool_choice = tool_choice
        self.bound_parallel_tool_calls = parallel_tool_calls
        return self

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.inputs.append(messages)
        return self.response


def _registry() -> GraphRegistry:
    return GraphRegistry(
        registry={
            MODEL: GraphConfig(
                graph=graph_module.simple_external_tools_graph,
                description="DUMMY",
                request_to_input=graph_module.request_to_input,
            )
        }
    )


async def test_client_tools_are_bound_and_returned_to_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingModel(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "Istanbul"},
                    "id": "call-1",
                }
            ],
        )
    )
    monkeypatch.setattr(graph_module, "ChatOpenAI", lambda **_: model)
    request = ResponseCreateRequest(
        model=MODEL,
        input="What is the weather?",
        tools=[{"type": "function", **WEATHER_TOOL["function"]}],
        tool_choice={"type": "function", "name": "get_weather"},
        parallel_tool_calls=False,
    )

    graph_request, messages, _ = decode_responses_request(request)
    result = await run_langgraph(graph_request, messages, _registry())

    assert model.bound_tools == [WEATHER_TOOL]
    assert model.bound_tool_choice == {
        "type": "function",
        "function": {"name": "get_weather"},
    }
    assert model.bound_parallel_tool_calls is False
    assert isinstance(result.output, AIMessage)
    assert result.output.tool_calls is not None
    assert result.output.tool_calls[0]["name"] == "get_weather"


async def test_tool_results_are_forwarded_with_the_complete_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingModel(AIMessage(content="It is sunny in Istanbul."))
    monkeypatch.setattr(graph_module, "ChatOpenAI", lambda **_: model)
    request = ResponseCreateRequest(
        model=MODEL,
        input=[
            {"role": "user", "content": "What is the weather?"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "get_weather",
                "arguments": '{"city":"Istanbul"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": '{"temperature": "sunny"}',
            },
        ],
        tools=[{"type": "function", **WEATHER_TOOL["function"]}],
    )

    graph_request, messages, _ = decode_responses_request(request)
    result = await run_langgraph(graph_request, messages, _registry())

    assert isinstance(result.output, AIMessage)
    assert result.output.content == "It is sunny in Istanbul."
    assert [message.type for message in model.inputs[0]] == [
        "system",
        "human",
        "ai",
        "tool",
    ]
    assert isinstance(model.inputs[0][-1], ToolMessage)
    assert model.inputs[0][-1].tool_call_id == "call-1"
