import json
from collections.abc import Callable
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore
from langgraph.types import CustomStreamPart
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream

from lgos_demo_api.graphs.persistent_plot_agent import (
    PersistentPlotAgentContext,
    PersistentPlotAgentSettings,
    context_factory,
    create_persistent_plot_agent,
    create_persistent_plot_agent_config,
)


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def _state(prompt: str) -> dict[str, list[HumanMessage]]:
    return {"messages": [HumanMessage(content=prompt)]}


def _registry(model: BaseChatModel) -> GraphRegistry:
    graph = create_persistent_plot_agent(InMemoryStore(), model)
    return GraphRegistry(
        registry={
            "persistent-plot-agent": create_persistent_plot_agent_config(lambda: graph),
        }
    )


def _last_tool_result(result: dict[str, Any]) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, ToolMessage):
            return message.text
    msg = "Agent result has no tool message."
    raise AssertionError(msg)


@pytest.mark.parametrize(
    ("user", "metadata", "param"),
    [
        (None, {"session_id": "thread-1"}, "user"),
        ("user-1", None, "metadata.session_id"),
    ],
)
def test_plot_requires_a_complete_persistence_scope(
    make_request: Callable[..., ChatCompletionRequest],
    user: str | None,
    metadata: dict[str, str] | None,
    param: str,
) -> None:
    request = make_request("persistent-plot-agent", user=user, metadata=metadata)

    with pytest.raises(OpenAIHTTPException) as exc_info:
        context_factory(request, None)

    assert exc_info.value.status_code == 400
    assert exc_info.value.error.param == param
    assert exc_info.value.error.code == "missing_persistence_scope"


async def test_agent_reuses_plot_data_only_in_the_same_thread(
    make_tool_calling_model: Callable[..., BaseChatModel],
) -> None:
    store = InMemoryStore()
    graph = create_persistent_plot_agent(
        store,
        make_tool_calling_model(
            _tool_call(
                "update_quarterly_revenue",
                {"updates": [{"quarter": "Q3", "revenue": 250}]},
                "update-1",
            ),
            AIMessage(content="Updated Q3."),
            _tool_call("show_quarterly_revenue", {}, "show-1"),
            AIMessage(content="Q3 is highest."),
            _tool_call("show_quarterly_revenue", {}, "show-2"),
            AIMessage(content="Q4 is highest."),
        ),
    )
    settings = PersistentPlotAgentSettings()
    first_thread = PersistentPlotAgentContext(
        user_id="user-1", session_id="thread-1", settings=settings
    )
    second_thread = PersistentPlotAgentContext(
        user_id="user-1", session_id="thread-2", settings=settings
    )

    await graph.ainvoke(_state("Set Q3 to 250."), context=first_thread)
    remembered = await graph.ainvoke(
        _state("Which quarter is highest?"),
        context=first_thread,
    )
    isolated = await graph.ainvoke(
        _state("Which quarter is highest?"),
        context=second_thread,
    )

    assert "Q3=250k" in _last_tool_result(remembered)
    assert "Q3=150k" in _last_tool_result(isolated)


def _public_event(value: object) -> dict[str, Any]:
    part = cast(CustomStreamPart, value)
    envelope = cast(dict[str, Any], part["data"])
    return cast(dict[str, Any], envelope["event"])


async def test_agent_emits_a_portable_chart_artifact(
    make_request,
    make_tool_calling_model: Callable[..., BaseChatModel],
) -> None:
    request = make_request(
        "persistent-plot-agent",
        user="user-1",
        metadata={
            "session_id": "thread-1",
            "langgraph_stream_events": "v1",
            "langgraph_runtime_settings": json.dumps(
                {
                    "chart_type": "line",
                    "currency": "EUR",
                    "show_legend": False,
                }
            ),
        },
    )
    registry = _registry(
        make_tool_calling_model(
            _tool_call("show_quarterly_revenue", {}, "show-1"),
            AIMessage(content="Q4 is highest at €230k."),
        )
    )

    stream = [
        item
        async for item in run_langgraph_stream(
            request.model,
            request.messages,
            registry,
            request,
        )
    ]
    events = [_public_event(item) for item in stream if isinstance(item, dict)]

    assert len(events) == 1
    assert events[0]["type"] == "artifact"
    artifact = events[0]["data"]
    assert artifact["kind"] == "chart"
    assert artifact["chart_type"] == "line"
    assert artifact["labels"] == ["Q1", "Q2", "Q3", "Q4"]
    assert artifact["series"] == [{"name": "Revenue", "values": [120, 180, 150, 230]}]
    assert artifact["show_legend"] is False
    assert any(isinstance(item, AIMessage) and "€230k" in item.text for item in stream)


async def test_agent_supports_non_streaming_completions(
    make_request,
    make_tool_calling_model: Callable[..., BaseChatModel],
) -> None:
    request = make_request(
        "persistent-plot-agent",
        user="user-1",
        metadata={"session_id": "thread-1"},
    )
    registry = _registry(
        make_tool_calling_model(
            _tool_call("show_quarterly_revenue", {}, "show-1"),
            AIMessage(content="Q4 is highest at $230k."),
        )
    )

    invocation = await run_langgraph(
        request.model,
        request.messages,
        registry,
        request,
    )

    output = invocation.output
    assert (output.text if isinstance(output, AIMessage) else output) == (
        "Q4 is highest at $230k."
    )
