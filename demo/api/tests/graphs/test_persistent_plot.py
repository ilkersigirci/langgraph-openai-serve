import json
from collections.abc import Callable
from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from langgraph.types import CustomStreamPart
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream

from lgos_demo_api.graphs.persistent_plot import (
    PersistentPlotContext,
    PersistentPlotSettings,
    context_factory,
    create_persistent_plot_graph,
    create_persistent_plot_graph_config,
)


def _state(prompt: str) -> dict[str, list[HumanMessage]]:
    return {"messages": [HumanMessage(content=prompt)]}


def _registry() -> GraphRegistry:
    graph = create_persistent_plot_graph(InMemoryStore())
    return GraphRegistry(
        registry={
            "persistent-plot": create_persistent_plot_graph_config(lambda: graph),
        }
    )


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
    request = make_request("persistent-plot", user=user, metadata=metadata)

    with pytest.raises(OpenAIHTTPException) as exc_info:
        context_factory(request, None)

    assert exc_info.value.status_code == 400
    assert exc_info.value.error.param == param
    assert exc_info.value.error.code == "missing_persistence_scope"


async def test_plot_data_is_reused_only_in_the_same_thread() -> None:
    graph = create_persistent_plot_graph(InMemoryStore())
    settings = PersistentPlotSettings()
    first_thread = PersistentPlotContext(
        user_id="user-1", session_id="thread-1", settings=settings
    )
    second_thread = PersistentPlotContext(
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

    assert remembered["messages"][-1].content == "Q3 is highest at $250k."
    assert isolated["messages"][-1].content == "Q4 is highest at $230k."


def _public_event(value: object) -> dict[str, Any]:
    part = cast(CustomStreamPart, value)
    envelope = cast(dict[str, Any], part["data"])
    return cast(dict[str, Any], envelope["event"])


async def test_plot_emits_a_portable_artifact(make_request) -> None:
    request = make_request(
        "persistent-plot",
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

    stream = [
        item
        async for item in run_langgraph_stream(
            request.model,
            request.messages,
            _registry(),
            request,
        )
    ]
    events = [_public_event(item) for item in stream if isinstance(item, dict)]

    assert len(events) == 1
    assert events[0]["type"] == "artifact"
    assert events[0]["data"]["kind"] == "plotly"
    figure = events[0]["data"]["figure"]
    assert figure["data"][0]["type"] == "scatter"
    assert figure["data"][0]["y"] == [120, 180, 150, 230]
    assert figure["layout"]["showlegend"] is False
    assert "€230k" in "".join(item for item in stream if isinstance(item, str))


async def test_plot_supports_non_streaming_completions(make_request) -> None:
    request = make_request(
        "persistent-plot",
        user="user-1",
        metadata={"session_id": "thread-1"},
    )

    invocation = await run_langgraph(
        request.model,
        request.messages,
        _registry(),
        request,
    )

    assert invocation.output.text == "Q4 is highest at $230k."
