import json
import re
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore
from langgraph_openai_serve import GraphRegistry, LanggraphOpenaiServe
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.service import generate_response
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.graph.runner import run_langgraph
from langgraph_openai_serve.graph.utils import prepare_run
from openai import AsyncOpenAI
from openai.types.responses import ResponseCompletedEvent, ResponseFunctionToolCall
from plotly import io as pio

from lgos_demo_api.graphs import persistent_plot_agent as plot_module
from lgos_demo_api.graphs.persistent_plot_agent import (
    DISPLAY_FILE_TOOL_NAME,
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


class StreamingToolCallingChatModel(FakeMessagesListChatModel):
    """Emit modern tool calls and text through the model streaming path."""

    def bind_tools(self, tools: list[BaseTool], **_kwargs: Any) -> Self:  # ty: ignore[invalid-method-override]
        return self

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        message = self._generate(*args, **kwargs).generations[0].message
        chunks = [
            tool_call_chunk(
                name=call["name"],
                args=json.dumps(call["args"]),
                id=call["id"],
                index=index,
            )
            for index, call in enumerate(message.tool_calls)
        ]
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                tool_call_chunks=chunks,
                chunk_position="last",
            )
        )


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
    make_request: Callable[..., ResponseCreateRequest],
    user: str | None,
    metadata: dict[str, str] | None,
    param: str,
) -> None:
    request = make_request("persistent-plot-agent", user=user, metadata=metadata)

    with pytest.raises(OpenAIHTTPException) as exc_info:
        context_factory(decode_responses_request(request)[0], None)

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


@pytest.mark.parametrize(
    ("chart_type", "currency", "show_legend", "trace_type"),
    [("line", "EUR", False, "scatter"), ("bar", "USD", True, "bar")],
)
async def test_agent_uploads_plotly_and_returns_display_file_call(
    monkeypatch: pytest.MonkeyPatch,
    make_tool_calling_model: Callable[..., BaseChatModel],
    chart_type: str,
    currency: str,
    show_legend: bool,
    trace_type: str,
) -> None:
    create_file = AsyncMock(return_value=SimpleNamespace(id="file-chart"))

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.files = SimpleNamespace(create=create_file)

        async def __aenter__(self) -> "FakeOpenAI":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(plot_module, "AsyncOpenAI", FakeOpenAI)
    request = ResponseCreateRequest.model_validate(
        {
            "model": "persistent-plot-agent",
            "input": "Show the chart.",
            "tools": [_display_file_tool()],
            "store": False,
            "user": "user-1",
            "metadata": {
                "session_id": "thread-1",
                "langgraph_runtime_settings": json.dumps(
                    {
                        "chart_type": chart_type,
                        "currency": currency,
                        "show_legend": show_legend,
                    }
                ),
            },
        }
    )
    registry = _registry(
        make_tool_calling_model(
            _tool_call("show_quarterly_revenue", {}, "show-1"),
            AIMessage(content="Q4 is highest at €230k."),
        )
    )
    graph_request, messages, resume = decode_responses_request(request)
    run = await prepare_run(graph_request, messages, registry, resume=resume)

    response = await generate_response(request, run)

    assert [item.type for item in response.output] == ["function_call"]
    display_call = response.output[0]
    assert isinstance(display_call, ResponseFunctionToolCall)
    assert display_call.name == DISPLAY_FILE_TOOL_NAME
    arguments = json.loads(display_call.arguments)
    expected_filename = arguments.pop("filename")
    assert re.fullmatch(
        r"quarterly-revenue-[0-9a-f]{12}\.plotly\.json",
        expected_filename,
    )
    assert arguments == {
        "file_id": "file-chart",
        "media_type": "application/vnd.plotly.v1+json",
        "title": "Quarterly revenue",
        "alt": f"Q4 is highest at {'€' if currency == 'EUR' else '$'}230k.",
    }
    assert display_call.call_id.startswith("lg_display_")

    create_file.assert_awaited_once()
    filename, content, media_type = create_file.await_args.kwargs["file"]
    assert filename == expected_filename
    assert media_type == "application/vnd.plotly.v1+json"
    figure = pio.from_json(content.decode())
    assert figure.data[0].type == trace_type
    assert list(figure.data[0].x) == ["Q1", "Q2", "Q3", "Q4"]
    assert list(figure.data[0].y) == [120, 180, 150, 230]
    assert figure.layout.yaxis.title.text == f"Revenue ({currency}, thousands)"
    assert figure.layout.showlegend is show_legend
    if chart_type == "line":
        assert figure.data[0].mode == "lines+markers"
    assert create_file.await_args.kwargs["purpose"] == "user_data"


async def test_streaming_agent_completes_with_display_file_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_file = AsyncMock(return_value=SimpleNamespace(id="file-chart"))

    class FakeOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            self.files = SimpleNamespace(create=create_file)

        async def __aenter__(self) -> "FakeOpenAI":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(plot_module, "AsyncOpenAI", FakeOpenAI)
    registry = _registry(
        StreamingToolCallingChatModel(
            responses=[
                _tool_call("show_quarterly_revenue", {}, "show-1"),
                AIMessage(content="Q4 is highest at $230k."),
            ]
        )
    )
    app = LanggraphOpenaiServe(graphs=registry).bind_openai_api().app
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(
            transport=transport,
            base_url="http://test",
            timeout=2,
        ) as http_client,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        ) as client,
    ):
        stream = await client.responses.create(
            model="persistent-plot-agent",
            input="Show the chart.",
            metadata={"session_id": "thread-1"},
            store=False,
            stream=True,
            tools=[_display_file_tool()],
            user="user-1",
        )
        events = [event async for event in stream]

    completed = next(
        (event for event in events if isinstance(event, ResponseCompletedEvent)),
        None,
    )
    assert completed is not None
    assert events[-1] is completed
    assert [item.type for item in completed.response.output] == ["function_call"]


def _display_file_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": DISPLAY_FILE_TOOL_NAME,
        "description": "Display a file from the OpenAI Files API.",
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                name: {"type": "string"}
                for name in ("file_id", "filename", "media_type", "title", "alt")
            },
            "required": ["file_id", "filename", "media_type", "title", "alt"],
        },
    }


async def test_agent_does_not_upload_when_display_tool_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    make_request,
    make_tool_calling_model: Callable[..., BaseChatModel],
) -> None:
    client = AsyncMock()
    monkeypatch.setattr(plot_module, "AsyncOpenAI", client)
    request = make_request(
        "persistent-plot-agent",
        user="user-1",
        metadata={"session_id": "thread-1"},
    )
    registry = _registry(
        make_tool_calling_model(
            _tool_call("show_quarterly_revenue", {}, "show-1"),
            AIMessage(content="Q4 is highest at €230k."),
        )
    )

    graph_request, messages, _ = decode_responses_request(request)

    invocation = await run_langgraph(graph_request, messages, registry)

    assert isinstance(invocation.output, AIMessage)
    assert invocation.output.text == "Q4 is highest at €230k."
    assert not invocation.output.tool_calls
    client.assert_not_called()


async def test_agent_supports_non_streaming_invocation(
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

    graph_request, messages, _ = decode_responses_request(request)

    invocation = await run_langgraph(graph_request, messages, registry)

    output = invocation.output
    assert (output.text if isinstance(output, AIMessage) else output) == (
        "Q4 is highest at $230k."
    )
