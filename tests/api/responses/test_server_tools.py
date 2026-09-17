"""Responses-compatible contracts for server-executed tools."""

import json
from typing import Any

import pytest
from anyio import Event, fail_after
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, MessagesState, StateGraph
from openai import AsyncOpenAI, InternalServerError
from openai.types.responses import Response, ResponseStreamEvent

from langgraph_openai_serve import (
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    GraphRequest,
)
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.server_tools import ServerToolTracker
from langgraph_openai_serve.api.responses.service import stream_response
from langgraph_openai_serve.graph.events import status_event
from langgraph_openai_serve.graph.utils import prepare_run
from tests.graph.support.registration import replace_graph_config

CALL = {
    "id": "call_package",
    "name": "package_version",
    "args": {"__arg1": "langgraph"},
}
CALL_ITEM = {
    "id": "ctc_package",
    "type": "custom_tool_call",
    "status": "completed",
    "call_id": "call_package",
    "name": "package_version",
    "input": "langgraph",
}
PACKAGE_TOOLS = [{"type": "custom", "name": "package_version"}]
SEARCH_TOOLS = [{"type": "web_search"}]
SEARCH_CALL = {
    "id": "call_search",
    "name": "web_search",
    "args": {"query": "OpenAI Responses API"},
}
WEB_SEARCH = {
    "type": "web_search_call",
    "id": "ws_call_search",
    "status": "completed",
    "action": {"type": "search", "query": "OpenAI Responses API"},
}


def test_nested_updates_do_not_expose_server_tool_activity() -> None:
    event: Any = {
        "type": "updates",
        "ns": ("subgraph:run",),
        "data": {"tools": {"messages": [AIMessage(content="", tool_calls=[CALL])]}},
    }

    assert list(ServerToolTracker({"package_version"}).items(event)) == []


def _register_single_node(
    registry: GraphRegistry,
    name: str,
    node: Any,
    *,
    server_tools: set[str],
) -> None:
    graph = (
        StateGraph(MessagesState)
        .add_node("answer", node)
        .set_entry_point("answer")
        .set_finish_point("answer")
        .compile()
    )
    registry.register(
        name,
        GraphConfig(
            graph=graph,
            description=name,
            server_tools=server_tools,
        ),
    )


async def _create(
    client: AsyncOpenAI,
    *,
    stream: bool = False,
    **request: Any,
) -> tuple[Response, list[ResponseStreamEvent]]:
    result = await client.responses.create(stream=stream, **request)
    if not stream:
        assert isinstance(result, Response)
        return result, []

    events = [event async for event in result]
    terminal = events[-1]
    assert terminal.type in {
        "response.completed",
        "response.failed",
        "response.incomplete",
    }
    return terminal.response, events


async def test_chat_completions_can_use_a_server_tool_graph_without_tools(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    replace_graph_config(
        graph_registry,
        "test",
        server_tools={"package_version"},
    )

    response = await openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert response.choices[0].message.content == "hello"


async def test_server_output_contains_only_selected_executed_calls(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    async def answer(_state: MessagesState):
        return {
            "messages": [
                AIMessage(content=[CALL_ITEM], tool_calls=[CALL, SEARCH_CALL]),
                ToolMessage(
                    content=[
                        {
                            "type": "custom_tool_call_output",
                            "output": "langgraph==installed-version",
                        }
                    ],
                    tool_call_id="call_package",
                ),
                ToolMessage(
                    content="search results",
                    name="web_search",
                    tool_call_id="call_search",
                ),
                AIMessage(content="Done."),
            ]
        }

    _register_single_node(
        graph_registry,
        "tools",
        answer,
        server_tools={"package_version", "web_search"},
    )
    response, _ = await _create(
        openai_client,
        model="tools",
        input="Help.",
        tools=[*PACKAGE_TOOLS, *SEARCH_TOOLS],
    )

    assert [item.type for item in response.output] == [
        "custom_tool_call",
        "custom_tool_call_output",
        "web_search_call",
        "message",
    ]
    assert response.output[0].input == "langgraph"
    assert response.output[1].call_id == response.output[0].call_id
    assert response.output[1].output == "langgraph==installed-version"


async def test_tool_choice_none_exposes_no_server_tools_to_the_graph(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    received: list[GraphRequest] = []

    def capture(request: GraphRequest, messages: list[BaseMessage]):
        received.append(request)
        return {"messages": messages}

    replace_graph_config(
        graph_registry,
        "test",
        server_tools={"package_version"},
        request_to_input=capture,
    )

    await openai_client.responses.create(
        model="test",
        input="Hello.",
        tools=PACKAGE_TOOLS,
        tool_choice="none",
    )

    assert received[0].server_tools == ()


@pytest.mark.parametrize("stream", [False, True])
async def test_server_execution_can_finish_with_a_client_function_call(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    stream: bool,
) -> None:
    async def answer(_state: MessagesState):
        return {
            "messages": [
                AIMessage(content=[CALL_ITEM], tool_calls=[CALL]),
                ToolMessage(
                    content=[
                        {
                            "type": "custom_tool_call_output",
                            "output": "langgraph==installed-version",
                        }
                    ],
                    tool_call_id="call_package",
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_report",
                            "name": "save_report",
                            "args": {"title": "Dependency report"},
                        }
                    ],
                ),
            ]
        }

    _register_single_node(
        graph_registry,
        "package",
        answer,
        server_tools={"package_version"},
    )
    response, _ = await _create(
        openai_client,
        model="package",
        input="Save a dependency report.",
        tools=[
            *PACKAGE_TOOLS,
            {
                "type": "function",
                "name": "save_report",
                "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ],
        stream=stream,
    )

    assert response.status == "completed"
    assert [item.type for item in response.output] == [
        "custom_tool_call",
        "custom_tool_call_output",
        "function_call",
    ]
    assert [item.call_id for item in response.output] == [
        "call_package",
        "call_package",
        "call_report",
    ]
    assert response.output[1].output == "langgraph==installed-version"
    assert json.loads(response.output[2].arguments) == {"title": "Dependency report"}


@pytest.mark.parametrize("stream", [False, True])
async def test_private_tools_and_nonstream_status_stay_out_of_output(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    stream: bool,
) -> None:
    async def answer(_state: MessagesState):
        get_stream_writer()(status_event("Checking."))
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "private_lookup",
                            "args": {"secret": "internal"},
                            "id": "call_private",
                        }
                    ],
                ),
                ToolMessage(content="private result", tool_call_id="call_private"),
                AIMessage(content="Done."),
            ]
        }

    _register_single_node(
        graph_registry,
        "package",
        answer,
        server_tools={"package_version"},
    )
    replace_graph_config(
        graph_registry,
        "package",
        features={GraphFeature.CLIENT_EVENTS},
    )

    response, _ = await _create(
        openai_client,
        model="package",
        input="Help.",
        tools=PACKAGE_TOOLS,
        stream=stream,
    )

    assert all(item.type == "message" for item in response.output)
    assert [item.phase for item in response.output] == (
        ["commentary", "final_answer"] if stream else ["final_answer"]
    )
    assert response.output_text == ("Checking.Done." if stream else "Done.")


async def test_server_custom_tool_exchange_events_and_replay(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    executions = []
    replayed: list[list[BaseMessage]] = []

    async def choose(state: MessagesState):
        if len(state["messages"]) > 1:
            replayed.append(state["messages"])
            return {"messages": AIMessage(content="The previous result was preserved.")}
        executions.append("package_version")
        return {
            "messages": [
                AIMessage(content=[{**CALL_ITEM, "index": 0}], tool_calls=[CALL]),
                ToolMessage(
                    content=[
                        {
                            "type": "custom_tool_call_output",
                            "output": "langgraph==installed-version",
                        }
                    ],
                    tool_call_id="call_package",
                ),
            ]
        }

    async def answer(_state: MessagesState):
        return {"messages": [AIMessage(content="The package version was found.")]}

    graph = (
        StateGraph(MessagesState)
        .add_node("choose", choose)
        .add_node("answer", answer)
        .set_entry_point("choose")
        .add_conditional_edges(
            "choose",
            lambda state: (
                END
                if state["messages"][-1].text == "The previous result was preserved."
                else "answer"
            ),
        )
        .set_finish_point("answer")
        .compile()
    )
    graph_registry.register(
        "package",
        GraphConfig(
            graph=graph,
            description="Package",
            server_tools={"package_version"},
        ),
    )

    response, events = await _create(
        openai_client,
        model="package",
        input="Version?",
        tools=PACKAGE_TOOLS,
        stream=True,
    )

    assert [item.type for item in response.output] == [
        "custom_tool_call",
        "custom_tool_call_output",
        "message",
    ]
    assert all(item.status == "completed" for item in response.output)
    assert [
        event.item.status
        for event in events
        if event.type == "response.output_item.added"
        and event.item.type in {"custom_tool_call", "custom_tool_call_output"}
    ] == ["in_progress", "completed"]
    assert [
        event.input
        for event in events
        if event.type == "response.custom_tool_call_input.done"
    ] == ["langgraph"]

    second = await openai_client.responses.create(
        model="package",
        tools=PACKAGE_TOOLS,
        input=[
            {"role": "user", "content": "Version?"},
            *response.output,
            {"role": "user", "content": "What version was found?"},
        ],
    )

    assert second.output_text == "The previous result was preserved."
    assert executions == ["package_version"]
    call_message, tool_message = replayed[0][1:3]
    assert isinstance(call_message, AIMessage)
    assert call_message.content[0]["id"] == response.output[0].id
    assert call_message.tool_calls[0]["args"] == {"__arg1": "langgraph"}
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == [
        {
            "type": "custom_tool_call_output",
            "output": "langgraph==installed-version",
        }
    ]


async def test_web_search_output_events_and_replay(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    received: list[list[BaseMessage]] = []

    async def answer(state: MessagesState):
        received.append(state["messages"])
        if len(state["messages"]) > 1:
            return {"messages": [AIMessage(content="The search found OpenAI docs.")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[SEARCH_CALL],
                ),
                ToolMessage(
                    content="search results",
                    name="web_search",
                    tool_call_id="call_search",
                ),
                AIMessage(content="OpenAI docs"),
            ]
        }

    _register_single_node(
        graph_registry,
        "search",
        answer,
        server_tools={"web_search"},
    )
    response, events = await _create(
        openai_client,
        model="search",
        input="Find the Responses docs.",
        tools=SEARCH_TOOLS,
        stream=True,
    )

    assert [item.type for item in response.output] == ["web_search_call", "message"]
    search = response.output[0]
    assert search.id == "ws_call_search"
    assert response.output_text == "OpenAI docs"
    assert [
        event.type
        for event in events
        if event.type.startswith("response.web_search_call.")
    ] == ["response.web_search_call.completed"]
    assert [
        event.item.type for event in events if event.type == "response.output_item.done"
    ] == ["web_search_call", "message"]

    await openai_client.responses.create(
        model="search",
        input=[
            {"role": "user", "content": "Find the Responses docs."},
            *response.output,
            {"role": "user", "content": "Summarize that."},
        ],
        tools=SEARCH_TOOLS,
    )
    replayed_search = received[-1][1]
    assert isinstance(replayed_search, AIMessage)
    assert replayed_search.content[0]["id"] == WEB_SEARCH["id"]
    assert (
        replayed_search.content[0]["action"]["query"] == (WEB_SEARCH["action"]["query"])
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_provider_native_search_blocks_are_not_public_server_activity(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    stream: bool,
) -> None:
    async def answer(_state: MessagesState):
        return {
            "messages": [
                AIMessage(
                    content=[
                        {
                            "type": "server_tool_call",
                            "name": "web_search",
                            "id": "ws_provider_search",
                            "args": {
                                "type": "search",
                                "query": "OpenAI Responses API",
                            },
                        },
                        {
                            "type": "server_tool_result",
                            "tool_call_id": "ws_provider_search",
                            "status": "success",
                        },
                        {"type": "text", "text": "OpenAI docs"},
                    ]
                )
            ]
        }

    _register_single_node(
        graph_registry,
        "provider-search",
        answer,
        server_tools={"web_search"},
    )
    response, events = await _create(
        openai_client,
        model="provider-search",
        input="Find the Responses docs.",
        tools=SEARCH_TOOLS,
        stream=stream,
    )

    assert [item.type for item in response.output] == ["message"]
    assert response.output_text == "OpenAI docs"
    if stream:
        assert not [
            event.type
            for event in events
            if event.type.startswith("response.web_search_call.")
        ]


@pytest.mark.parametrize(
    ("tool_status", "search_status"),
    [("success", "completed"), ("error", "failed")],
)
async def test_server_search_reports_its_terminal_status(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    tool_status: str,
    search_status: str,
) -> None:
    async def answer(_state: MessagesState):
        return {
            "messages": [
                AIMessage(content="", tool_calls=[SEARCH_CALL]),
                ToolMessage(
                    content="search results",
                    name="web_search",
                    tool_call_id="call_search",
                    status=tool_status,
                ),
                AIMessage(content="Docs"),
            ]
        }

    _register_single_node(
        graph_registry,
        "search",
        answer,
        server_tools={"web_search"},
    )
    response, events = await _create(
        openai_client,
        model="search",
        input="Find docs.",
        tools=SEARCH_TOOLS,
        stream=True,
    )

    assert [
        event.delta for event in events if event.type == "response.output_text.delta"
    ] == ["Docs"]
    assert response.output[0].status == search_status
    assert response.output[0].action.query == "OpenAI Responses API"
    assert response.output_text == "Docs"
    search_events = [
        event.type
        for event in events
        if event.type.startswith("response.web_search_call.")
    ]
    assert search_events == (
        ["response.web_search_call.completed"] if search_status == "completed" else []
    )


@pytest.mark.parametrize("fail", [False, True])
async def test_server_answer_streams_before_graph_finishes_and_retains_partial_output(
    graph_registry: GraphRegistry, fail: bool
) -> None:
    finish = Event()

    async def search(_state: MessagesState):
        get_stream_writer()(status_event("Searching"))
        return {
            "messages": [
                AIMessage(content="Private preamble", tool_calls=[SEARCH_CALL]),
                ToolMessage(content="Results", tool_call_id="call_search"),
            ]
        }

    async def answer(state: MessagesState):
        message = await FakeListChatModel(responses=["Docs"]).ainvoke(state["messages"])
        await finish.wait()
        if fail:
            message = "Answer finalization failed"
            raise RuntimeError(message)
        return {"messages": [message]}

    graph = (
        StateGraph(MessagesState)
        .add_node("search", search)
        .add_node("answer", answer)
        .set_entry_point("search")
        .add_edge("search", "answer")
        .set_finish_point("answer")
        .compile()
    )
    graph_registry.register(
        "live-search",
        GraphConfig(
            graph=graph,
            description="Live search",
            server_tools={"web_search"},
            features={GraphFeature.CLIENT_EVENTS},
        ),
    )
    request = ResponseCreateRequest(
        model="live-search", input="Find docs", tools=SEARCH_TOOLS
    )
    decoded, messages, _ = decode_responses_request(request, {"web_search"})
    run = await prepare_run(decoded, messages, graph_registry)
    events = []
    with fail_after(5):
        async for frame in stream_response(request, run):
            event = json.loads(frame.split("data: ", 1)[1])
            events.append(event)
            if event["type"] == "response.output_text.delta" and event["delta"] == "D":
                # The graph cannot finish until the first answer token reaches us.
                finish.set()

    assert finish.is_set()
    terminal = events[-1]
    assert terminal["type"] == ("response.failed" if fail else "response.completed")
    response = terminal["response"]
    assert [item["type"] for item in response["output"]] == [
        "message",
        "web_search_call",
        "message",
    ]
    assert response["output"][0]["phase"] == "commentary"
    assert response["output"][-1]["content"][0]["text"] == "Docs"
    assert response["output"][-1]["status"] == ("incomplete" if fail else "completed")


@pytest.mark.parametrize("stream", [False, True])
async def test_unfinished_server_execution_fails(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    stream: bool,
) -> None:
    async def unfinished(_state: MessagesState):
        return {"messages": [AIMessage(content=[CALL_ITEM], tool_calls=[CALL])]}

    _register_single_node(
        graph_registry,
        "package",
        unfinished,
        server_tools={"package_version"},
    )

    if not stream:
        with pytest.raises(InternalServerError):
            await openai_client.responses.create(
                model="package",
                input="Version?",
                tools=PACKAGE_TOOLS,
            )
        return

    failed, events = await _create(
        openai_client,
        model="package",
        input="Version?",
        tools=PACKAGE_TOOLS,
        stream=True,
    )
    assert [event.type for event in events][-2:] == ["error", "response.failed"]
    assert failed.status == "failed"
    assert [item.type for item in failed.output] == ["custom_tool_call"]
    assert failed.output[0].call_id == "call_package"


async def test_repeated_server_tool_call_id_fails(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    async def repeated(_state: MessagesState):
        result = ToolMessage(
            content="langgraph==installed-version", tool_call_id="call_package"
        )
        return {
            "messages": [
                AIMessage(content=[CALL_ITEM], tool_calls=[CALL]),
                result,
                AIMessage(content=[CALL_ITEM], tool_calls=[CALL]),
                result,
                AIMessage(content="Done."),
            ]
        }

    _register_single_node(
        graph_registry,
        "package",
        repeated,
        server_tools={"package_version"},
    )

    with pytest.raises(InternalServerError):
        await openai_client.responses.create(
            model="package",
            input="Version?",
            tools=PACKAGE_TOOLS,
        )
