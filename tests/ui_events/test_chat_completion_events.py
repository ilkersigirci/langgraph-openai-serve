import json
from typing import TypedDict

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from langgraph_openai_serve import (
    GraphCapabilities,
    GraphConfig,
    GraphRegistry,
    LangchainOpenaiApiServe,
)


def make_client(
    message_graph,
    *,
    capabilities: GraphCapabilities | None = None,
    raise_server_exceptions: bool = True,
):
    registry = GraphRegistry(
        registry={
            "test": GraphConfig(
                graph=message_graph,
                streamable_node_names=["generate"],
                capabilities=capabilities or GraphCapabilities(),
            )
        }
    )
    app = LangchainOpenaiApiServe(graphs=registry).bind_openai_chat_completion().app
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def request_payload(*, stream: bool = False) -> dict:
    return {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": stream,
        "x_langgraph_openai_serve": {
            "ui_events": {"enabled": True, "thread_id": "chat-1"}
        },
    }


def event_lines(content: str) -> list[dict]:
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def test_event_mode_non_streaming_response(message_graph) -> None:
    with make_client(message_graph) as client:
        response = client.post("/v1/chat/completions", json=request_payload())

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    events = event_lines(message["content"])

    assert response.json()["choices"][0]["finish_reason"] == "stop"
    assert events[0]["type"] == "RUN_STARTED"
    assert events[0]["threadId"] == "chat-1"
    assert events[-1]["type"] == "RUN_FINISHED"
    assert "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "TEXT_MESSAGE_CONTENT"
    ) == "hello"


def test_event_mode_streaming_response_has_complete_event_lines(message_graph) -> None:
    with make_client(message_graph) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json=request_payload(stream=True),
        ) as response:
            chunks = list(response.iter_lines())

    data_lines = [
        line.removeprefix("data: ")
        for line in chunks
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content_parts = [
        json.loads(line)["choices"][0]["delta"].get("content")
        for line in data_lines
    ]
    event_parts = [part for part in content_parts if part]

    assert all(part.endswith("\n") for part in event_parts)
    events = [event for part in event_parts for event in event_lines(part)]
    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1]["type"] == "RUN_FINISHED"
    assert "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "TEXT_MESSAGE_CONTENT"
    ) == "hello"


def test_model_listing_advertises_ui_event_capabilities(message_graph) -> None:
    capabilities = GraphCapabilities(hitl=True, citations=True, state=True)

    with make_client(message_graph, capabilities=capabilities) as client:
        response = client.get("/v1/models")

    model = response.json()["data"][0]

    assert model["x_langgraph_openai_serve"]["ui_event_protocol"]["version"] == "v1"
    assert model["x_langgraph_openai_serve"]["capabilities"] == {
        "ui_events": True,
        "hitl": True,
        "citations": True,
        "state": True,
    }


def test_hitl_event_mode_requires_thread_id(message_graph) -> None:
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "x_langgraph_openai_serve": {"ui_events": True},
    }

    with make_client(
        message_graph,
        capabilities=GraphCapabilities(hitl=True),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "thread_id" in response.text


def test_hitl_interrupt_maps_to_ui_event_tool_call_and_resumes() -> None:
    class ApprovalState(TypedDict, total=False):
        answer: str

    async def approve(state: ApprovalState):
        result = interrupt({"prompt": "Approve?"})
        return {"answer": "approved" if result["approved"] else "rejected"}

    graph = (
        StateGraph(ApprovalState)
        .add_node("approve", approve)
        .set_entry_point("approve")
        .set_finish_point("approve")
        .compile(checkpointer=InMemorySaver())
    )
    registry = GraphRegistry(
        registry={
            "approval": GraphConfig(
                graph=graph,
                capabilities=GraphCapabilities(hitl=True),
                output_to_text=lambda output: output["answer"],
            )
        }
    )
    app = LangchainOpenaiApiServe(graphs=registry).bind_openai_chat_completion().app
    payload = {
        "model": "approval",
        "messages": [{"role": "user", "content": "Hi"}],
        "x_langgraph_openai_serve": {
            "ui_events": {"enabled": True, "thread_id": "approval-thread"}
        },
    }

    with TestClient(app) as client:
        interrupted = client.post("/v1/chat/completions", json=payload).json()
        tool_call = interrupted["choices"][0]["message"]["tool_calls"][0]
        resumed = client.post(
            "/v1/chat/completions",
            json={
                **payload,
                "messages": [
                    {"role": "user", "content": "Hi"},
                    interrupted["choices"][0]["message"],
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps({"approved": True}),
                    },
                ],
            },
        ).json()

    arguments = json.loads(tool_call["function"]["arguments"])
    assert interrupted["choices"][0]["finish_reason"] == "tool_calls"
    assert tool_call["function"]["name"] == "ui_event"
    assert arguments["type"] == "CUSTOM"
    assert arguments["name"] == "hitl.request"
    assert arguments["value"]["responseSchema"]["type"] == "object"
    assert resumed["choices"][0]["finish_reason"] == "stop"
    assert "approved" in resumed["choices"][0]["message"]["content"]
