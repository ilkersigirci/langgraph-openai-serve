from typing import Any

import pytest
from fastapi import FastAPI
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from openai import AsyncOpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
    client_event,
    status_event,
)
from tests.graph.support.schemas import MessageState

pytestmark = pytest.mark.anyio

STREAM_EVENTS_METADATA = {"langgraph_stream_events": "v1"}
PROGRESS_DATA = {"completed": 2, "total": 5}


def client_event_graph() -> Any:
    model = FakeListChatModel(responses=["draft", "answer"])

    async def generate(state: MessageState):
        writer = get_stream_writer()

        # Generic and malformed custom data must never become public.
        writer({"type": "progress", "data": {"secret": "private"}})
        writer(
            {
                "type": "langgraph_openai_serve.client_event",
                "event": {"type": "status", "data": {"secret": "private"}},
            }
        )

        writer(status_event("Preparing answer"))
        draft = await model.ainvoke(state["messages"])
        writer(client_event("progress", PROGRESS_DATA, namespace=("research",)))
        answer = await model.ainvoke([*state["messages"], draft])
        writer(status_event("Answer ready", done=True))
        return {"messages": [answer]}

    return (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )


@pytest.fixture
def fastapi_app() -> FastAPI:
    registry = GraphRegistry(
        registry={
            "client-events": GraphConfig(
                graph=client_event_graph,
                streamable_node_names=["generate"],
            )
        }
    )
    return LanggraphOpenaiServe(graphs=registry).bind_openai_api().app


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(None, id="not-requested"),
        pytest.param({"langgraph_stream_events": "v2"}, id="unsupported-version"),
    ],
)
async def test_client_events_require_the_v1_opt_in(
    openai_client: AsyncOpenAI,
    metadata: dict[str, str] | None,
) -> None:
    stream = await openai_client.chat.completions.create(
        model="client-events",
        messages=[{"role": "user", "content": "Research this"}],
        stream=True,
        metadata=metadata,
    )
    chunks = [chunk async for chunk in stream]

    assert all(
        "langgraph_openai_serve" not in (chunk.model_extra or {}) for chunk in chunks
    )
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == (
        "draftanswer"
    )


async def test_v1_stream_exposes_only_public_events_in_graph_order(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await openai_client.chat.completions.create(
        model="client-events",
        messages=[{"role": "user", "content": "Research this"}],
        stream=True,
        metadata=STREAM_EVENTS_METADATA,
    )
    chunks = [chunk async for chunk in stream]

    extensions: list[dict[str, Any]] = []
    event_chunks = []
    timeline: list[tuple[str, str]] = []
    for chunk in chunks:
        extension = (chunk.model_extra or {}).get("langgraph_openai_serve")
        if isinstance(extension, dict):
            extensions.append(extension)
            event_chunks.append(chunk)
            timeline.append(("event", extension["event"]["type"]))
        elif content := chunk.choices[0].delta.content:
            if timeline and timeline[-1][0] == "text":
                timeline[-1] = ("text", timeline[-1][1] + content)
            else:
                timeline.append(("text", content))

    assert timeline == [
        ("event", "status"),
        ("text", "draft"),
        ("event", "progress"),
        ("text", "answer"),
        ("event", "status"),
    ]
    assert extensions == [
        {
            "schema_version": 1,
            "event": {
                "type": "status",
                "namespace": [],
                "data": {
                    "description": "Preparing answer",
                    "done": False,
                    "hidden": False,
                },
            },
        },
        {
            "schema_version": 1,
            "event": {
                "type": "progress",
                "namespace": ["research"],
                "data": PROGRESS_DATA,
            },
        },
        {
            "schema_version": 1,
            "event": {
                "type": "status",
                "namespace": [],
                "data": {
                    "description": "Answer ready",
                    "done": True,
                    "hidden": False,
                },
            },
        },
    ]
    assert all(chunk.choices[0].delta.tool_calls is None for chunk in chunks)

    assert len({chunk.id for chunk in chunks}) == 1
    assert len({chunk.created for chunk in chunks}) == 1
    assert {chunk.model for chunk in chunks} == {"client-events"}
    for chunk in event_chunks:
        assert chunk.object == "chat.completion.chunk"
        assert len(chunk.choices) == 1
        assert chunk.choices[0].index == 0
        assert chunk.choices[0].delta.model_dump(exclude_none=True) == {}
        assert chunk.choices[0].finish_reason is None
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.parametrize(
    ("event_type", "data", "namespace"),
    [
        pytest.param("debug", {}, (), id="unsupported-type"),
        pytest.param("status", {"value": object()}, (), id="non-json-data"),
        pytest.param("status", {}, (1,), id="non-string-namespace"),
    ],
)
def test_client_event_rejects_invalid_public_values(
    event_type: Any,
    data: Any,
    namespace: tuple[Any, ...],
) -> None:
    with pytest.raises(ValueError):
        client_event(event_type, data, namespace=namespace)


def test_status_event_builds_the_portable_status_shape() -> None:
    assert status_event(
        "Generating audio",
        done=True,
        hidden=True,
        namespace=("media",),
    ) == client_event(
        "status",
        {
            "description": "Generating audio",
            "done": True,
            "hidden": True,
        },
        namespace=("media",),
    )

    with pytest.raises(ValueError):
        status_event("")
