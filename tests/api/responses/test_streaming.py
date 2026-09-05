import json
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.content import create_citation, create_text_block
from langgraph.config import get_stream_writer
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import StateGraph
from openai import AsyncOpenAI
from openai.types.responses import ResponseStreamEvent

from langgraph_openai_serve import (
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    LanggraphOpenaiServe,
    client_event,
    status_event,
)
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.graph.interrupt.state import checkpoint_key
from tests.graph.support.interrupt import make_interrupt_graph
from tests.graph.support.message import make_message_graph
from tests.graph.support.schemas import MessageState

FIXTURES = Path(__file__).with_name("fixtures")
FINAL_TEXT = "Fixture answer."
USAGE = {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11}


def _status_graph(*, multiple: bool = False, stream_final: bool = True) -> Any:
    model = FakeListChatModel(responses=[FINAL_TEXT])
    if not stream_final:
        model = model.with_config(tags=[TAG_NOSTREAM])

    async def generate(state: MessageState) -> dict[str, list[AIMessage]]:
        writer = get_stream_writer()
        writer(status_event("Checking inputs."))
        if multiple:
            writer(status_event("Not for clients", hidden=True))
            writer(client_event("status", {"description": ""}))
            writer(client_event("progress", {"completed": 1, "total": 2}))
            writer(client_event("artifact", {"id": "private-to-responses"}))
            writer({"type": "progress", "data": {"private": True}})
        answer = await model.ainvoke(state["messages"])
        if multiple:
            writer(status_event("Answer ready", done=True))
        return {"messages": [answer]}

    return (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )


def _failing_graph() -> Any:
    async def fail(_state: MessageState) -> None:
        msg = "Fixture graph failed."
        raise RuntimeError(msg)

    return (
        StateGraph(MessageState)
        .add_node("fail", fail)
        .set_entry_point("fail")
        .set_finish_point("fail")
        .compile()
    )


@pytest.fixture
def fastapi_app() -> FastAPI:
    citation_text = "🌍 Café source"
    citation_message = AIMessage(
        content_blocks=[
            create_text_block(
                text=citation_text,
                annotations=[
                    create_citation(
                        url="https://example.com/globe",
                        title="Globe",
                        start_index=0,
                        end_index=0,
                        cited_text="🌍",
                    ),
                    create_citation(
                        url="https://example.com/source",
                        title="Source",
                        start_index=len(citation_text) - len("source"),
                        end_index=len(citation_text) - 1,
                        cited_text="source",
                    ),
                ],
            )
        ]
    )
    registry = GraphRegistry(
        registry={
            "golden": GraphConfig(
                graph=lambda: _status_graph(stream_final=False),
                description="DUMMY",
                streamable_node_names=["generate"],
                features={GraphFeature.CLIENT_EVENTS},
            ),
            "commentary": GraphConfig(
                graph=lambda: _status_graph(multiple=True),
                description="DUMMY",
                streamable_node_names=["generate"],
                features={GraphFeature.CLIENT_EVENTS},
            ),
            "fallback": GraphConfig(
                graph=lambda: make_message_graph("fallback"),
                description="DUMMY",
            ),
            "mismatch": GraphConfig(
                graph=lambda: make_message_graph("streamed"),
                description="DUMMY",
                streamable_node_names=["generate"],
                output_to_message=lambda _output: AIMessage("durable"),
            ),
            "usage": GraphConfig(
                graph=lambda: make_message_graph("counted"),
                description="DUMMY",
                streamable_node_names=["generate"],
                output_to_message=lambda _output: AIMessage(
                    "counted",
                    usage_metadata=USAGE,
                ),
            ),
            "citations": GraphConfig(
                graph=lambda: make_message_graph(citation_text),
                description="DUMMY",
                output_to_message=lambda _output: citation_message,
            ),
            "failure": GraphConfig(
                graph=_failing_graph,
                description="DUMMY",
            ),
        }
    )
    return LanggraphOpenaiServe(graphs=registry).bind_openai_api().app


async def _events(
    openai_client: AsyncOpenAI,
    model: str,
) -> list[ResponseStreamEvent]:
    stream = await openai_client.responses.create(
        model=model,
        input="Hello",
        store=False,
        stream=True,
    )
    return [event async for event in stream]


def _fixture_event_types(name: str) -> list[str]:
    with FIXTURES.joinpath(name).open(encoding="utf-8") as fixture:
        payloads = cast("list[dict[str, Any]]", json.load(fixture))
    return [cast("str", payload["type"]) for payload in payloads]


async def test_text_stream_matches_golden_lifecycle_and_stable_identity(
    openai_client: AsyncOpenAI,
) -> None:
    events = await _events(openai_client, "golden")

    assert [event.type for event in events] == _fixture_event_types("text_stream.json")
    assert [event.sequence_number for event in events] == list(range(len(events)))

    response_events = [
        event
        for event in events
        if event.type
        in {"response.created", "response.in_progress", "response.completed"}
    ]
    assert len({event.response.id for event in response_events}) == 1
    completed = response_events[-1].response
    assert completed.created_at == response_events[0].response.created_at
    assert completed.completed_at == completed.created_at
    assert completed.status == "completed"
    assert [item.phase for item in completed.output] == [
        "commentary",
        "final_answer",
    ]
    assert [item.content[0].text for item in completed.output] == [
        "Checking inputs.",
        FINAL_TEXT,
    ]

    added = [event for event in events if event.type == "response.output_item.added"]
    assert [event.output_index for event in added] == [0, 1]
    assert [event.item.phase for event in added] == ["commentary", "final_answer"]
    for output_index, output_item in enumerate(completed.output):
        item_events = [
            event
            for event in events
            if getattr(event, "output_index", None) == output_index
        ]
        event_item_ids = {
            getattr(event, "item_id", None)
            or getattr(getattr(event, "item", None), "id", None)
            for event in item_events
        }
        assert event_item_ids == {output_item.id}


async def test_raw_stream_uses_named_compact_sse_without_done_sentinel(
    client: AsyncClient,
) -> None:
    async with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "fallback", "input": "Hello", "stream": True},
    ) as response:
        body = (await response.aread()).decode()

    assert response.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" not in body
    frames = [frame for frame in body.split("\n\n") if frame]
    assert frames
    for frame in frames:
        event_line, data_line = frame.splitlines()
        event_type = event_line.removeprefix("event: ")
        assert event_type
        assert data_line.startswith("data: {")
        assert json.loads(data_line.removeprefix("data: "))["type"] == event_type
        assert ": " not in data_line.removeprefix("data: ")


async def test_responses_exposes_only_visible_statuses_as_commentary(
    openai_client: AsyncOpenAI,
) -> None:
    events = await _events(openai_client, "commentary")

    completed = events[-1].response
    assert [item.phase for item in completed.output] == [
        "commentary",
        "final_answer",
        "commentary",
    ]
    assert [item.content[0].text for item in completed.output] == [
        "Checking inputs.",
        FINAL_TEXT,
        "Answer ready",
    ]
    assert [
        event.item.phase
        for event in events
        if event.type == "response.output_item.added"
    ] == ["commentary", "final_answer", "commentary"]


async def test_status_commentary_requires_feature_and_streaming(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
) -> None:
    response = await openai_client.responses.create(
        model="golden",
        input="Hello",
    )
    assert [item.phase for item in response.output] == ["final_answer"]

    fastapi_app.state.graph_registry.get_graph("golden").features.clear()
    events = await _events(openai_client, "golden")
    completed = events[-1].response
    assert [item.phase for item in completed.output] == ["final_answer"]


async def test_non_streamable_graph_emits_final_text_fallback(
    openai_client: AsyncOpenAI,
) -> None:
    events = await _events(openai_client, "fallback")

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[-1].response.output_text == "fallback"


async def test_streamed_final_text_mismatch_ends_in_failed_response(
    openai_client: AsyncOpenAI,
) -> None:
    events = await _events(openai_client, "mismatch")

    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert [event.type for event in events][-2:] == ["error", "response.failed"]
    assert not any(event.type == "response.completed" for event in events)
    assert events[-2].message == "Internal server error"
    failed = events[-1].response
    assert failed.status == "failed"
    assert failed.completed_at is None
    assert failed.error is not None
    assert failed.error.code == "server_error"


async def test_graph_failure_matches_golden_terminal_order(
    openai_client: AsyncOpenAI,
) -> None:
    events = await _events(openai_client, "failure")

    assert [event.type for event in events] == _fixture_event_types(
        "failed_stream.json"
    )
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert events[-1].response.status == "failed"


async def test_completed_stream_contains_provider_usage(
    openai_client: AsyncOpenAI,
) -> None:
    events = await _events(openai_client, "usage")

    usage = events[-1].response.usage
    assert usage is not None
    assert usage.input_tokens == USAGE["input_tokens"]
    assert usage.output_tokens == USAGE["output_tokens"]
    assert usage.total_tokens == USAGE["total_tokens"]
    assert usage.input_tokens_details.cached_tokens == 0
    assert usage.input_tokens_details.cache_write_tokens == 0
    assert usage.output_tokens_details.reasoning_tokens == 0


async def test_responses_citations_use_unicode_inclusive_boundaries(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.responses.create(
        model="citations",
        input="Hello",
    )
    events = await _events(openai_client, "citations")

    annotation_events = [
        event
        for event in events
        if event.type == "response.output_text.annotation.added"
    ]
    assert [event.annotation_index for event in annotation_events] == [0, 1]

    for output in [response.output[0], events[-1].response.output[0]]:
        part = output.content[0]
        assert part.text == "🌍 Café source"
        assert [annotation.model_dump() for annotation in part.annotations] == [
            {
                "end_index": 0,
                "start_index": 0,
                "title": "Globe",
                "type": "url_citation",
                "url": "https://example.com/globe",
            },
            {
                "end_index": len(part.text) - 1,
                "start_index": len(part.text) - len("source"),
                "title": "Source",
                "type": "url_citation",
                "url": "https://example.com/source",
            },
        ]


async def test_interrupt_stream_completes_and_releases_run_lease(
    sqlite_checkpointer: Any,
) -> None:
    run_id = str(uuid.uuid4())
    coordinator = InMemoryRunCoordinator()
    registry = GraphRegistry(
        registry={
            "interrupt": GraphConfig(
                graph=lambda: make_interrupt_graph(
                    checkpointer=sqlite_checkpointer,
                ),
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                run_coordinator=coordinator,
            )
        }
    )
    app = LanggraphOpenaiServe(graphs=registry).bind_openai_api().app

    async with (
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as http_client,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        ) as openai_client,
    ):
        stream = await openai_client.responses.create(
            model="interrupt",
            input="Hello",
            metadata={"langgraph_run_id": run_id},
            store=False,
            stream=True,
        )
        events = [event async for event in stream]

    assert events[-1].type == "response.completed"
    assert events[-1].response.output[0].type == "function_call"
    assert events[-1].response.output[0].name == "langgraph_interrupt"
    async with coordinator(checkpoint_key("interrupt", run_id)):
        pass
