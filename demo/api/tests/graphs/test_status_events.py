from typing import Any, cast

import pytest
from langgraph.types import CustomStreamPart
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph_stream

from lgos_demo_api.graphs import status_events


def _public_event(value: object) -> dict[str, Any]:
    part = cast(CustomStreamPart, value)
    envelope = cast(dict[str, Any], part["data"])
    return cast(dict[str, Any], envelope["event"])


@pytest.mark.anyio
async def test_graph_streams_portable_status_updates(
    make_request,
    monkeypatch,
) -> None:
    monkeypatch.setattr(status_events, "STATUS_EVENT_DELAY_SECONDS", 0)
    request = make_request(
        "status-events",
        content="Prepare the media workflow.",
    )
    registry = GraphRegistry(
        registry={"status-events": status_events.status_event_graph_config}
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

    assert events == [
        {
            "type": "status",
            "namespace": ["media"],
            "data": {
                "description": "Generating audio",
                "done": False,
                "hidden": False,
            },
        },
        {
            "type": "status",
            "namespace": ["media"],
            "data": {
                "description": "Calculating embeddings",
                "done": False,
                "hidden": False,
            },
        },
        {
            "type": "status",
            "namespace": ["media"],
            "data": {
                "description": "Media ready",
                "done": True,
                "hidden": False,
            },
        },
    ]
    assert "".join(item for item in stream if isinstance(item, str)) == (
        status_events.ANSWER
    )
