from itertools import pairwise
from typing import Any, cast

import pytest
from langgraph.types import CustomStreamPart
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph_stream

from lgos_demo_api.graphs import custom_events


def _public_event(value: object) -> dict[str, Any]:
    part = cast(CustomStreamPart, value)
    envelope = cast(dict[str, Any], part["data"])
    return cast(dict[str, Any], envelope["event"])


@pytest.mark.anyio
async def test_showcase_streams_a_small_event_timeline(
    make_request, monkeypatch
) -> None:
    monkeypatch.setattr(custom_events, "SHOWCASE_EVENT_DELAY_SECONDS", 0)
    request = make_request(
        "custom-event-showcase",
        content="Build the compatibility report.",
    )
    registry = GraphRegistry(
        registry={
            "custom-event-showcase": custom_events.custom_event_showcase_graph_config
        }
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

    assert [(event["type"], event["namespace"]) for event in events] == [
        ("status", ["research"]),
        ("progress", ["research"]),
        ("progress", ["research"]),
        ("progress", ["research"]),
        ("artifact", ["research", "report"]),
    ]
    assert [event["data"].get("completed") for event in events] == [
        None,
        1,
        2,
        3,
        None,
    ]
    assert events[-1]["data"]["id"] == "compatibility-brief"
    assert "".join(item for item in stream if isinstance(item, str)) == (
        custom_events.ANSWER
    )

    event_positions = [
        index for index, item in enumerate(stream) if isinstance(item, dict)
    ]
    for start, stop in pairwise(event_positions[1:]):
        assert any(isinstance(item, str) for item in stream[start + 1 : stop])
