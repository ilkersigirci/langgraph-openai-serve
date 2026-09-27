from langgraph_openai_serve import GraphRegistry, status_event
from langgraph_openai_serve.graph.runner import run_langgraph_stream

from lgos_demo_api.graphs import status_events


async def test_graph_streams_portable_status_updates(
    make_graph_input,
    monkeypatch,
) -> None:
    monkeypatch.setattr(status_events, "STATUS_EVENT_DELAY_SECONDS", 0)
    graph_request, messages = make_graph_input(
        "status-events",
        content="Prepare the media workflow.",
    )
    registry = GraphRegistry(
        graphs={"status-events": status_events.status_event_graph_config}
    )

    stream = [
        item async for item in run_langgraph_stream(graph_request, messages, registry)
    ]

    assert [item["data"] for item in stream if isinstance(item, dict)] == [
        status_event("Generating audio"),
        status_event("Calculating embeddings"),
        status_event("Media ready"),
    ]
    assert "".join(item for item in stream if isinstance(item, str)) == (
        status_events.ANSWER
    )
