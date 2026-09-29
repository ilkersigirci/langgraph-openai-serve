import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph_stream

from lgos_demo_api.graphs import streaming_long_mock


@pytest.mark.parametrize(
    ("history", "first_line"),
    [
        ([], "1. This is sentence 1 of 100."),
        (
            [
                HumanMessage("Count to 100."),
                AIMessage("1. This is sentence 1 of 100.\n2. This is sen"),
            ],
            'Previous answer ended with: "2. This is sen"',
        ),
    ],
    ids=["first-turn", "after-stop"],
)
async def test_long_answer_streams_after_quoting_the_returned_history(
    make_graph_input,
    monkeypatch: pytest.MonkeyPatch,
    history: list[BaseMessage],
    first_line: str,
) -> None:
    monkeypatch.setattr(streaming_long_mock, "CHARACTER_DELAY_SECONDS", 0)
    graph_request, messages = make_graph_input(
        "streaming-long-mock",
        messages=[*history, HumanMessage("Count to 100.")],
    )
    registry = GraphRegistry(
        graphs={
            "streaming-long-mock": streaming_long_mock.streaming_long_mock_graph_config
        }
    )

    events = [
        event async for event in run_langgraph_stream(graph_request, messages, registry)
    ]

    deltas = [event for event in events if isinstance(event, str)]
    answer = events[-1]
    assert isinstance(answer, AIMessage)
    assert len(deltas) > 1
    assert "".join(deltas) == answer.text
    lines = answer.text.splitlines()
    assert lines[0] == first_line
    assert lines[-1] == "100. This is sentence 100 of 100."
