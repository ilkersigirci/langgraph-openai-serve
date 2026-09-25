import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from lgos_demo_api.graphs.background_interrupt import create_background_interrupt_graph
from lgos_demo_api.graphs.background_report import BackgroundReportSettings


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (" APPROVE ", "Background report for: Quarterly risks"),
        ("reject", "Report rejected; no report finalized."),
        ("", None),
        ("revise this", None),
        (True, None),
    ],
)
async def test_report_requires_an_offered_review_decision(
    sqlite_checkpointer: AsyncSqliteSaver,
    answer: str | bool,
    expected: str | None,
) -> None:
    graph = create_background_interrupt_graph(sqlite_checkpointer)
    config = {"configurable": {"thread_id": "report-review"}}
    settings = BackgroundReportSettings(delay_seconds=0)
    paused = await graph.ainvoke(
        {"messages": [HumanMessage(content="Quarterly risks")]},
        config=config,
        context=settings,
    )
    (review,) = paused["__interrupt__"]
    assert review.value == {
        "question": "Approve this report?",
        "report": "Background report for: Quarterly risks",
        "choices": ["approve", "reject"],
        "allow_other": False,
    }
    assert not any(isinstance(message, AIMessage) for message in paused["messages"])

    continuation = Command(resume={review.id: answer})
    if expected is None:
        with pytest.raises(ValueError, match="must be approve or reject"):
            await graph.ainvoke(continuation, config=config, context=settings)
        return

    completed = await graph.ainvoke(continuation, config=config, context=settings)

    assert "__interrupt__" not in completed
    assert completed["messages"][-1].text == expected
