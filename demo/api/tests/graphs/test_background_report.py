from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from lgos_demo_api.graphs.background_report import (
    BackgroundReportSettings,
    create_background_report_graph,
)


async def test_report_is_published_from_the_checkpointed_draft_without_a_model() -> (
    None
):
    graph = create_background_report_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "report"}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Quarterly risks")]},
        config,
        context=BackgroundReportSettings(finalize_delay_seconds=0),
    )

    report = result["messages"][-1]
    assert "Background report for: Quarterly risks" in report.text
    assert report == result["draft"]
