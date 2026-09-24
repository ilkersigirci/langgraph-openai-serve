from langchain_core.messages import HumanMessage

from lgos_demo_api.graphs.background_report import (
    BackgroundReportSettings,
    background_report_graph,
)


async def test_report_quotes_the_request_without_a_model() -> None:
    result = await background_report_graph.ainvoke(
        {"messages": [HumanMessage(content="Quarterly risks")]},
        context=BackgroundReportSettings(delay_seconds=0),
    )

    assert result["messages"][-1].text == "Background report for: Quarterly risks"
