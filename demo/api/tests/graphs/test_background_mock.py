from langchain_core.messages import HumanMessage
from langgraph_openai_serve import status_event

from lgos_demo_api.graphs.background_mock import background_mock_graph
from lgos_demo_api.graphs.background_report import BackgroundReportSettings


async def test_report_quotes_the_request_without_a_model() -> None:
    result = await background_mock_graph.ainvoke(
        {"messages": [HumanMessage(content="Quarterly risks")]},
        context=BackgroundReportSettings(delay_seconds=0),
    )

    assert result["messages"][-1].text == "Background report for: Quarterly risks"


async def test_report_announces_its_work_while_streaming() -> None:
    parts = [
        part
        async for part in background_mock_graph.astream(
            {"messages": [HumanMessage(content="Quarterly risks")]},
            context=BackgroundReportSettings(delay_seconds=0),
            stream_mode="custom",
        )
    ]

    assert parts == [status_event("Writing the report")]
