from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage
from langgraph_openai_serve import GraphRequest

from lgos_demo_api.graphs.hosted_tool import get_current_time, request_to_input


async def test_current_time_uses_the_lgos_clock() -> None:
    before = datetime.now(timezone.utc).replace(microsecond=0)
    result = await get_current_time.ainvoke({"timezone": "Asia/Tokyo"})
    after = datetime.now(timezone.utc)
    actual = datetime.fromisoformat(result.removeprefix("Asia/Tokyo: "))
    assert before <= actual <= after
    assert actual.utcoffset().total_seconds() == 9 * 3600


async def test_unknown_timezone_is_actionable() -> None:
    result = await get_current_time.ainvoke({"timezone": "not/a/timezone"})
    assert (
        result
        == "Unknown timezone: not/a/timezone. Use an IANA name such as Europe/Istanbul."
    )


@pytest.mark.parametrize(
    ("names", "choice", "enabled"),
    [
        ([], None, False),
        (["display_file"], None, False),
        (["lgos_current_time"], None, True),
        (["lgos_current_time"], "none", False),
    ],
)
def test_only_the_requested_server_tool_is_enabled(
    names: list[str],
    choice: str | None,
    enabled: bool,
) -> None:
    request = GraphRequest(
        model="hosted-tool",
        metadata={},
        user=None,
        tools=(),
        tool_choice=choice,
        parallel_tool_calls=None,
        hosted_tools=tuple(name for name in names if name.startswith("lgos_")),
    )
    messages = [HumanMessage(content="Time?")]

    assert request_to_input(request, messages).time_requested is enabled
