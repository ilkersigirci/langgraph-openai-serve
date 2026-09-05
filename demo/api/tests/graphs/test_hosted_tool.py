from datetime import datetime, timezone

import pytest
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest

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
@pytest.mark.parametrize(
    "tool_builder",
    [
        lambda name: (
            {"type": "custom", "name": name}
            if name.startswith("lgos_")
            else {"type": "function", "name": name}
        ),
        lambda name: (
            {"type": name}
            if name.startswith("lgos_")
            else {"type": "function", "name": name}
        ),
    ],
)
def test_only_the_requested_server_tool_is_enabled(
    names: list[str],
    choice: str | None,
    enabled: bool,
    tool_builder: object,
) -> None:
    request = ResponseCreateRequest(
        model="hosted-tool",
        input="Time?",
        tools=[tool_builder(name) for name in names],  # type: ignore[operator]
        tool_choice=choice,
    )
    graph_request, messages, _ = decode_responses_request(request)
    assert request_to_input(graph_request, messages).time_requested is enabled
