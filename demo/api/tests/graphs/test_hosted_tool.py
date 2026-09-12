from datetime import datetime, timezone

from lgos_demo_api.graphs.hosted_tool import lgos_current_time


async def test_current_time_uses_the_lgos_clock() -> None:
    before = datetime.now(timezone.utc).replace(microsecond=0)
    result = (await lgos_current_time.ainvoke("Asia/Tokyo"))[0]["output"]
    after = datetime.now(timezone.utc)
    actual = datetime.fromisoformat(result.removeprefix("Asia/Tokyo: "))
    assert before <= actual <= after
    assert actual.utcoffset().total_seconds() == 9 * 3600


async def test_unknown_timezone_is_actionable() -> None:
    result = (await lgos_current_time.ainvoke("not/a/timezone"))[0]["output"]
    assert (
        result
        == "Unknown timezone: not/a/timezone. Use an IANA name such as Europe/Istanbul."
    )
