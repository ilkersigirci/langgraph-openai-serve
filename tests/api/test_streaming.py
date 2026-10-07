from collections.abc import AsyncGenerator
from typing import Any, cast

from anyio import Event, fail_after
from langchain_core.callbacks import UsageMetadataCallbackHandler

from langgraph_openai_serve.api.streaming import StreamOwner
from langgraph_openai_serve.graph.run import GraphRun

KEEPALIVE = ": ping\n\n"


def unprepared_run() -> GraphRun:
    return GraphRun(
        request=cast("Any", None),
        config=cast("Any", None),
        graph=cast("Any", None),
        inputs=None,
        context=None,
        runnable_config={},
        usage_callback=UsageMetadataCallbackHandler(),
    )


async def test_idle_stream_sends_keepalive_comments_through_proxies() -> None:
    resumed = Event()

    async def source() -> AsyncGenerator[str, None]:
        yield "data: first\n\n"
        await resumed.wait()
        yield "data: last\n\n"

    owner = StreamOwner(keepalive_interval=0.01)
    response = owner.start(source(), unprepared_run())
    chunks: list[str] = []
    try:
        with fail_after(1):
            async for chunk in response.body_iterator:
                chunks.append(cast("str", chunk))
                if chunk == KEEPALIVE:
                    resumed.set()
    finally:
        await owner.aclose()

    assert chunks[0] == "data: first\n\n"
    assert KEEPALIVE in chunks
    assert chunks[-1] == "data: last\n\n"
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    # Nginx would otherwise buffer the stream and hold back the keepalives.
    assert response.headers["x-accel-buffering"] == "no"
