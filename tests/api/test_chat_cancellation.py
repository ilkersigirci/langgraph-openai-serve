import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
)

pytestmark = pytest.mark.anyio

_TEST_TIMEOUT = 5.0
_PROVIDER_CHUNK = (
    'data: {"id":"chatcmpl-provider","object":"chat.completion.chunk",'
    '"created":0,"model":"provider","choices":[{"index":0,'
    '"delta":{"content":"token"},"finish_reason":null}]}\n\n'
)


class CancellationState(BaseModel):
    messages: list[BaseMessage]


@dataclass
class _ProviderLifecycle:
    waiting: asyncio.Event = field(default_factory=asyncio.Event)
    disconnected: asyncio.Event = field(default_factory=asyncio.Event)
    completed_normally: asyncio.Event = field(default_factory=asyncio.Event)
    finalized: asyncio.Event = field(default_factory=asyncio.Event)
    normal_completion_release: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _NodeLifecycle:
    streaming: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    completed_normally: asyncio.Event = field(default_factory=asyncio.Event)
    finalizer_started: asyncio.Event = field(default_factory=asyncio.Event)
    finalizer_release: asyncio.Event = field(default_factory=asyncio.Event)
    finalizer_finished: asyncio.Event = field(default_factory=asyncio.Event)


async def _wait_for_event(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=_TEST_TIMEOUT)


async def _wait_until_started(
    server: uvicorn.Server, server_task: asyncio.Task[None]
) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("Uvicorn stopped before accepting requests")
            await asyncio.sleep(0)


@asynccontextmanager
async def _serve_over_tcp(app: FastAPI) -> AsyncIterator[str]:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", 0))
    host, port = server_socket.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            lifespan="off",
            log_level="warning",
            access_log=False,
            timeout_graceful_shutdown=1,
        )
    )
    server_task = asyncio.create_task(
        server.serve(sockets=[server_socket]), name="test-uvicorn-server"
    )

    try:
        await _wait_until_started(server, server_task)
        yield f"http://{host}:{port}/v1"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=_TEST_TIMEOUT)
        except TimeoutError:
            server.force_exit = True
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task
            raise
        finally:
            server_socket.close()


def _build_fake_provider(lifecycle: _ProviderLifecycle) -> FastAPI:
    app = FastAPI()

    async def chunks() -> AsyncIterator[str]:
        try:
            yield _PROVIDER_CHUNK
            lifecycle.waiting.set()
            await lifecycle.normal_completion_release.wait()
        except asyncio.CancelledError:
            lifecycle.disconnected.set()
            raise
        else:
            lifecycle.completed_normally.set()
            yield "data: [DONE]\n\n"
        finally:
            lifecycle.finalized.set()

    @app.post("/v1/chat/completions")
    async def create_chat_completion() -> StreamingResponse:
        return StreamingResponse(chunks(), media_type="text/event-stream")

    return app


def _build_downstream_graph(
    provider_base_url: str, lifecycle: _NodeLifecycle
) -> CompiledStateGraph:
    async def slow_node(state: CancellationState) -> CancellationState:
        try:
            async with AsyncOpenAI(
                api_key="test",
                base_url=provider_base_url,
                max_retries=0,
                timeout=_TEST_TIMEOUT,
            ) as provider_client:
                provider_stream = await provider_client.chat.completions.create(
                    model="provider",
                    messages=[{"role": "user", "content": "start"}],
                    stream=True,
                )
                async with provider_stream:
                    async for _chunk in provider_stream:
                        lifecycle.streaming.set()
        except asyncio.CancelledError:
            lifecycle.cancelled.set()
            raise
        else:
            lifecycle.completed_normally.set()
            return state
        finally:
            lifecycle.finalizer_started.set()
            await lifecycle.finalizer_release.wait()
            lifecycle.finalizer_finished.set()

    return (
        StateGraph(CancellationState)
        .add_node("slow", slow_node)
        .set_entry_point("slow")
        .set_finish_point("slow")
        .compile()
    )


async def test_closing_openai_stream_cancels_graph_and_provider() -> None:
    provider = _ProviderLifecycle()
    node = _NodeLifecycle()

    async with _serve_over_tcp(_build_fake_provider(provider)) as provider_url:
        graph = _build_downstream_graph(provider_url, node)
        registry = GraphRegistry(registry={"cancellable": GraphConfig(graph=graph)})
        app = LanggraphOpenaiServe(graphs=registry).bind_openai_api().app
        stream: AsyncStream[ChatCompletionChunk] | None = None

        async with _serve_over_tcp(app) as base_url:
            try:
                async with AsyncOpenAI(
                    api_key="test",
                    base_url=base_url,
                    max_retries=0,
                    timeout=_TEST_TIMEOUT,
                ) as client:
                    stream = await client.chat.completions.create(
                        model="cancellable",
                        messages=[{"role": "user", "content": "start"}],
                        stream=True,
                    )
                    await _wait_for_event(node.streaming)
                    await _wait_for_event(provider.waiting)

                    await stream.close()

                    await _wait_for_event(node.cancelled)
                    await _wait_for_event(provider.disconnected)
                    await _wait_for_event(provider.finalized)
                    await _wait_for_event(node.finalizer_started)
                    assert not node.completed_normally.is_set()
                    assert not provider.completed_normally.is_set()

                    node.finalizer_release.set()
                    await _wait_for_event(node.finalizer_finished)
                    assert not node.completed_normally.is_set()
            finally:
                provider.normal_completion_release.set()
                node.finalizer_release.set()
                if stream is not None:
                    await stream.close()
