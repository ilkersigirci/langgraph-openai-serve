"""The registry and app helpers shared by server tests and the ``lgos`` command test."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from openai import AsyncOpenAI

from langgraph_openai_serve import GraphConfig, GraphFeature, GraphRegistry
from langgraph_openai_serve.server import ServerResources, ServerSettings
from tests.graph.support.interrupt import make_interrupt_graph
from tests.graph.support.message import make_message_graph


def create_registry(resources: ServerResources) -> GraphRegistry:
    return GraphRegistry(
        graphs={
            "chat": GraphConfig(
                graph=make_message_graph(),
                description="Answer with a fixed message.",
                features=frozenset({GraphFeature.BACKGROUND}),
            ),
            "approval": GraphConfig(
                graph=make_interrupt_graph(checkpointer=resources.checkpointer),
                description="Pause for an answer.",
                features=frozenset({GraphFeature.INTERRUPTS}),
            ),
        },
        run_coordinator=resources.run_coordinator,
    )


def server_settings(**overrides: object) -> ServerSettings:
    # Explicit values win over any LGOS_* variables in the test environment.
    return ServerSettings(
        **{"POSTGRES_URI": None, "BACKGROUND": "none", "CORS_ORIGINS": [], **overrides}
    )


@asynccontextmanager
async def started(app: FastAPI) -> AsyncGenerator[AsyncOpenAI, None]:
    # Run the app lifespan and call it through ASGI, without a listening socket.
    async with (
        app.router.lifespan_context(app),
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            max_retries=0,
            http_client=AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ),
        ) as client,
    ):
        yield client
