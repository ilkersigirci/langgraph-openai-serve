"""Serve the registry in-process for tests, with a fake model and no services."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph_openai_serve.server import ServerSettings, create_app
from openai import AsyncOpenAI

from {{ cookiecutter.project_slug }}.registry import (
    create_registry,
)

ANSWER = "Hello from the graph."


def create_test_app(*, background: Literal["none", "memory"] = "none") -> FastAPI:
    model = FakeListChatModel(responses=[ANSWER])
    return create_app(
        lambda resources: create_registry(resources, model=model),
        # Explicit values win over LGOS_* variables that just loads from .env.
        settings=ServerSettings(POSTGRES_URI=None, BACKGROUND=background),
    )


@asynccontextmanager
async def started(app: FastAPI) -> AsyncGenerator[AsyncOpenAI]:
    """Run the app lifespan and call it through ASGI, without a listening socket."""
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
