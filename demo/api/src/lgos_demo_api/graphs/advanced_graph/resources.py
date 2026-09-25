"""Own the external clients the advanced graph needs in one process."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from openai import AsyncOpenAI

from lgos_demo_api.core.settings import settings
from lgos_demo_api.graphs.advanced_graph.graph import (
    create_advanced_graph,
    create_model,
)
from lgos_demo_api.graphs.advanced_graph.knowledge import (
    OpenAICompatibleKnowledgeBase,
)
from lgos_demo_api.graphs.advanced_graph.state import AdvancedGraph


def _vector_store_connection() -> tuple[str, str]:
    """Resolve storage credentials without leaking the model provider's key."""
    if settings.VECTOR_STORE_BASE_URL:
        return (
            settings.VECTOR_STORE_BASE_URL,
            settings.VECTOR_STORE_API_KEY or "DUMMY",
        )
    return (
        settings.OPENAI_BASE_URL,
        settings.VECTOR_STORE_API_KEY or settings.OPENAI_API_KEY,
    )


@asynccontextmanager
async def open_advanced_graph(
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
) -> AsyncIterator[AdvancedGraph]:
    """
    Build the advanced graph and close its clients on exit.

    The API and the background worker both run this graph, so both open it here.

    Yields:
        The compiled advanced graph.

    """
    vector_store_base_url, vector_store_api_key = _vector_store_connection()
    async with (
        httpx2.AsyncClient(timeout=60) as upstream_http,
        AsyncOpenAI(
            base_url=vector_store_base_url,
            api_key=vector_store_api_key,
            default_headers=(
                {"x-bf-api-key": settings.VECTOR_STORE_BIFROST_KEY_NAME}
                if settings.VECTOR_STORE_BIFROST_KEY_NAME
                else None
            ),
            http_client=upstream_http,
            max_retries=0,
        ) as vector_store_client,
        AsyncOpenAI(
            base_url=settings.FILES_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
        ) as files_client,
    ):
        knowledge = (
            OpenAICompatibleKnowledgeBase(vector_store_client, settings.VECTOR_STORE_ID)
            if settings.VECTOR_STORE_ID
            else None
        )
        yield create_advanced_graph(
            model=create_model(upstream_http),
            knowledge=knowledge,
            files=files_client,
            checkpointer=checkpointer,
            store=store,
        )


__all__ = ["open_advanced_graph"]
