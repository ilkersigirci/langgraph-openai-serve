"""Own the external clients the advanced graph needs in one process."""

from collections.abc import AsyncGenerator
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


@asynccontextmanager
async def open_advanced_graph(
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
) -> AsyncGenerator[AdvancedGraph, None]:
    """
    Build the advanced graph and close its clients on exit.

    The API and the background worker both run this graph, so both open it here.

    Yields:
        The compiled advanced graph.

    """
    async with (
        httpx2.AsyncClient(timeout=60) as upstream_http,
        AsyncOpenAI(
            base_url=settings.vector_store_base_url,
            api_key=settings.OPENAI_GATEWAY_API_KEY,
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
