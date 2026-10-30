from collections.abc import AsyncIterator, Callable

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph_openai_serve import GraphRequest

from lgos_demo_api.graphs.mcp_mock import MockToolCallingChatModel


@pytest.fixture
def anyio_backend() -> str:
    """Run the API test suite on its supported async backend."""
    return "asyncio"


@pytest.fixture
async def sqlite_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield checkpointer


@pytest.fixture
def make_graph_input() -> Callable[..., tuple[GraphRequest, list[BaseMessage]]]:
    """Build protocol-neutral inputs used by demo graph runner tests."""

    def _make_graph_input(
        model: str,
        *,
        content: str = "question",
        user: str | None = None,
        metadata: dict[str, str] | None = None,
        messages: list[BaseMessage] | None = None,
    ) -> tuple[GraphRequest, list[BaseMessage]]:
        request = GraphRequest(
            model=model,
            metadata=metadata or {},
            user=user,
            tools=(),
            tool_choice=None,
            parallel_tool_calls=None,
        )
        return request, (
            messages if messages is not None else [HumanMessage(content=content)]
        )

    return _make_graph_input


@pytest.fixture
def make_tool_calling_model() -> Callable[..., MockToolCallingChatModel]:
    """Build a deterministic sequence model for tool-calling graph tests."""

    def _make(*responses: AIMessage) -> MockToolCallingChatModel:
        return MockToolCallingChatModel(responses=list(responses))

    return _make
