from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph_openai_serve import GraphRequest


class MockToolCallingChatModel(FakeMessagesListChatModel):
    """Deterministic model with the tool-binding surface used by demo graphs."""

    def bind_tools(
        self, tools: list[BaseTool], **kwargs: Any
    ) -> "MockToolCallingChatModel":  # ty: ignore[invalid-method-override]
        return self


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
