from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest


class MockToolCallingChatModel(FakeMessagesListChatModel):
    """Deterministic model with the tool-binding surface used by create_agent."""

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
def make_request() -> Callable[..., ResponseCreateRequest]:
    """Build Responses requests used by demo graph tests."""

    def _make_request(
        model: str,
        *,
        content: str = "question",
        user: str | None = None,
        metadata: dict[str, str] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> ResponseCreateRequest:
        return ResponseCreateRequest(
            model=model,
            input=(
                messages
                if messages is not None
                else [{"role": "user", "content": content}]
            ),
            user=user,
            metadata=metadata,
        )

    return _make_request


@pytest.fixture
def make_tool_calling_model() -> Callable[..., MockToolCallingChatModel]:
    """Build a deterministic sequence model for agent tests."""

    def _make(*responses: AIMessage) -> MockToolCallingChatModel:
        return MockToolCallingChatModel(responses=list(responses))

    return _make
