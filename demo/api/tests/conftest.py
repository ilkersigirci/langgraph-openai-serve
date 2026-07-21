from collections.abc import Callable
from typing import Any

import pytest
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest, Role


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Run the API test suite on its supported async backend."""
    return "asyncio"


@pytest.fixture
def make_request() -> Callable[..., ChatCompletionRequest]:
    """Build OpenAI chat requests used by graph tests."""

    def _make_request(
        model: str,
        *,
        content: str = "question",
        user: str | None = None,
        metadata: dict[str, str] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=model,
            messages=(
                messages
                if messages is not None
                else [{"role": Role.USER, "content": content}]
            ),
            user=user,
            metadata=metadata,
        )

    return _make_request
