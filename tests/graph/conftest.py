from collections.abc import Callable
from typing import Any

import pytest

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest, Role
from tests.graph.support.message import make_message_graph as build_message_graph


@pytest.fixture
def make_request() -> Callable[..., ChatCompletionRequest]:
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
            messages=messages or [{"role": Role.USER, "content": content}],
            user=user,
            metadata=metadata,
        )

    return _make_request


@pytest.fixture
def make_message_graph() -> Callable[..., Any]:
    return build_message_graph
