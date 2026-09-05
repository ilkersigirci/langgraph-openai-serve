from collections.abc import Callable

import pytest

from langgraph_openai_serve import GraphRequest


@pytest.fixture
def make_request() -> Callable[..., GraphRequest]:
    """Build protocol-neutral requests used by package graph tests."""

    def _make_request(
        model: str,
        *,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> GraphRequest:
        return GraphRequest(
            model=model,
            user=user,
            metadata=metadata or {},
            tools=(),
            tool_choice=None,
            parallel_tool_calls=None,
        )

    return _make_request
