from collections.abc import Callable

import pytest

from langgraph_openai_serve import GraphRequest
from tests.graph.support.request import graph_request


@pytest.fixture
def make_request() -> Callable[..., GraphRequest]:
    """Build protocol-neutral requests used by package graph tests."""
    return graph_request
