"""Replace response_store_pair to run the contracts against another adapter."""

import pytest

from langgraph_openai_serve import InMemoryResponseStore, ResponseStore


@pytest.fixture
def response_store_pair() -> tuple[ResponseStore, ResponseStore]:
    """Two clients sharing a fresh, empty persistence namespace."""
    store = InMemoryResponseStore()
    return store, store


@pytest.fixture
def response_store(
    response_store_pair: tuple[ResponseStore, ResponseStore],
) -> ResponseStore:
    return response_store_pair[0]
