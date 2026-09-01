from unittest.mock import AsyncMock

import pytest

from lgos_openwebui.functions.generic import Pipe
from lgos_openwebui.functions.generic import pipe as generic_pipe

from .openwebui_support import model


@pytest.fixture
def anyio_backend() -> str:
    """Run the Open WebUI test suite on its supported async backend."""
    return "asyncio"


@pytest.fixture
def configured_pipe(monkeypatch: pytest.MonkeyPatch) -> Pipe:
    monkeypatch.setattr(
        generic_pipe,
        "_retrieve_model",
        AsyncMock(return_value=model()),
    )
    return Pipe()
