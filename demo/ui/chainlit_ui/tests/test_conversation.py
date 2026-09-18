from unittest.mock import AsyncMock, Mock

import pytest


async def test_limited_functionality_warning_uses_transient_toast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lgos_chainlit import conversation as chat

    send_toast = AsyncMock()
    monkeypatch.setattr(
        chat.cl,
        "context",
        Mock(emitter=Mock(send_toast=send_toast)),
    )

    await chat.send_limited_functionality_warning()

    send_toast.assert_awaited_once_with(
        chat.LIMITED_FUNCTIONALITY_MESSAGE,
        type="warning",
    )
