from pathlib import Path
from typing import cast

import pytest


def test_messages_from_thread_orders_and_filters_persisted_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))

    from chainlit.types import ThreadDict  # noqa: PLC0415
    from demo.ui.chainlit_ui.history import messages_from_thread  # noqa: PLC0415

    thread = cast(
        ThreadDict,
        {
            "steps": [
                {
                    "id": "assistant",
                    "type": "assistant_message",
                    "output": "Hello!",
                    "createdAt": "2026-01-01T00:00:01.500Z",
                },
                {
                    "id": "tool",
                    "type": "tool",
                    "output": "internal result",
                    "createdAt": "2026-01-01T00:00:01.500Z",
                },
                {
                    "id": "user",
                    "type": "user_message",
                    "output": "Hello",
                    "createdAt": "2026-01-01T00:00:01Z",
                },
                {
                    "id": "error",
                    "type": "assistant_message",
                    "output": "Backend failed",
                    "createdAt": "2026-01-01T00:00:03Z",
                    "isError": True,
                },
            ]
        },
    )

    assert messages_from_thread(thread) == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello!"},
    ]
