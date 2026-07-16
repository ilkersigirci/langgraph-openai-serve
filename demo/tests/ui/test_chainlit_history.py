from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest


@pytest.mark.anyio
async def test_text_only_chat_messages_ignores_stale_user_session_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    from chainlit.context import init_http_context  # noqa: PLC0415
    from demo.ui.chainlit_ui import history  # noqa: PLC0415

    init_http_context()
    history.cl.chat_context.clear()
    history.cl.user_session.set("messages", [])
    history.cl.chat_context.add(
        history.cl.Message(content="Hello", type="user_message")
    )
    history.cl.chat_context.add(history.cl.Message(content="Hello!"))

    assert history.text_only_chat_messages() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert history.cl.user_session.get("messages") == []


@pytest.mark.anyio
async def test_text_only_chat_message_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep the model-context policy explicit across Chainlit upgrades."""
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    from chainlit.context import init_http_context  # noqa: PLC0415
    from demo.ui.chainlit_ui import history  # noqa: PLC0415

    init_http_context()
    history.cl.chat_context.clear()

    included_messages = [
        history.cl.Message(content="User turn", type="user_message"),
        history.cl.Message(content="Model turn"),
        history.cl.Message(content="Task manually stopped."),
    ]
    excluded_messages = [
        history.cl.Message(content="Partial assistant output"),
        history.cl.Message(content="Chat completion failed: unavailable"),
        history.cl.AskActionMessage(content="Approve this action?", actions=[]),
        history.cl.Message(content="Approval timed out."),
    ]
    for message in excluded_messages:
        history.mark_model_context_excluded(message)

    for message in [
        *included_messages,
        *excluded_messages,
        history.cl.ErrorMessage(content="Chainlit callback failed"),
    ]:
        history.cl.chat_context.add(message)

    assert history.text_only_chat_messages() == [
        {"role": "user", "content": "User turn"},
        {"role": "assistant", "content": "Model turn"},
        {"role": "assistant", "content": "Task manually stopped."},
    ]


@pytest.mark.anyio
async def test_persisted_chainlit_errors_are_excluded_after_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    from chainlit.context import init_http_context  # noqa: PLC0415
    from chainlit.types import ThreadDict  # noqa: PLC0415
    from demo.ui.chainlit_ui import history  # noqa: PLC0415

    init_http_context()
    history.cl.chat_context.clear()
    thread = cast(
        ThreadDict,
        {
            "steps": [
                {
                    "id": "error",
                    "type": "assistant_message",
                    "name": "Error",
                    "output": "Backend failed",
                    "createdAt": "2026-01-01T00:00:01Z",
                    "isError": True,
                    "metadata": {"existing": "value"},
                },
                {
                    "id": "assistant",
                    "type": "assistant_message",
                    "name": "Assistant",
                    "output": "Valid turn",
                    "createdAt": "2026-01-01T00:00:02Z",
                },
            ]
        },
    )

    history.mark_persisted_errors_excluded(thread)
    for step in thread["steps"]:
        history.cl.chat_context.add(history.cl.Message.from_dict(step))

    assert thread["steps"][0]["metadata"] == {
        "existing": "value",
        history.MODEL_CONTEXT_EXCLUDED_KEY: True,
    }
    assert history.text_only_chat_messages() == [
        {"role": "assistant", "content": "Valid turn"}
    ]


def test_mark_model_context_excluded_preserves_message_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    from demo.ui.chainlit_ui import history  # noqa: PLC0415

    message = Mock(metadata={"existing": "value"})

    history.mark_model_context_excluded(message)

    assert message.metadata == {
        "existing": "value",
        history.MODEL_CONTEXT_EXCLUDED_KEY: True,
    }
