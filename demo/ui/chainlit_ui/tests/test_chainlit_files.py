"""Chainlit attachment upload tests."""

import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletionMessageParam

from lgos_chainlit.utils import clients
from lgos_chainlit.utils import files as file_utils


def set_profile_file_upload(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
) -> None:
    """Set the effective Chainlit profile configuration for an upload test."""
    monkeypatch.setattr(
        file_utils,
        "chainlit_context",
        SimpleNamespace(
            session=SimpleNamespace(
                config=SimpleNamespace(
                    features=SimpleNamespace(
                        spontaneous_file_upload=SimpleNamespace(enabled=enabled)
                    )
                )
            )
        ),
    )


def test_packaged_chainlit_config_enables_file_attachments() -> None:
    config_path = Path(file_utils.__file__).parents[1] / ".chainlit" / "config.toml"

    with config_path.open("rb") as config_file:
        upload = tomllib.load(config_file)["features"]["spontaneous_file_upload"]

    assert upload == {
        "enabled": True,
        "accept": ["*/*"],
        "max_files": 5,
        "max_size_mb": 10,
    }


async def test_current_attachments_become_openai_file_parts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "chainlit-upload"
    path.write_bytes(b"file content")
    create = AsyncMock(return_value=SimpleNamespace(id="file-123"))
    set_profile_file_upload(monkeypatch, enabled=True)
    monkeypatch.setattr(clients.files_client.files, "create", create)
    monkeypatch.setattr(clients.settings.OPENAI, "files_provider", "lgos-files")
    message = SimpleNamespace(
        elements=[
            SimpleNamespace(
                path=str(path),
                name="report.pdf",
                mime="application/pdf",
            )
        ]
    )

    messages = await file_utils.with_file_parts(
        [
            cast(
                "ChatCompletionMessageParam",
                {"role": "user", "content": "Summarize it."},
            )
        ],
        message,
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize it."},
                {"type": "file", "file": {"file_id": "file-123"}},
            ],
        }
    ]
    create.assert_awaited_once()
    assert create.await_args.kwargs["purpose"] == "user_data"
    assert create.await_args.kwargs["extra_query"] == {"provider": "lgos-files"}
    filename, content, content_type = create.await_args.kwargs["file"]
    assert filename == "report.pdf"
    assert content.closed
    assert content_type == "application/pdf"


async def test_message_without_attachments_is_unchanged() -> None:
    messages = [
        cast("ChatCompletionMessageParam", {"role": "user", "content": "Hello"})
    ]

    result = await file_utils.with_file_parts(
        messages,
        SimpleNamespace(elements=[]),
    )

    assert result is messages


async def test_file_only_message_still_reaches_chat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")
    set_profile_file_upload(monkeypatch, enabled=True)
    monkeypatch.setattr(
        clients.files_client.files,
        "create",
        AsyncMock(return_value=SimpleNamespace(id="file-123")),
    )

    result = await file_utils.with_file_parts(
        [],
        SimpleNamespace(
            content="",
            elements=[
                SimpleNamespace(
                    path=str(path),
                    name="payload.bin",
                    mime="application/octet-stream",
                )
            ],
        ),
    )

    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": {"file_id": "file-123"}},
            ],
        }
    ]


async def test_unsupported_profile_rejects_before_central_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")
    create = AsyncMock()
    set_profile_file_upload(monkeypatch, enabled=False)
    monkeypatch.setattr(clients.files_client.files, "create", create)

    with pytest.raises(
        ValueError,
        match="The selected graph does not support file inputs",
    ):
        await file_utils.with_file_parts(
            [],
            SimpleNamespace(elements=[SimpleNamespace(path=str(path))]),
        )

    create.assert_not_awaited()
