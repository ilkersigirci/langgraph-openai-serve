"""Chainlit attachment upload tests."""

import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from lgos_chainlit import files as file_utils


def test_packaged_chainlit_config_enables_file_attachments() -> None:
    config_path = Path(file_utils.__file__).parent / ".chainlit" / "config.toml"

    with config_path.open("rb") as config_file:
        upload = tomllib.load(config_file)["features"]["spontaneous_file_upload"]

    assert upload == {
        "enabled": True,
        "accept": ["*/*"],
        "max_files": 5,
        "max_size_mb": 10,
    }


async def test_file_inputs_use_the_configured_gateway_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    upload = AsyncMock(return_value=[{"role": "user", "content": "uploaded"}])
    monkeypatch.setattr(
        file_utils,
        "files_request",
        lambda: (client, "litellm_proxy"),
    )
    monkeypatch.setattr(file_utils, "with_openai_response_file_parts", upload)
    input_items = [{"role": "user", "content": "Summarize it."}]
    message = Mock()

    result = await file_utils.with_response_file_parts(input_items, message)

    assert result == [{"role": "user", "content": "uploaded"}]
    upload.assert_awaited_once_with(
        input_items,
        message,
        client=client,
        extra_query={"provider": "litellm_proxy"},
    )
