"""Behavior tests for the file-input demo graph."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call

import pytest
from langchain_core.messages import HumanMessage

from lgos_demo_api.graphs import file_input as file_input_module


async def test_file_ids_are_resolved_and_sent_as_responses_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads = {
        "file-image": SimpleNamespace(
            response=SimpleNamespace(headers={"content-type": "image/png"}),
            aread=AsyncMock(return_value=b"image"),
        ),
        "file-document": SimpleNamespace(
            response=SimpleNamespace(
                headers={"content-type": "application/octet-stream"}
            ),
            aread=AsyncMock(return_value=b"document"),
        ),
    }
    filenames = {
        "file-image": SimpleNamespace(filename="chart.png"),
        "file-document": SimpleNamespace(filename="report.pdf"),
    }
    create_response = AsyncMock(return_value=SimpleNamespace(output_text="Summary"))
    clients: list[Any] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.files = SimpleNamespace(
                retrieve=AsyncMock(side_effect=lambda file_id: filenames[file_id]),
                content=AsyncMock(side_effect=lambda file_id: downloads[file_id]),
            )
            self.responses = SimpleNamespace(create=create_response)
            clients.append(self)

        async def __aenter__(self) -> "FakeOpenAI":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

    monkeypatch.setattr(file_input_module, "AsyncOpenAI", FakeOpenAI)

    result = await file_input_module.file_input_graph.ainvoke(
        file_input_module.FileInputState(
            messages=[
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What do these show?"},
                        {"type": "file", "file": {"file_id": "file-image"}},
                        {"type": "file", "file": {"file_id": "file-document"}},
                    ]
                )
            ]
        )
    )

    assert result["messages"][-1].content == "Summary"
    assert clients[0].kwargs == {
        "base_url": file_input_module.settings.FILES_BASE_URL,
        "api_key": "DUMMY",
        "max_retries": 0,
    }
    clients[0].files.retrieve.assert_has_awaits(
        [call("file-image"), call("file-document")]
    )
    clients[0].files.content.assert_has_awaits(
        [call("file-image"), call("file-document")]
    )
    create_response.assert_awaited_once_with(
        model=file_input_module.settings.OPENAI_MODEL,
        instructions=file_input_module.INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What do these show?"},
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": "data:image/png;base64,aW1hZ2U=",
                    },
                    {
                        "type": "input_file",
                        "filename": "report.pdf",
                        "file_data": "data:application/pdf;base64,ZG9jdW1lbnQ=",
                    },
                ],
            }
        ],
    )


async def test_missing_file_returns_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncMock()
    monkeypatch.setattr(file_input_module, "AsyncOpenAI", client)

    result = await file_input_module.file_input_graph.ainvoke(
        file_input_module.FileInputState(
            messages=[HumanMessage(content="Summarize my file.")]
        )
    )

    assert result["messages"][-1].content == "Attach a file and try again."
    client.assert_not_called()
