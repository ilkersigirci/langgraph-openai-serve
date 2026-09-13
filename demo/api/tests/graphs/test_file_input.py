"""Behavior tests for the file-input demo graph."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph_openai_serve import GraphFeature

from lgos_demo_api.graphs import file_input as file_input_module


def test_graph_advertises_file_inputs() -> None:
    assert file_input_module.file_input_graph_config.supports(GraphFeature.FILE_INPUTS)


@pytest.mark.parametrize("outcome", ["completed", "refusal", "incomplete"])
async def test_file_inputs_use_responses_and_preserve_provider_output(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
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
    clients: list[Any] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.files = SimpleNamespace(
                retrieve=AsyncMock(side_effect=lambda file_id: filenames[file_id]),
                content=AsyncMock(side_effect=lambda file_id: downloads[file_id]),
            )
            clients.append(self)

        async def __aenter__(self) -> "FakeOpenAI":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

    monkeypatch.setattr(file_input_module, "AsyncOpenAI", FakeOpenAI)

    requests = []
    status = "incomplete" if outcome == "incomplete" else "completed"

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_file",
                "object": "response",
                "created_at": 1,
                "model": "file-model",
                "status": status,
                "incomplete_details": (
                    {"reason": "max_output_tokens"} if status == "incomplete" else None
                ),
                "output": [
                    {
                        "id": "msg_file",
                        "type": "message",
                        "role": "assistant",
                        "status": status,
                        "content": [
                            {"type": "refusal", "refusal": "I cannot read that."}
                            if outcome == "refusal"
                            else {
                                "type": "output_text",
                                "text": "Summary",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(
            model="file-model",
            base_url="https://model.test/v1",
            api_key="DUMMY",
            http_async_client=client,
            use_responses_api=True,
        )
        monkeypatch.setattr(file_input_module, "ChatOpenAI", lambda **_: model)
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

    answer = result["messages"][-1]
    assert answer.text == ("" if outcome == "refusal" else "Summary")
    assert answer.response_metadata["status"] == status
    assert answer.usage_metadata["total_tokens"] == 5
    if outcome == "refusal":
        assert answer.content_blocks[0]["value"]["refusal"] == "I cannot read that."
    if outcome == "incomplete":
        assert answer.response_metadata["incomplete_details"] == {
            "reason": "max_output_tokens"
        }
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
    assert len(requests) == 1
    assert requests[0]["input"] == [
        {
            "type": "message",
            "role": "system",
            "content": file_input_module.INSTRUCTIONS,
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What do these show?"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aW1hZ2U=",
                },
                {
                    "type": "input_file",
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,ZG9jdW1lbnQ=",
                },
            ],
        },
    ]


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
