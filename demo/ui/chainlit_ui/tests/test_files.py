"""Attachment uploads and generated-file display through the gateway Files API."""

import json
import tomllib
from pathlib import Path

import chainlit as cl
import httpx2
from chainlit.config import config
from chainlit_utils.openai.files import file_upload_overrides

import lgos_chainlit
from lgos_chainlit import chat, display_files
from tests.support import (
    function_call,
    message,
    response,
    streamed,
    transcript,
    user_message,
)


def test_packaged_chainlit_config_enables_file_attachments() -> None:
    config_path = Path(lgos_chainlit.__file__).parent / ".chainlit" / "config.toml"

    with config_path.open("rb") as config_file:
        upload = tomllib.load(config_file)["features"]["spontaneous_file_upload"]

    assert upload == {
        "enabled": True,
        "accept": ["*/*"],
        "max_files": 5,
        "max_size_mb": 10,
    }


def _attach_notes(chainlit_context, tmp_path: Path) -> cl.Message:
    chainlit_context.session.chat_profile = "lgos-a/file-input"
    # Chainlit applies the selected profile's overrides to WebSocket sessions.
    chainlit_context.session.config = config.with_overrides(file_upload_overrides(True))
    notes = tmp_path / "notes.txt"
    notes.write_text("Quarterly notes")
    return user_message(
        "Summarize it.",
        elements=[cl.File(name="notes.txt", path=str(notes), mime="text/plain")],
    )


async def test_attachments_upload_through_the_gateway_files_route(
    chainlit_context,
    fake_gateway,
    tmp_path: Path,
) -> None:
    fake_gateway.replies += [
        httpx2.Response(
            200,
            json={
                "id": "file-notes",
                "object": "file",
                "bytes": 15,
                "created_at": 0,
                "filename": "notes.txt",
                "purpose": "user_data",
                "status": "processed",
            },
        ),
        streamed(response(message("Summary."))),
    ]

    await chat.on_message(_attach_notes(chainlit_context, tmp_path))

    upload = fake_gateway.requests[0]
    assert (upload.method, upload.url.path, dict(upload.url.params)) == (
        "POST",
        "/v1/files",
        {"provider": "litellm_proxy"},
    )
    assert fake_gateway.bodies("/v1/responses")[0]["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize it."},
                {"type": "input_file", "file_id": "file-notes"},
            ],
        }
    ]
    assert transcript()[-1] == "Summary."


async def test_failed_upload_is_visible_and_sends_no_response_request(
    chainlit_context,
    fake_gateway,
    tmp_path: Path,
) -> None:
    fake_gateway.replies.append(httpx2.Response(503, json={"error": "unavailable"}))

    await chat.on_message(_attach_notes(chainlit_context, tmp_path))

    assert [request.url.path for request in fake_gateway.requests] == ["/v1/files"]
    assert transcript()[-1].startswith("Response failed: Error code: 503")


async def test_display_file_persists_an_interactive_plotly_element(
    chainlit_context,
    fake_gateway,
) -> None:
    chart = {"data": [{"type": "bar", "x": ["Q1", "Q2"], "y": [120, 180]}]}
    fake_gateway.replies.append(httpx2.Response(200, json=chart))
    call = function_call(
        "display_file",
        json.dumps(
            {
                "file_id": "file-chart",
                "filename": "chart.plotly.json",
                "media_type": "application/vnd.plotly.v1+json",
                "title": "Quarterly revenue",
                "alt": "Q4 is highest.",
            }
        ),
        call_id="call_chart",
    )

    output = await display_files.display_file(call)

    assert output == {
        "type": "function_call_output",
        "call_id": "call_chart",
        "output": '{"displayed":true}',
    }
    [download] = fake_gateway.requests
    assert download.url.path == "/v1/files/file-chart/content"
    [shown] = cl.chat_context.get()
    assert shown.content == "Quarterly revenue"
    assert shown.metadata == {"lgos_chainlit.exclude_from_model_context": True}
    [element] = shown.elements
    assert isinstance(element, cl.Plotly)
    assert element.display == "inline"
    assert json.loads(element.content)["data"] == chart["data"]
