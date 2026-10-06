"""Bridge Open WebUI attachments and generated Responses files."""

from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

import httpx
from openai.types.responses import ResponseFunctionToolCall

from .api import _client
from .contracts import (
    DISPLAY_FILE_TOOL_NAME,
    PLOTLY_MEDIA_TYPE,
    DisplayFileArguments,
    OpenWebUIEventEmitter,
    OpenWebUIMessage,
    OpenWebUIMetadata,
    OpenWebUIRequest,
    PlotlyFigure,
)


async def _with_response_file_parts(
    messages: Sequence[OpenWebUIMessage],
    metadata: OpenWebUIMetadata,
    request: OpenWebUIRequest | None,
    *,
    base_url: str,
    api_key: str,
    timeout: float,  # ruff: ignore[async-function-with-timeout] - Forward the HTTP client timeout; this is not a task cancellation scope.
    provider: str,
) -> list[OpenWebUIMessage]:
    """Upload this turn's files and attach native Responses input parts."""
    # Unlike __files__, the user message lists only files attached this turn.
    files = [
        file
        for file in (metadata.user_message.files if metadata.user_message else [])
        if file.type == "file" and file.id
    ]
    user_message_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].role == "user"
        ),
        None,
    )
    if not files or user_message_index is None:
        return list(messages)

    parts: list[dict[str, str]] = []
    async with (
        _openwebui_client(request, timeout) as openwebui,
        _client(base_url=base_url, api_key=api_key, timeout=timeout) as client,
    ):
        for file in files:
            try:
                response = await openwebui.get(
                    f"/api/v1/files/{quote(file.id, safe='')}/content"
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                msg = f"Open WebUI attachment is unavailable: {file.name}"
                raise ValueError(msg) from exc
            uploaded = await client.files.create(
                file=(file.name, response.content, response.headers["content-type"]),
                purpose="user_data",
                extra_query={"provider": provider},
            )
            parts.append({"type": "input_file", "file_id": uploaded.id})

    message = messages[user_message_index]
    content = message.content
    if isinstance(content, str):
        content_parts: list[Any] = (
            [{"type": "input_text", "text": content}] if content else []
        )
    elif isinstance(content, list):
        content_parts = list(content)
    else:
        content_parts = []
    updated = message.model_copy(update={"content": [*content_parts, *parts]})
    return [
        *messages[:user_message_index],
        updated,
        *messages[user_message_index + 1 :],
    ]


def _plotly_html(content: bytes) -> str:
    figure = PlotlyFigure.model_validate_json(content).model_dump_json(
        exclude_unset=True
    )
    # JSON is embedded inside a script: prevent labels from closing that element.
    figure = figure.replace("<", "\\u003c")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-4.0.0.min.js" charset="utf-8"></script>
</head><body style="margin:0">
<div id="plot" style="height:450px"></div>
<script>
const figure = {figure};
Plotly.newPlot("plot", {{...figure, config: {{responsive: true}}}}).then(plot => {{
  parent.postMessage({{type: "iframe:height", height: plot.offsetHeight}}, "*");
}});
</script></body></html>"""


async def _handle_display_file(
    call: ResponseFunctionToolCall,
    event_emitter: OpenWebUIEventEmitter | None,
    request: OpenWebUIRequest | None,
    *,
    files_base_url: str,
    api_key: str,
    timeout: float,  # ruff: ignore[async-function-with-timeout] - Forward the HTTP client timeout; this is not a task cancellation scope.
    provider: str,
) -> dict[str, str]:
    """Persist a generated image or interactive chart through native UI events."""
    if call.name != DISPLAY_FILE_TOOL_NAME:
        msg = f"Unsupported client function: {call.name}"
        raise ValueError(msg)
    if event_emitter is None:
        msg = "Open WebUI did not provide an event emitter for display_file."
        raise ValueError(msg)
    try:
        arguments = DisplayFileArguments.model_validate_json(call.arguments)
    except ValueError as exc:
        msg = "The display_file call contains invalid arguments."
        raise ValueError(msg) from exc

    async with _client(
        base_url=files_base_url,
        api_key=api_key,
        timeout=timeout,
    ) as client:
        download = await client.files.content(
            arguments.file_id,
            extra_query={"provider": provider},
        )
        content = await download.aread()

    if arguments.media_type == PLOTLY_MEDIA_TYPE:
        event = {"type": "embeds", "data": {"embeds": [_plotly_html(content)]}}
    else:
        async with _openwebui_client(request, timeout) as openwebui:
            response = await openwebui.post(
                "/api/v1/files/",
                params={"process": "false"},
                files={"file": (arguments.filename, content, arguments.media_type)},
            )
            response.raise_for_status()
        event = {
            "type": "files",
            "data": {
                "files": [
                    {
                        "type": "image",
                        "url": f"/api/v1/files/{response.json()['id']}/content",
                        "name": arguments.filename,
                    }
                ]
            },
        }
    await event_emitter(event)
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": '{"displayed":true}',
    }


def _openwebui_client(
    request: OpenWebUIRequest | None,
    timeout: float,
) -> httpx.AsyncClient:
    """Call the running Open WebUI app in-process with the caller's credentials."""
    authorization = request.headers.get("authorization") if request else None
    if request is None or not authorization:
        msg = "Open WebUI request credentials are unavailable for file transfer."
        raise ValueError(msg)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=request.app),
        base_url="http://openwebui.internal",
        headers={"Authorization": authorization},
        timeout=timeout,
    )
