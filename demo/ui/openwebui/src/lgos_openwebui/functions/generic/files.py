"""Bridge Open WebUI attachments and generated Responses files."""

from base64 import b64decode
from binascii import Error as Base64Error
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
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
    OpenWebUIFile,
    OpenWebUIMessage,
    OpenWebUIMetadata,
    PlotlyFigure,
)


async def _with_response_file_parts(
    messages: Sequence[OpenWebUIMessage],
    files: Sequence[OpenWebUIFile],
    metadata: OpenWebUIMetadata,
    request: object | None = None,
    *,
    base_url: str,
    api_key: str,
    timeout: float,
    provider: str,
) -> list[OpenWebUIMessage]:
    """Upload this turn's files and attach native Responses input parts."""
    current_files = _current_files(metadata)
    if not current_files:
        return list(messages)
    user_message_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].role == "user"
        ),
        None,
    )
    if user_message_index is None:
        return list(messages)

    current_file_ids = {file.id for file in current_files if file.id is not None}
    path_attachments = {
        file.id: attachment
        for file in files
        if file.type == "file"
        and file.id in current_file_ids
        and (attachment := _path_attachment(file)) is not None
    }
    images = iter(_image_attachments(messages[user_message_index]))
    attachments: list[tuple[Path | bytes, str, str]] = []
    for file in current_files:
        file_id = file.id
        if file_id is None:
            continue
        image_attachment = None
        if _content_type(file).startswith("image/"):
            image_attachment = next(images, None)
        if attachment := path_attachments.get(file_id):
            attachments.append(attachment)
            continue
        if image_attachment is not None:
            content, content_type = image_attachment
            attachments.append((content, _filename(file), content_type))
            continue
        filename = _filename(file)
        content, content_type = await _download_openwebui_file(
            request,
            file_id,
            filename=filename,
            timeout=timeout,
        )
        attachments.append((content, filename, content_type or _content_type(file)))

    parts: list[dict[str, str]] = []
    async with _client(base_url=base_url, api_key=api_key, timeout=timeout) as client:
        for source, filename, content_type in attachments:
            try:
                content = (
                    source.open("rb") if isinstance(source, Path) else BytesIO(source)
                )
                with content:
                    uploaded = await client.files.create(
                        file=(filename, content, content_type),
                        purpose="user_data",
                        extra_query={"provider": provider},
                    )
            except OSError as exc:
                msg = f"Open WebUI attachment is unavailable: {filename}"
                raise ValueError(msg) from exc
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


def _current_files(metadata: OpenWebUIMetadata) -> list[OpenWebUIFile]:
    """Return files attached to this turn, excluding Open WebUI's active history."""
    if metadata.user_message is None:
        return []
    return [
        file for file in metadata.user_message.files if file.type == "file" and file.id
    ]


def _path_attachment(file: OpenWebUIFile) -> tuple[Path, str, str] | None:
    if file.file is None:
        return None
    path_value = file.file.path
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_file():
        return None
    return path, _filename(file, fallback=path.name), _content_type(file)


async def _download_openwebui_file(
    request: object | None,
    file_id: str,
    *,
    filename: str,
    timeout: float,
) -> tuple[bytes, str | None]:
    """Download an attachment through the authenticated Open WebUI API."""
    client, headers = _openwebui_client(request, timeout)
    try:
        async with client as openwebui:
            response = await openwebui.get(
                f"/api/v1/files/{quote(file_id, safe='')}/content",
                headers=headers,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        msg = f"Open WebUI attachment is unavailable: {filename}"
        raise ValueError(msg) from exc
    content_type = response.headers.get("content-type")
    return response.content, content_type if content_type else None


def _image_attachments(
    message: OpenWebUIMessage,
) -> list[tuple[bytes, str]]:
    content = message.content
    if not isinstance(content, list):
        return []

    images = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(url, str):
            continue
        header, separator, encoded = url.partition(",")
        if (
            not separator
            or not header.startswith("data:image/")
            or ";base64" not in header
        ):
            continue
        content_type = header.removeprefix("data:").partition(";")[0]
        try:
            images.append((b64decode(encoded, validate=True), content_type))
        except (Base64Error, ValueError) as exc:
            raise ValueError(
                "Open WebUI returned an invalid image attachment."
            ) from exc
    return images


def _filename(file: OpenWebUIFile, *, fallback: str | None = None) -> str:
    stored_filename = file.file.filename if file.file is not None else None
    filename = stored_filename or file.name or fallback
    if not filename:
        raise ValueError("Open WebUI returned an invalid file attachment.")
    return filename


def _content_type(file: OpenWebUIFile) -> str:
    metadata = file.file.meta if file.file is not None else None
    return (
        file.content_type
        or (metadata.content_type if metadata is not None else None)
        or "application/octet-stream"
    )


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
    request: object | None,
    *,
    files_base_url: str,
    api_key: str,
    timeout: float,
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
        stored_id = await _store_openwebui_file(
            request,
            filename=arguments.filename,
            media_type=arguments.media_type,
            content=content,
            timeout=timeout,
        )
        event = {
            "type": "files",
            "data": {
                "files": [
                    {
                        "type": "image",
                        "url": f"/api/v1/files/{stored_id}/content",
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


async def _store_openwebui_file(
    request: object | None,
    *,
    filename: str,
    media_type: str,
    content: bytes,
    timeout: float,
) -> str:
    """Upload bytes through the authenticated Open WebUI Files endpoint."""
    client, headers = _openwebui_client(request, timeout)
    async with client as openwebui:
        response = await openwebui.post(
            "/api/v1/files/",
            params={"process": "false"},
            headers=headers,
            files={"file": (filename, content, media_type)},
        )
        response.raise_for_status()
        payload = response.json()
    file_id = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(file_id, str) or not file_id:
        msg = "Open WebUI returned an invalid stored file."
        raise ValueError(msg)
    return file_id


def _openwebui_client(
    request: object | None,
    timeout: float,
) -> tuple[httpx.AsyncClient, dict[str, str]]:
    """Build an authenticated client for the current Open WebUI application."""
    headers = getattr(request, "headers", None)
    authorization = headers.get("authorization") if headers is not None else None
    if not isinstance(authorization, str) or not authorization:
        msg = "Open WebUI request credentials are unavailable for file transfer."
        raise ValueError(msg)

    app = getattr(request, "app", None)
    if app is not None:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://openwebui.internal",
            timeout=timeout,
        )
    else:
        base_url = getattr(request, "base_url", None)
        if base_url is None:
            msg = "Open WebUI request URL is unavailable for file transfer."
            raise ValueError(msg)
        client = httpx.AsyncClient(base_url=str(base_url), timeout=timeout)
    return client, {"Authorization": authorization}
