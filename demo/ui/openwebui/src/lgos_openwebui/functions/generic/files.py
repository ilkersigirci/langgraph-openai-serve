"""Bridge Open WebUI attachments and generated Responses files."""

from base64 import b64decode
from binascii import Error as Base64Error
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import httpx
from openai.types.responses import ResponseFunctionToolCall

from .api import _client
from .contracts import (
    DISPLAY_FILE_TOOL_NAME,
    PLOTLY_MEDIA_TYPE,
    DisplayFileArguments,
    PlotlyFigure,
)


async def _with_response_file_parts(
    messages: list[dict[str, Any]],
    files: list[dict[str, Any]] | None,
    metadata: dict[str, Any] | None,
    *,
    base_url: str,
    api_key: str,
    timeout: float,
    provider: str,
) -> list[dict[str, Any]]:
    """Upload this turn's files and attach native Responses input parts."""
    current_files = _current_files(metadata)
    if not current_files:
        return messages
    user_message_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )
    if user_message_index is None:
        return messages

    current_file_ids = {cast(str, file["id"]) for file in current_files}
    path_attachments = {
        file_id: _path_attachment(file)
        for file in files or []
        if isinstance(file, dict)
        and file.get("type") == "file"
        and isinstance(file_id := file.get("id"), str)
        and file_id in current_file_ids
    }
    images = iter(_image_attachments(messages[user_message_index]))
    attachments: list[tuple[Path | bytes, str, str]] = []
    for file in current_files:
        file_id = cast(str, file["id"])
        if attachment := path_attachments.get(file_id):
            attachments.append(attachment)
            continue
        if _content_type(file).startswith("image/"):
            try:
                content, content_type = next(images)
            except StopIteration as exc:
                msg = f"Open WebUI attachment is unavailable: {_filename(file)}"
                raise ValueError(msg) from exc
            attachments.append((content, _filename(file), content_type))
            continue
        msg = f"Open WebUI attachment is unavailable: {_filename(file)}"
        raise ValueError(msg)

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
    content = message.get("content")
    if isinstance(content, str):
        content_parts: list[Any] = (
            [{"type": "input_text", "text": content}] if content else []
        )
    elif isinstance(content, list):
        content_parts = list(content)
    else:
        content_parts = []
    updated = {**message, "content": [*content_parts, *parts]}
    return [
        *messages[:user_message_index],
        updated,
        *messages[user_message_index + 1 :],
    ]


def _current_files(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return files attached to this turn, excluding Open WebUI's active history."""
    user_message = (metadata or {}).get("user_message")
    if not isinstance(user_message, dict):
        return []
    files = user_message.get("files")
    if not isinstance(files, list):
        return []
    return [
        file
        for file in files
        if isinstance(file, dict)
        and file.get("type") == "file"
        and isinstance(file.get("id"), str)
        and file["id"]
    ]


def _path_attachment(file: dict[str, Any]) -> tuple[Path, str, str]:
    stored = file.get("file")
    if not isinstance(stored, dict):
        raise ValueError("Open WebUI returned an invalid file attachment.")
    path_value = stored.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("Open WebUI returned an invalid file attachment.")

    path = Path(path_value)
    return path, _filename(file, fallback=path.name), _content_type(file)


def _image_attachments(
    message: dict[str, Any],
) -> list[tuple[bytes, str]]:
    content = message.get("content")
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


def _filename(file: dict[str, Any], *, fallback: str | None = None) -> str:
    stored = file.get("file")
    stored_filename = stored.get("filename") if isinstance(stored, dict) else None
    filename = stored_filename or file.get("name") or fallback
    if not isinstance(filename, str) or not filename:
        raise ValueError("Open WebUI returned an invalid file attachment.")
    return filename


def _content_type(file: dict[str, Any]) -> str:
    stored = file.get("file")
    metadata = stored.get("meta") if isinstance(stored, dict) else None
    content_type = file.get("content_type") or (
        metadata.get("content_type") if isinstance(metadata, dict) else None
    )
    return (
        content_type
        if isinstance(content_type, str) and content_type
        else "application/octet-stream"
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
    event_emitter: Any,
    request: Any,
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
    request: Any,
    *,
    filename: str,
    media_type: str,
    content: bytes,
    timeout: float,
) -> str:
    """Upload bytes through the authenticated Open WebUI Files endpoint."""
    headers = getattr(request, "headers", None)
    authorization = headers.get("authorization") if headers is not None else None
    base_url = getattr(request, "base_url", None)
    if not isinstance(authorization, str) or not authorization or base_url is None:
        msg = "Open WebUI request credentials are unavailable for file storage."
        raise ValueError(msg)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{str(base_url).rstrip('/')}/api/v1/files/",
            params={"process": "false"},
            headers={"Authorization": authorization},
            files={"file": (filename, content, media_type)},
        )
        response.raise_for_status()
        payload = response.json()
    file_id = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(file_id, str) or not file_id:
        msg = "Open WebUI returned an invalid stored file."
        raise ValueError(msg)
    return file_id
