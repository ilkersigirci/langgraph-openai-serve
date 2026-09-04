"""Translate Open WebUI attachments to OpenAI file content parts."""

from base64 import b64decode
from binascii import Error as Base64Error
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)

from .api import _client


async def _with_file_parts(
    messages: list[ChatCompletionMessageParam],
    files: list[dict[str, Any]] | None,
    metadata: dict[str, Any] | None,
    *,
    base_url: str,
    api_key: str,
    timeout: float,
    provider: str,
    supported: bool,
) -> list[ChatCompletionMessageParam]:
    """Upload this turn's files and attach their IDs to its user message."""
    current_files = _current_files(metadata)
    if not current_files:
        return messages
    if not supported:
        raise ValueError("The selected model does not support file inputs.")

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

    parts: list[ChatCompletionContentPartParam] = []
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
                        extra_query={"provider": provider} if provider else None,
                    )
            except OSError as exc:
                msg = f"Open WebUI attachment is unavailable: {filename}"
                raise ValueError(msg) from exc
            parts.append({"type": "file", "file": {"file_id": uploaded.id}})

    message = messages[user_message_index]
    content = message.get("content")
    if isinstance(content, str):
        content_parts: list[Any] = (
            [{"type": "text", "text": content}] if content else []
        )
    elif isinstance(content, list):
        content_parts = list(content)
    else:
        content_parts = []
    updated = {**message, "content": [*content_parts, *parts]}
    return [
        *messages[:user_message_index],
        cast("ChatCompletionMessageParam", updated),
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
    message: ChatCompletionMessageParam,
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
