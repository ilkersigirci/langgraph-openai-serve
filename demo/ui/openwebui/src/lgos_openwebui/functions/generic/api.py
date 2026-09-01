"""OpenAI-compatible client and Chat Completions helpers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from .contracts import (
    CHAT_COMPLETION_REQUEST_FIELDS,
    LGOS_MODEL_OWNER,
    STREAM_EVENTS_METADATA_KEY,
    STREAM_EVENTS_METADATA_VALUE,
)


@asynccontextmanager
async def _chat(
    *,
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    model_id: str,
    request_metadata: dict[str, str] | None = None,
    include_client_events: bool = False,
    user_id: str | None = None,
    request_options: dict[str, Any] | None = None,
) -> AsyncIterator[AsyncStream[ChatCompletionChunk]]:
    metadata = dict(request_metadata or {})
    if include_client_events:
        metadata[STREAM_EVENTS_METADATA_KEY] = STREAM_EVENTS_METADATA_VALUE

    request = _chat_request(model_id, messages, metadata, user_id, request_options)

    async with await client.chat.completions.create(
        **request,
        stream=True,
    ) as stream:
        yield stream


async def _chat_completion(
    *,
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    model_id: str,
    request_metadata: dict[str, str] | None = None,
    user_id: str | None = None,
    request_options: dict[str, Any] | None = None,
) -> ChatCompletion:
    """Create one non-streaming Chat Completion."""
    request = _chat_request(
        model_id,
        messages,
        request_metadata,
        user_id,
        request_options,
    )
    return await client.chat.completions.create(**request)


def _chat_request(
    model_id: str,
    messages: list[ChatCompletionMessageParam],
    metadata: dict[str, str] | None,
    user_id: str | None,
    request_options: dict[str, Any] | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        **_model_request(model_id),
        "messages": messages,
        **(request_options or {}),
    }
    if user_id is not None:
        request["user"] = user_id
    if metadata:
        request["metadata"] = metadata
    return request


def _request_options(body: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    """Forward only Chat Completions options supported by LGOS."""
    fields = (
        (*CHAT_COMPLETION_REQUEST_FIELDS, "stream_options")
        if stream
        else CHAT_COMPLETION_REQUEST_FIELDS
    )
    return {key: body[key] for key in fields if key in body}


def _client(
    *,
    base_url: str,
    api_key: str,
    timeout: float,
) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
        default_headers={"User-Agent": "lgos-openwebui"},
    )


def _model_id(body: dict[str, Any]) -> str:
    qualified_model_id = body.get("model")
    if not isinstance(qualified_model_id, str):
        msg = "Open WebUI did not provide a valid model ID."
        raise ValueError(msg)

    _, separator, model_id = qualified_model_id.partition(".")
    if not separator or not model_id:
        msg = "Open WebUI did not provide a valid model ID."
        raise ValueError(msg)

    return model_id


async def _list_model_ids(client: AsyncOpenAI) -> list[str]:
    models = await client.models.list()
    return [model.id for model in models.data if model.owned_by == LGOS_MODEL_OWNER]


def _model_request(model_id: str) -> dict[str, Any]:
    provider, separator, upstream_model = model_id.partition("/")
    if not provider or not separator or not upstream_model:
        msg = f"Bifrost model ID must use the provider/model format: {model_id!r}."
        raise ValueError(msg)

    return {
        "model": upstream_model,
        "extra_headers": {"x-model-provider": provider},
    }
