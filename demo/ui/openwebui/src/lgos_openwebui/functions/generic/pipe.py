"""Open WebUI manifold Pipe entrypoint for registered LGOS models."""

import os
from collections.abc import AsyncIterator
from typing import Any

from openai import OpenAIError
from pydantic import BaseModel, Field

from .api import (
    _chat,
    _chat_completion,
    _client,
    _list_model_ids,
    _model_id,
    _model_request,
    _request_options,
)
from .contracts import (
    CLIENT_EVENTS_FEATURE,
    NO_CHOICES_MESSAGE,
    InterruptCancelled,
    PipeChunk,
    PipeResponse,
)
from .events import _chart_embed_event, _client_event, _status_event
from .interrupts import (
    _interrupt_cancelled_response,
    _openwebui_chunk,
    _openwebui_completion,
    _request_messages,
)
from .metadata import (
    _emit_limited_functionality_warning,
    _extension_supports,
    _model_extension,
    _request_metadata,
    _retrieve_model,
)


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = Field(
            default=os.environ.get(
                "OPENAI_API_BASE_URL",
                "http://lgos-bifrost:8080/openai_passthrough/v1",
            ),
            description="OpenAI-compatible base URL used for retrieval and chat.",
        )
        OPENAI_CATALOG_BASE_URL: str = Field(
            default=os.environ.get(
                "OPENAI_CATALOG_BASE_URL",
                "http://lgos-bifrost:8080/v1",
            ),
            description="OpenAI-compatible base URL used to list the model catalog.",
        )
        OPENAI_API_KEY: str = Field(
            default=os.environ.get("OPENAI_API_KEY", "DUMMY"),
            description="API key sent to the configured OpenAI-compatible endpoints.",
            json_schema_extra={"input": {"type": "password"}},
        )
        OPENAI_API_TIMEOUT: float = Field(
            default=30,
            gt=0,
            description="OpenAI-compatible request timeout in seconds.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def pipes(self) -> list[dict[str, str]]:
        """Expose every registered LangGraph model to Open WebUI."""
        async with _client(
            base_url=self.valves.OPENAI_CATALOG_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            timeout=self.valves.OPENAI_API_TIMEOUT,
        ) as client:
            model_ids = await _list_model_ids(client)
            return [
                {
                    "id": model_id,
                    "name": f"Generic / {model_id}",
                }
                for model_id in model_ids
            ]

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
    ) -> PipeResponse:
        """Use the same Chat Completions mode requested by Open WebUI."""
        if body.get("stream") is True:
            return self._stream(
                body,
                __event_emitter__=__event_emitter__,
                __metadata__=__metadata__,
                __user__=__user__,
            )
        return await self._complete(
            body,
            __event_emitter__=__event_emitter__,
            __metadata__=__metadata__,
            __user__=__user__,
        )

    async def _stream(
        self,
        body: dict[str, Any],
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
    ) -> AsyncIterator[PipeChunk]:
        """Forward chat, leaving tool execution and interaction to Open WebUI.

        Yields:
            PipeChunk objects for Open WebUI stream.
        """
        openwebui_metadata = __metadata__ or {}
        base_url = self.valves.OPENAI_API_BASE_URL
        api_key = self.valves.OPENAI_API_KEY
        timeout = self.valves.OPENAI_API_TIMEOUT

        try:
            model_id = _model_id(body)
            _model_request(model_id)
            messages = _request_messages(body)
        except InterruptCancelled:
            yield _interrupt_cancelled_response(model_id, streaming=True)
            return
        except ValueError as exc:
            yield _error(str(exc))
            return

        try:
            async with _client(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            ) as client:
                model = await _retrieve_model(client, model_id)
                extension = _model_extension(model)
                if extension is None:
                    await _emit_limited_functionality_warning(__event_emitter__)
                request_metadata = _request_metadata(
                    model=model,
                    metadata=openwebui_metadata,
                )
                include_client_events = _extension_supports(
                    extension,
                    CLIENT_EVENTS_FEATURE,
                )
                async with _chat(
                    client=client,
                    messages=messages,
                    model_id=model_id,
                    request_metadata=request_metadata,
                    include_client_events=include_client_events,
                    user_id=_user_id(__user__),
                    request_options=_request_options(body, stream=True),
                ) as stream:
                    async for chunk in stream:
                        client_event = _client_event(chunk)
                        if client_event is not None:
                            ui_event = _status_event(
                                client_event
                            ) or _chart_embed_event(client_event)
                            if __event_emitter__ is not None and ui_event is not None:
                                await __event_emitter__(ui_event)
                            continue

                        yield _openwebui_chunk(chunk)
        except ValueError as exc:
            yield _error(str(exc))
        except OpenAIError as exc:
            yield _error(f"Error calling LangGraph API: {exc}")

    async def _complete(
        self,
        body: dict[str, Any],
        __event_emitter__: Any,
        __metadata__: dict[str, Any] | None,
        __user__: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return one complete OpenAI response without replaying stream events."""
        try:
            model_id = _model_id(body)
            _model_request(model_id)
            messages = _request_messages(body)
        except InterruptCancelled:
            return _interrupt_cancelled_response(model_id, streaming=False)
        except ValueError as exc:
            return _error(str(exc))

        try:
            async with _client(
                base_url=self.valves.OPENAI_API_BASE_URL,
                api_key=self.valves.OPENAI_API_KEY,
                timeout=self.valves.OPENAI_API_TIMEOUT,
            ) as client:
                model = await _retrieve_model(client, model_id)
                if _model_extension(model) is None:
                    await _emit_limited_functionality_warning(__event_emitter__)
                request_metadata = _request_metadata(
                    model=model,
                    metadata=__metadata__ or {},
                )
                response = await _chat_completion(
                    client=client,
                    messages=messages,
                    model_id=model_id,
                    request_metadata=request_metadata,
                    user_id=_user_id(__user__),
                    request_options=_request_options(body, stream=False),
                )
                if not response.choices:
                    return _error(NO_CHOICES_MESSAGE)

                return _openwebui_completion(response)
        except ValueError as exc:
            return _error(str(exc))
        except OpenAIError as exc:
            return _error(f"Error calling LangGraph API: {exc}")


def _user_id(user: dict[str, Any] | None) -> str | None:
    user_id = (user or {}).get("id")
    return user_id if isinstance(user_id, str) and user_id else None


def _error(detail: str) -> dict[str, Any]:
    return {"error": {"detail": detail}}
