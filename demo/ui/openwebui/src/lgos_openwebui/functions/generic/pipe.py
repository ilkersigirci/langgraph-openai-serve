"""Open WebUI manifold Pipe backed exclusively by the Responses API."""

import os
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import OpenAIError
from openai.types.responses import Response, ResponseFunctionToolCall
from pydantic import BaseModel, Field

from .api import (
    _catalog_base_url,
    _client,
    _list_model_ids,
    _model_id,
    _model_request,
)
from .contracts import (
    DISPLAY_FILE_TOOL_NAME,
    INTERRUPT_CANCELLED_MESSAGE,
    INTERRUPT_TOOL_NAME,
    InterruptCancelled,
    PipeChunk,
    PipeResponse,
)
from .files import _handle_display_file, _with_response_file_parts
from .gateway import GatewayConfig, GatewayType, gateway_config
from .interrupts import (
    _ask_user_to_resume,
    _openwebui_interrupt_chunk,
    _openwebui_interrupt_completion,
)
from .metadata import _request_metadata
from .responses import (
    _emit_response_sources,
    _openwebui_text_chunk,
    _responses_continuation,
    _responses_final_text,
    _responses_function_calls,
    _responses_input,
    _responses_request,
)


class Pipe:
    class Valves(BaseModel):
        OPENAI_GATEWAY_TYPE: GatewayType = Field(
            default=cast(
                "GatewayType", os.environ.get("OPENAI_GATEWAY_TYPE", "litellm")
            ),
            description="Gateway used for all OpenAI requests.",
        )
        OPENAI_GATEWAY_BASE_URL: str | None = Field(
            default=os.environ.get("OPENAI_GATEWAY_BASE_URL") or None,
            description="Optional gateway root override.",
        )
        OPENAI_API_KEY: str = Field(
            default=os.environ.get("OPENAI_API_KEY", "sk-lgos-litellm-demo"),
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
        model_ids = []
        gateway = self._gateway()
        model_prefixes = gateway.model_prefixes
        catalogs = (
            tuple(
                (
                    _catalog_base_url(gateway.catalog_detail_base_url, model_prefix),
                    model_prefix,
                )
                for model_prefix in model_prefixes
            )
            if model_prefixes
            else ((gateway.catalog_base_url, None),)
        )
        for base_url, model_prefix in catalogs:
            async with _client(
                base_url=base_url,
                api_key=self.valves.OPENAI_API_KEY,
                timeout=self.valves.OPENAI_API_TIMEOUT,
            ) as client:
                model_ids.extend(
                    await _list_model_ids(client, model_prefix=model_prefix)
                )
        return [
            {"id": model_id, "name": f"Generic / {model_id}"} for model_id in model_ids
        ]

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __files__: list[dict[str, Any]] | None = None,
        __request__: Any = None,
    ) -> PipeResponse:
        """Run the selected graph through OpenAI Responses."""
        if body.get("stream") is True:
            return self._stream(
                body,
                __event_emitter__=__event_emitter__,
                __metadata__=__metadata__,
                __user__=__user__,
                __files__=__files__,
                __request__=__request__,
            )
        return await self._complete(
            body,
            __event_emitter__=__event_emitter__,
            __metadata__=__metadata__,
            __user__=__user__,
            __files__=__files__,
            __request__=__request__,
        )

    async def _stream(
        self,
        body: dict[str, Any],
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __files__: list[dict[str, Any]] | None = None,
        __request__: Any = None,
    ) -> AsyncIterator[PipeChunk]:
        """Yield Open WebUI chunks while the SDK owns Response accumulation."""
        try:
            model_id, input_items = await self._request_input(
                body, __metadata__, __files__
            )
        except InterruptCancelled:
            yield INTERRUPT_CANCELLED_MESSAGE
            return
        except ValueError as exc:
            yield _error(str(exc))
            return

        try:
            gateway = self._gateway()
            request = _responses_request(
                model_id,
                input_items,
                _request_metadata(__metadata__ or {}),
                _user_id(__user__),
                provider_routing=gateway.provider_routing,
                model_prefixes=gateway.model_prefixes,
            )
            async with _client(
                base_url=gateway.responses_base_url,
                api_key=self.valves.OPENAI_API_KEY,
                timeout=self.valves.OPENAI_API_TIMEOUT,
            ) as client:
                while True:
                    final_text_streamed = False
                    phases: dict[int, str | None] = {}
                    async with client.responses.stream(**request) as stream:
                        async for event in stream:
                            if event.type == "response.output_item.added":
                                if event.item.type == "message":
                                    phases[event.output_index] = event.item.phase
                            elif (
                                event.type == "response.output_text.delta"
                                and phases.get(event.output_index) != "commentary"
                            ):
                                final_text_streamed = True
                                yield _openwebui_text_chunk(model_id, event.delta)
                            elif (
                                event.type == "response.output_text.done"
                                and phases.get(event.output_index) == "commentary"
                                and __event_emitter__ is not None
                                and event.text
                            ):
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": event.text,
                                            "done": True,
                                        },
                                    }
                                )
                        response = cast("Response", await stream.get_final_response())

                    _raise_for_response(response)
                    await _emit_response_sources(response, __event_emitter__)
                    calls = _responses_function_calls(response)
                    if not calls:
                        if not final_text_streamed:
                            yield _openwebui_text_chunk(
                                model_id, _responses_final_text(response)
                            )
                        return
                    if _all_calls(calls, INTERRUPT_TOOL_NAME):
                        yield _openwebui_interrupt_chunk(model_id, calls)
                        return
                    if not _all_calls(calls, DISPLAY_FILE_TOOL_NAME):
                        raise ValueError(
                            "LangGraph API returned a mixed function-call batch."
                        )
                    outputs = [
                        await _handle_display_file(
                            call,
                            __event_emitter__,
                            __request__,
                            files_base_url=gateway.files_base_url,
                            api_key=self.valves.OPENAI_API_KEY,
                            timeout=self.valves.OPENAI_API_TIMEOUT,
                            provider=gateway.files_provider,
                        )
                        for call in calls
                    ]
                    request["input"].extend(_responses_continuation(response, outputs))
        except (ValueError, RuntimeError, OpenAIError) as exc:
            yield _error(f"Responses request failed: {exc}")

    async def _complete(
        self,
        body: dict[str, Any],
        __event_emitter__: Any,
        __metadata__: dict[str, Any] | None,
        __user__: dict[str, Any] | None,
        __files__: list[dict[str, Any]] | None,
        __request__: Any,
    ) -> PipeChunk:
        """Return native Pipe text or an ask-user call from Responses output."""
        answer_parts: list[str] = []
        try:
            model_id, input_items = await self._request_input(
                body, __metadata__, __files__
            )
        except InterruptCancelled:
            return INTERRUPT_CANCELLED_MESSAGE
        except ValueError as exc:
            return _error(str(exc))

        try:
            gateway = self._gateway()
            request = _responses_request(
                model_id,
                input_items,
                _request_metadata(__metadata__ or {}),
                _user_id(__user__),
                provider_routing=gateway.provider_routing,
                model_prefixes=gateway.model_prefixes,
            )
            async with _client(
                base_url=gateway.responses_base_url,
                api_key=self.valves.OPENAI_API_KEY,
                timeout=self.valves.OPENAI_API_TIMEOUT,
            ) as client:
                while True:
                    response = await client.responses.create(**request)
                    _raise_for_response(response)
                    await _emit_response_sources(response, __event_emitter__)
                    answer_parts.append(_responses_final_text(response))
                    calls = _responses_function_calls(response)
                    if not calls:
                        return "".join(answer_parts)
                    if _all_calls(calls, INTERRUPT_TOOL_NAME):
                        return _openwebui_interrupt_completion(model_id, calls)
                    if not _all_calls(calls, DISPLAY_FILE_TOOL_NAME):
                        raise ValueError(
                            "LangGraph API returned a mixed function-call batch."
                        )
                    outputs = [
                        await _handle_display_file(
                            call,
                            __event_emitter__,
                            __request__,
                            files_base_url=gateway.files_base_url,
                            api_key=self.valves.OPENAI_API_KEY,
                            timeout=self.valves.OPENAI_API_TIMEOUT,
                            provider=gateway.files_provider,
                        )
                        for call in calls
                    ]
                    request["input"].extend(_responses_continuation(response, outputs))
        except (ValueError, RuntimeError, OpenAIError) as exc:
            return _error(f"Responses request failed: {exc}")

    async def _request_input(
        self,
        body: dict[str, Any],
        metadata: dict[str, Any] | None,
        files: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        model_id = _model_id(body)
        gateway = self._gateway()
        _model_request(
            model_id,
            provider_routing=gateway.provider_routing,
            model_prefixes=gateway.model_prefixes,
        )
        raw_messages = body.get("messages")
        messages = raw_messages if isinstance(raw_messages, list) else []
        if resume := _ask_user_to_resume(messages):
            return model_id, resume
        messages = await _with_response_file_parts(
            messages,
            files,
            metadata,
            base_url=gateway.files_base_url,
            api_key=self.valves.OPENAI_API_KEY,
            timeout=self.valves.OPENAI_API_TIMEOUT,
            provider=gateway.files_provider,
        )
        return model_id, _responses_input(messages)

    def _gateway(self) -> GatewayConfig:
        return gateway_config(
            self.valves.OPENAI_GATEWAY_TYPE,
            self.valves.OPENAI_GATEWAY_BASE_URL,
            local=False,
        )


def _all_calls(calls: list[ResponseFunctionToolCall], name: str) -> bool:
    return bool(calls) and all(call.name == name for call in calls)


def _raise_for_response(response: Response) -> None:
    if response.status == "completed":
        return
    detail = response.error
    raise RuntimeError(detail.message if detail is not None else "Response failed.")


def _user_id(user: dict[str, Any] | None) -> str | None:
    user_id = (user or {}).get("id")
    return user_id if isinstance(user_id, str) and user_id else None


def _error(detail: str) -> dict[str, Any]:
    return {"error": {"detail": detail}}
