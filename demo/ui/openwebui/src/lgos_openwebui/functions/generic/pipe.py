"""Open WebUI manifold Pipe backed exclusively by the Responses API."""

import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any, cast

from openai import OpenAIError
from openai.types.responses import Response, ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field

from .api import (
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
from .gateway import (
    GatewayConfig,
    GatewayRoot,
    GatewayType,
    gateway_config,
    litellm_models,
)
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
    _responses_tools,
)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        msg = f"{name} must be configured."
        raise RuntimeError(msg)
    return value


class Pipe:
    class Valves(BaseModel):
        model_config = ConfigDict(validate_default=True)

        OPENAI_GATEWAY_TYPE: GatewayType = Field(
            default_factory=lambda: cast(
                "GatewayType", _required_environment("OPENAI_GATEWAY_TYPE")
            ),
            description="Gateway used for all OpenAI requests.",
        )
        OPENAI_GATEWAY_BASE_URL: GatewayRoot = Field(
            default_factory=lambda: _required_environment("OPENAI_GATEWAY_BASE_URL"),
            description="Gateway root without the OpenAI API path.",
        )
        OPENAI_GATEWAY_API_KEY: str = Field(
            default_factory=lambda: _required_environment("OPENAI_GATEWAY_API_KEY"),
            min_length=1,
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
        gateway = self._gateway()
        async with _client(
            base_url=f"{gateway.root_url}/v1",
            api_key=self.valves.OPENAI_GATEWAY_API_KEY,
            timeout=self.valves.OPENAI_API_TIMEOUT,
        ) as client:
            if gateway.provider_routing:
                model_ids = await _list_model_ids(client)
            else:
                payload = await client.get(
                    f"{gateway.root_url}/model/info", cast_to=object
                )
                model_ids = [model.id for model in litellm_models(payload)]
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
        results = self._run(
            body,
            __event_emitter__=__event_emitter__,
            __metadata__=__metadata__,
            __user__=__user__,
            __files__=__files__,
            __request__=__request__,
        )
        if body.get("stream") is True:
            return results
        async with aclosing(results):
            return await anext(results)

    async def _run(
        self,
        body: dict[str, Any],
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __files__: list[dict[str, Any]] | None = None,
        __request__: Any = None,
    ) -> AsyncGenerator[PipeChunk, None]:
        """Own one Responses/tool loop for both Pipe response modes."""
        streaming = body.get("stream") is True
        answer_parts: list[str] = []
        latest_status = ""
        finished = False
        try:
            metadata = __metadata__ or {}
            model_id, input_items, previous_response_id = await self._request_input(
                body, __metadata__, __files__
            )
            gateway = self._gateway()
            request = _responses_request(
                model_id,
                input_items,
                _request_metadata(metadata),
                _user_id(__user__),
                provider_routing=gateway.provider_routing,
                tools=_responses_tools(model_id, metadata),
                previous_response_id=previous_response_id,
            )
            async with _client(
                base_url=gateway.responses_base_url,
                api_key=self.valves.OPENAI_GATEWAY_API_KEY,
                timeout=self.valves.OPENAI_API_TIMEOUT,
            ) as client:
                while True:
                    final_text_streamed = False
                    phases: dict[int, str | None] = {}
                    if streaming:
                        async with client.responses.stream(**request) as stream:
                            async for event in stream:
                                if event.type == "response.output_item.added":
                                    if event.item.type == "message":
                                        phases[event.output_index] = event.item.phase
                                elif (
                                    event.type == "response.output_text.delta"
                                    or event.type == "response.refusal.delta"
                                ) and phases.get(event.output_index) != "commentary":
                                    final_text_streamed = True
                                    yield _openwebui_text_chunk(model_id, event.delta)
                                elif (
                                    event.type == "response.incomplete"
                                    or event.type == "response.failed"
                                ):
                                    _raise_for_response(event.response)
                                elif event.type == "response.web_search_call.completed":
                                    await _emit_status(
                                        __event_emitter__,
                                        "Web search completed.",
                                        done=True,
                                    )
                                elif (
                                    event.type == "response.output_text.done"
                                    and phases.get(event.output_index) == "commentary"
                                    and event.text
                                ):
                                    latest_status = event.text
                                    await _emit_status(
                                        __event_emitter__, latest_status, done=False
                                    )
                            response = cast(
                                "Response", await stream.get_final_response()
                            )
                    else:
                        response = await client.responses.create(**request)

                    _raise_for_response(response)
                    await _emit_response_sources(response, __event_emitter__)
                    final_text = _responses_final_text(response)
                    if streaming and not final_text_streamed and final_text:
                        yield _openwebui_text_chunk(model_id, final_text)
                    answer_parts.append(final_text)
                    calls = _responses_function_calls(response)
                    if not calls:
                        finished = True
                        if not streaming:
                            yield "".join(answer_parts)
                        return
                    if _all_calls(calls, INTERRUPT_TOOL_NAME):
                        finished = True
                        yield (
                            _openwebui_interrupt_chunk(model_id, response.id, calls)
                            if streaming
                            else _openwebui_interrupt_completion(
                                model_id,
                                response.id,
                                calls,
                                content="".join(answer_parts),
                            )
                        )
                        return
                    if not _all_calls(calls, DISPLAY_FILE_TOOL_NAME):
                        raise ValueError(
                            "LangGraph API returned an unsupported or mixed function-call batch."
                        )
                    outputs = [
                        await _handle_display_file(
                            call,
                            __event_emitter__,
                            __request__,
                            files_base_url=gateway.files_base_url,
                            api_key=self.valves.OPENAI_GATEWAY_API_KEY,
                            timeout=self.valves.OPENAI_API_TIMEOUT,
                            provider=gateway.files_provider,
                        )
                        for call in calls
                    ]
                    if request.pop("previous_response_id", None) is not None:
                        # Interrupt answers belong only to the paused checkpoint.
                        # Client tools continue from the UI's transcript instead.
                        request["input"] = _responses_input(body["messages"])
                    request["input"].extend(_responses_continuation(response, outputs))
        except InterruptCancelled:
            yield INTERRUPT_CANCELLED_MESSAGE
        except (ValueError, RuntimeError, OpenAIError) as exc:
            yield _error(f"Responses request failed: {exc}")
        finally:
            if latest_status:
                await _emit_status(
                    __event_emitter__,
                    latest_status if finished else f"Stopped: {latest_status}",
                    done=True,
                )

    async def _request_input(
        self,
        body: dict[str, Any],
        metadata: dict[str, Any] | None,
        files: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]], str | None]:
        model_id = _model_id(body)
        gateway = self._gateway()
        _model_request(
            model_id,
            provider_routing=gateway.provider_routing,
        )
        raw_messages = body.get("messages")
        messages = raw_messages if isinstance(raw_messages, list) else []
        if resume := _ask_user_to_resume(messages):
            input_items, previous_response_id = resume
            return model_id, input_items, previous_response_id
        messages = await _with_response_file_parts(
            messages,
            files,
            metadata,
            base_url=gateway.files_base_url,
            api_key=self.valves.OPENAI_GATEWAY_API_KEY,
            timeout=self.valves.OPENAI_API_TIMEOUT,
            provider=gateway.files_provider,
        )
        return model_id, _responses_input(messages), None

    def _gateway(self) -> GatewayConfig:
        return gateway_config(
            self.valves.OPENAI_GATEWAY_TYPE,
            self.valves.OPENAI_GATEWAY_BASE_URL,
        )


def _all_calls(calls: list[ResponseFunctionToolCall], name: str) -> bool:
    return bool(calls) and all(call.name == name for call in calls)


def _raise_for_response(response: Response) -> None:
    if response.status == "completed":
        return
    if response.status == "incomplete":
        reason = response.incomplete_details
        raise RuntimeError(
            f"Response incomplete: {reason.reason if reason else 'unknown reason'}."
        )
    detail = response.error
    raise RuntimeError(detail.message if detail is not None else "Response failed.")


async def _emit_status(event_emitter: Any, description: str, *, done: bool) -> None:
    if event_emitter is not None:
        await event_emitter(
            {"type": "status", "data": {"description": description, "done": done}}
        )


def _user_id(user: dict[str, Any] | None) -> str | None:
    user_id = (user or {}).get("id")
    return user_id if isinstance(user_id, str) and user_id else None


def _error(detail: str) -> dict[str, Any]:
    return {"error": {"detail": detail}}
