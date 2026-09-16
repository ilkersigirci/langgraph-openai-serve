"""Open WebUI manifold Pipe backed exclusively by the Responses API."""

import os
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, cast

from openai import OpenAIError
from openai.types.responses import Response, ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field

from .api import (
    _client,
    _list_model_ids,
    _model_request,
)
from .contracts import (
    DISPLAY_FILE_TOOL_NAME,
    INTERRUPT_CANCELLED_MESSAGE,
    INTERRUPT_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    InterruptCancelled,
    OpenWebUIEventEmitter,
    OpenWebUIInvocation,
    PipeChunk,
    PipeResponse,
    is_server_tool_model,
    supports_web_search,
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
    _openwebui_mcp_tools,
    _openwebui_text_chunk,
    _openwebui_tool_chunk,
    _raise_for_response,
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


@dataclass(frozen=True, slots=True)
class PreparedResponsesRequest:
    """Validated upstream request state retained across client-tool turns."""

    model_id: str
    streaming: bool
    gateway: GatewayConfig
    openwebui_mcp_names: dict[str, str]
    replay_input: list[dict[str, Any]]
    request: dict[str, Any]


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
            description="API key used for Responses, Files, and MCP.",
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
        __tools__: dict[str, Any] | None = None,
    ) -> PipeResponse:
        """Run the selected graph through OpenAI Responses."""
        streaming = isinstance(body, Mapping) and body.get("stream") is True
        try:
            invocation = OpenWebUIInvocation.from_host(
                body=body,
                metadata=__metadata__,
                user=__user__,
                files=__files__,
                tools=__tools__,
            )
            if __event_emitter__ is not None and not callable(__event_emitter__):
                raise ValueError("Open WebUI provided an invalid event emitter.")
        except ValueError as exc:
            failure = _error(f"Responses request failed: {exc}")
            return _single_chunk(failure) if streaming else failure

        results = self._run(
            invocation,
            event_emitter=cast("OpenWebUIEventEmitter | None", __event_emitter__),
            host_request=__request__,
        )
        if invocation.body.stream:
            return results
        async with aclosing(results):
            return await anext(results)

    async def _run(
        self,
        invocation: OpenWebUIInvocation,
        *,
        event_emitter: OpenWebUIEventEmitter | None,
        host_request: object | None,
    ) -> AsyncGenerator[PipeChunk, None]:
        """Own one Responses/tool loop for both Pipe response modes."""
        answer_parts: list[str] = []
        latest_status = ""
        finished = False
        try:
            prepared = await self._prepare_request(
                invocation,
                host_request=host_request,
            )
            async with _client(
                base_url=prepared.gateway.responses_base_url,
                api_key=self.valves.OPENAI_GATEWAY_API_KEY,
                timeout=self.valves.OPENAI_API_TIMEOUT,
            ) as client:
                while True:
                    # Execute one SDK-owned Responses turn. Text deltas remain
                    # live while the SDK assembles the typed final Response.
                    final_text_streamed = False
                    phases: dict[int, str | None] = {}
                    if prepared.streaming:
                        async with client.responses.stream(
                            **prepared.request
                        ) as stream:
                            async for event in stream:
                                if event.type == "response.output_item.added":
                                    if event.item.type == "message":
                                        phases[event.output_index] = event.item.phase
                                elif (
                                    event.type == "response.output_text.delta"
                                    or event.type == "response.refusal.delta"
                                ) and phases.get(event.output_index) != "commentary":
                                    final_text_streamed = True
                                    yield _openwebui_text_chunk(
                                        prepared.model_id, event.delta
                                    )
                                elif (
                                    event.type == "response.incomplete"
                                    or event.type == "response.failed"
                                ):
                                    _raise_for_response(event.response)
                                elif (
                                    event.type == "response.output_text.done"
                                    and phases.get(event.output_index) == "commentary"
                                    and event.text
                                ):
                                    latest_status = event.text
                                    await _emit_status(
                                        event_emitter, latest_status, done=False
                                    )
                            response = cast(
                                "Response", await stream.get_final_response()
                            )
                    else:
                        response = await client.responses.create(**prepared.request)

                    # Classify the typed terminal result before selecting one
                    # Open WebUI rendering path.
                    _raise_for_response(response)
                    await _emit_response_sources(response, event_emitter)
                    final_text = _responses_final_text(response)
                    if prepared.streaming and not final_text_streamed and final_text:
                        yield _openwebui_text_chunk(prepared.model_id, final_text)
                    answer_parts.append(final_text)
                    calls = _responses_function_calls(response)
                    if not calls:
                        finished = True
                        if not prepared.streaming:
                            yield "".join(answer_parts)
                        return
                    if _all_calls(calls, INTERRUPT_TOOL_NAME):
                        finished = True
                        yield (
                            _openwebui_interrupt_chunk(
                                prepared.model_id, response.id, calls
                            )
                            if prepared.streaming
                            else _openwebui_interrupt_completion(
                                prepared.model_id,
                                response.id,
                                calls,
                                content="".join(answer_parts),
                            )
                        )
                        return
                    if prepared.openwebui_mcp_names and all(
                        call.name in prepared.openwebui_mcp_names for call in calls
                    ):
                        finished = True
                        yield _openwebui_tool_chunk(
                            prepared.model_id,
                            calls,
                            prepared.openwebui_mcp_names,
                        )
                        return
                    if not _all_calls(calls, DISPLAY_FILE_TOOL_NAME):
                        raise ValueError(
                            "LangGraph API returned an unsupported or mixed function-call batch."
                        )
                    outputs = [
                        await _handle_display_file(
                            call,
                            event_emitter,
                            host_request,
                            files_base_url=prepared.gateway.files_base_url,
                            api_key=self.valves.OPENAI_GATEWAY_API_KEY,
                            timeout=self.valves.OPENAI_API_TIMEOUT,
                            provider=prepared.gateway.files_provider,
                        )
                        for call in calls
                    ]
                    if prepared.request.pop("previous_response_id", None) is not None:
                        # Interrupt answers belong only to the paused checkpoint.
                        # Client tools continue from the UI's transcript instead.
                        prepared.request["input"] = list(prepared.replay_input)
                    prepared.request["input"].extend(
                        _responses_continuation(response, outputs)
                    )
        except InterruptCancelled:
            yield INTERRUPT_CANCELLED_MESSAGE
        except (ValueError, RuntimeError, OpenAIError) as exc:
            yield _error(f"Responses request failed: {exc}")
        finally:
            if latest_status:
                await _emit_status(
                    event_emitter,
                    latest_status if finished else f"Stopped: {latest_status}",
                    done=True,
                )

    async def _prepare_request(
        self,
        invocation: OpenWebUIInvocation,
        *,
        host_request: object | None,
    ) -> PreparedResponsesRequest:
        gateway = self._gateway()
        model_id = invocation.body.model_id
        _model_request(
            model_id,
            provider_routing=gateway.provider_routing,
        )
        mcp_tools, openwebui_mcp_names = _openwebui_mcp_tools(invocation.mcp_tools)
        # Open WebUI v0.11.3 enters its native tool loop only for streams.
        if mcp_tools and not invocation.body.stream:
            raise ValueError("Open WebUI MCP tool execution requires streaming.")
        transcript_mcp_names = {
            openwebui_name: gateway_name
            for gateway_name, openwebui_name in openwebui_mcp_names.items()
        }
        replay_input = _responses_input(
            invocation.body.messages,
            mcp_tool_names=transcript_mcp_names,
        )
        if resume := _ask_user_to_resume(invocation.body.messages):
            input_items, previous_response_id = resume
        else:
            messages = await _with_response_file_parts(
                invocation.body.messages,
                invocation.files,
                invocation.metadata,
                host_request,
                base_url=gateway.files_base_url,
                api_key=self.valves.OPENAI_GATEWAY_API_KEY,
                timeout=self.valves.OPENAI_API_TIMEOUT,
                provider=gateway.files_provider,
            )
            input_items = _responses_input(
                messages,
                mcp_tool_names=transcript_mcp_names,
            )
            previous_response_id = None

        tools = _responses_tools(model_id, invocation.metadata)
        tools.extend(mcp_tools)
        return PreparedResponsesRequest(
            model_id=model_id,
            streaming=invocation.body.stream,
            gateway=gateway,
            openwebui_mcp_names=openwebui_mcp_names,
            replay_input=replay_input,
            request=_responses_request(
                model_id,
                input_items,
                _request_metadata(
                    invocation.metadata,
                    include_runtime_settings=not is_server_tool_model(model_id),
                    excluded_runtime_settings=(
                        {WEB_SEARCH_TOOL_NAME} if supports_web_search(model_id) else ()
                    ),
                ),
                invocation.user.id or None,
                provider_routing=gateway.provider_routing,
                tools=tools,
                previous_response_id=previous_response_id,
            ),
        )

    def _gateway(self) -> GatewayConfig:
        return gateway_config(
            self.valves.OPENAI_GATEWAY_TYPE,
            self.valves.OPENAI_GATEWAY_BASE_URL,
        )


def _all_calls(calls: list[ResponseFunctionToolCall], name: str) -> bool:
    return bool(calls) and all(call.name == name for call in calls)


async def _emit_status(
    event_emitter: OpenWebUIEventEmitter | None,
    description: str,
    *,
    done: bool,
) -> None:
    if event_emitter is not None:
        await event_emitter(
            {"type": "status", "data": {"description": description, "done": done}}
        )


def _error(detail: str) -> dict[str, Any]:
    return {"error": {"detail": detail}}


async def _single_chunk(chunk: PipeChunk) -> AsyncGenerator[PipeChunk, None]:
    yield chunk
