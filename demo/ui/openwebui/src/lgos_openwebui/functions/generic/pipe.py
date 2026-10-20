"""Open WebUI manifold Pipe backed exclusively by the Responses API."""

from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, cast

from openai import OpenAIError
from openai.types.responses import Response, ResponseFunctionToolCall
from pydantic import BaseModel, Field

from .api import _client
from .catalog import (
    USERVALVES_MODEL_ID,
    OpenWebUIModel,
    openwebui_models,
)
from .contracts import (
    BACKGROUND_SETTING_NAME,
    DISPLAY_FILE_TOOL_NAME,
    INTERRUPT_CANCELLED_MESSAGE,
    INTERRUPT_TOOL_NAME,
    InterruptCancelled,
    OpenWebUIBody,
    OpenWebUIEventEmitter,
    OpenWebUIMetadata,
    OpenWebUIRequest,
    PipeChunk,
    PipeResponse,
)
from .files import _handle_display_file, _with_response_file_parts
from .gateway import (
    GatewayConfig,
    GatewayRoot,
    GatewayType,
    gateway_config,
    gateway_connection,
    gateway_models,
)
from .interrupts import (
    _ask_user_to_resume,
    _openwebui_interrupt_chunks,
    _openwebui_interrupt_completion,
)
from .metadata import _request_metadata
from .responses import (
    _background_response,
    _emit_response_sources,
    _openwebui_chunk,
    _openwebui_mcp_tools,
    _openwebui_tool_chunk,
    _raise_for_response,
    _responses_continuation,
    _responses_final_text,
    _responses_function_calls,
    _responses_input,
    _responses_request,
    _responses_tools,
)


@dataclass(frozen=True, slots=True)
class PreparedResponsesRequest:
    """Validated upstream request state retained across client-tool turns."""

    model_id: str
    background: bool
    streaming: bool
    gateway: GatewayConfig
    api_key: str
    openwebui_mcp_names: dict[str, str]
    replay_input: list[dict[str, Any]]
    request: dict[str, Any]


class Pipe:
    class Valves(BaseModel):
        """Tune the Pipe's gateway requests."""

        OPENAI_GATEWAY_TYPE: GatewayType = Field(
            default="litellm",
            description="Gateway implementation: litellm or bifrost.",
        )
        OPENAI_GATEWAY_BASE_URL: GatewayRoot = Field(
            default="",
            description=(
                "Gateway root URL reachable from Open WebUI. The Pipe sends what "
                "Open WebUI's OpenAI connection to its /v1 sends."
            ),
        )
        MODEL_PREFIX: str = Field(
            default="lgos",
            pattern=r"^[^\s/*]+$",
            description="Gateway model namespace served by this Function.",
        )
        MCP_GATEWAY_CONNECTION_ID: str = Field(
            default="lgos-gateway",
            min_length=1,
            description="Open WebUI MCP connection used for gateway tools.",
        )
        USERVALVES_GRAPH_MODEL_ID: str = Field(
            default="",
            description="Optional graph model used by the UserValves demo Filter.",
        )
        OPENAI_API_TIMEOUT: float = Field(
            default=30,
            gt=0,
            description="OpenAI-compatible request timeout in seconds.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def pipes(
        self, __oauth_token__: Mapping[str, Any] | None = None
    ) -> list[OpenWebUIModel]:
        """List the user's gateway graphs with the model info Open WebUI shows."""
        settings = self.valves
        gateway = _gateway(settings)
        async with _client(
            base_url=f"{gateway.root_url}/v1",
            api_key=await _user_credential(gateway, __oauth_token__),
            timeout=self.valves.OPENAI_API_TIMEOUT,
        ) as client:
            models = await gateway_models(client, gateway, settings.MODEL_PREFIX)
        return openwebui_models(
            models,
            mcp_gateway_connection_id=settings.MCP_GATEWAY_CONNECTION_ID,
            uservalves_graph_model_id=settings.USERVALVES_GRAPH_MODEL_ID or None,
        )

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: OpenWebUIEventEmitter | None = None,
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __request__: OpenWebUIRequest | None = None,
        __tools__: dict[str, dict[str, Any]] | None = None,
        __oauth_token__: Mapping[str, Any] | None = None,
    ) -> PipeResponse:
        """Run the selected graph through OpenAI Responses."""
        # Open WebUI selects its response mode with the same truthiness test.
        streaming = bool(body.get("stream"))
        results = self._run(
            body,
            __metadata__ or {},
            user_id=_gateway_user_id(__user__ or {}),
            tools=__tools__ or {},
            streaming=streaming,
            event_emitter=__event_emitter__,
            host_request=__request__,
            oauth_token=__oauth_token__,
        )
        if streaming:
            return results
        async with aclosing(results):
            return await anext(results)

    async def _run(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements] - One lifecycle owns streaming, client tools, and terminal cleanup.
        self,
        body: dict[str, Any],
        metadata: dict[str, Any],
        *,
        user_id: str | None,
        tools: Mapping[str, Mapping[str, Any]],
        streaming: bool,
        event_emitter: OpenWebUIEventEmitter | None,
        host_request: OpenWebUIRequest | None,
        oauth_token: Mapping[str, Any] | None,
    ) -> AsyncGenerator[PipeChunk, None]:
        """
        Own one Responses/tool loop for both Pipe response modes.

        Yields:
            Streaming chunks or the completed non-streaming answer.

        """
        answer_parts: list[str] = []
        latest_status = ""
        finished = False

        async def publish_background_status(status: str) -> None:
            nonlocal latest_status
            latest_status = f"Background response {status.replace('_', ' ')}."
            await _emit_status(event_emitter, latest_status, done=False)

        try:  # ruff: ignore[too-many-nested-blocks, too-many-statements-in-try-clause] - The whole turn shares cancellation, UI error reporting, and status cleanup.
            prepared = await self._prepare_request(
                OpenWebUIBody.model_validate(body),
                OpenWebUIMetadata.model_validate(metadata),
                user_id=user_id,
                tools=tools,
                streaming=streaming,
                host_request=host_request,
                oauth_token=oauth_token,
            )
            async with _client(
                base_url=prepared.gateway.responses_base_url,
                api_key=prepared.api_key,
                timeout=self.valves.OPENAI_API_TIMEOUT,
            ) as client:
                while True:
                    # Execute one SDK-owned Responses turn. Text deltas remain
                    # live while the SDK assembles the typed final Response.
                    final_text_streamed = False
                    phases: dict[int, str | None] = {}
                    if prepared.background:
                        response = await _background_response(
                            client,
                            prepared.request,
                            publish_background_status,
                            provider_routing=prepared.gateway.provider_routing,
                        )
                        status = response.status or "unknown"
                        latest_status = (
                            f"Background response {status.replace('_', ' ')}."
                        )
                    elif prepared.streaming:
                        async with client.responses.stream(
                            **prepared.request
                        ) as stream:
                            async for event in stream:
                                if event.type == "response.output_item.added":
                                    if event.item.type == "message":
                                        phases[event.output_index] = event.item.phase
                                elif (
                                    event.type == "response.output_text.delta"  # ruff: ignore[repeated-equality-comparison] - Explicit comparisons preserve discriminated-union narrowing in ty.
                                    or event.type == "response.refusal.delta"
                                ) and phases.get(event.output_index) != "commentary":
                                    final_text_streamed = True
                                    yield _openwebui_chunk(
                                        prepared.model_id, {"content": event.delta}
                                    )
                                elif (
                                    event.type == "response.incomplete"  # ruff: ignore[repeated-equality-comparison] - Explicit comparisons preserve discriminated-union narrowing in ty.
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
                        yield _openwebui_chunk(
                            prepared.model_id, {"content": final_text}
                        )
                    answer_parts.append(final_text)
                    calls = _responses_function_calls(response)
                    if not calls:
                        finished = True
                        if not prepared.streaming:
                            yield "".join(answer_parts)
                        return
                    if _all_calls(calls, INTERRUPT_TOOL_NAME):
                        finished = True
                        if not prepared.streaming:
                            yield _openwebui_interrupt_completion(
                                prepared.model_id,
                                response.id,
                                calls,
                                content="".join(answer_parts),
                            )
                            return
                        for chunk in _openwebui_interrupt_chunks(
                            prepared.model_id,
                            response.id,
                            calls,
                            after_text=any(answer_parts),
                        ):
                            yield chunk
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
                        msg = "LangGraph API returned an unsupported or mixed function-call batch."
                        raise ValueError(msg)  # ruff: ignore[raise-within-try] - Local validation uses the same error reporting as remote failures.
                    outputs = [
                        await _handle_display_file(
                            call,
                            event_emitter,
                            host_request,
                            files_base_url=prepared.gateway.files_base_url,
                            api_key=prepared.api_key,
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
        body: OpenWebUIBody,
        metadata: OpenWebUIMetadata,
        *,
        user_id: str | None,
        tools: Mapping[str, Mapping[str, Any]],
        streaming: bool,
        host_request: OpenWebUIRequest | None,
        oauth_token: Mapping[str, Any] | None,
    ) -> PreparedResponsesRequest:
        settings = self.valves
        gateway = _gateway(settings)
        api_key = await _user_credential(gateway, oauth_token)
        model_id = body.model_id
        if body.model_id == USERVALVES_MODEL_ID and settings.USERVALVES_GRAPH_MODEL_ID:
            model_id = settings.USERVALVES_GRAPH_MODEL_ID
        mcp_tools, openwebui_mcp_names = _openwebui_mcp_tools(
            tools, settings.MCP_GATEWAY_CONNECTION_ID
        )
        # Open WebUI enters its native tool loop only for streams.
        if mcp_tools and not streaming:
            msg = "Open WebUI MCP tool execution requires streaming."
            raise ValueError(msg)
        transcript_mcp_names = {
            openwebui_name: gateway_name
            for gateway_name, openwebui_name in openwebui_mcp_names.items()
        }
        replay_input = _responses_input(
            body.messages,
            mcp_tool_names=transcript_mcp_names,
        )
        if resume := _ask_user_to_resume(body.messages):
            input_items, previous_response_id = resume
        else:
            messages = await _with_response_file_parts(
                body.messages,
                metadata,
                host_request,
                base_url=gateway.files_base_url,
                api_key=api_key,
                timeout=self.valves.OPENAI_API_TIMEOUT,
                provider=gateway.files_provider,
            )
            input_items = _responses_input(
                messages,
                mcp_tool_names=transcript_mcp_names,
            )
            previous_response_id = None

        background = metadata.chat_variables.get(BACKGROUND_SETTING_NAME) is True
        return PreparedResponsesRequest(
            model_id=model_id,
            background=background,
            streaming=streaming,
            gateway=gateway,
            api_key=api_key,
            openwebui_mcp_names=openwebui_mcp_names,
            replay_input=replay_input,
            request=_responses_request(
                model_id,
                input_items,
                _request_metadata(metadata),
                user_id,
                background=background,
                tools=[
                    *_responses_tools(metadata),
                    *mcp_tools,
                ],
                previous_response_id=previous_response_id,
            ),
        )


def _gateway(valves: Pipe.Valves) -> GatewayConfig:
    # Open WebUI constructs the Pipe before assigning its stored valves. Only
    # the initial empty URL needs a runtime check; Valves validates saved input.
    if not valves.OPENAI_GATEWAY_BASE_URL:
        msg = "Set OPENAI_GATEWAY_BASE_URL in the Generic Function valves."
        raise RuntimeError(msg)
    return gateway_config(valves.OPENAI_GATEWAY_TYPE, valves.OPENAI_GATEWAY_BASE_URL)


def _gateway_user_id(user: Mapping[str, Any]) -> str | None:
    """
    Name the user the gateway bills.

    Under SSO the gateway verifies the access token's OIDC subject, which
    differs from Open WebUI's own user ID; password users have no subject.
    """
    oidc = (user.get("oauth") or {}).get("oidc") or {}
    return oidc.get("sub") or user.get("id") or None


async def _user_credential(
    gateway: GatewayConfig,
    oauth_token: Mapping[str, Any] | None,
) -> str:
    """Send what Open WebUI's own connection to the gateway sends."""
    auth_type, key = await gateway_connection(gateway.root_url)
    if auth_type == "bearer" and key:
        return key
    if auth_type != "system_oauth":
        msg = (
            "Set bearer with an API key, or system_oauth, on Open WebUI's "
            "connection to the gateway."
        )
        raise RuntimeError(msg)
    # Falling back to the shared key would bypass the user's gateway permissions.
    access_token = (oauth_token or {}).get("access_token")
    if not isinstance(access_token, str) or not access_token:
        msg = "Sign in again with SSO; Open WebUI has no access token for the gateway."
        raise RuntimeError(msg)
    return access_token


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
