"""
title: Generic
author: langgraph-openai-serve
version: 0.9
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError
from openai.lib.streaming.chat import (
    AsyncChatCompletionStream,
    ChunkEvent,
    ContentDeltaEvent,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from pydantic import BaseModel, Field, field_validator

# These values mirror the public LGOS wire contract. This standalone Open WebUI
# Function must not import the server package:
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/models/schemas.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/chat/utils/interrupts.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/chat/utils/events.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/chat/schemas.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/client_settings.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/features.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/utils.py
INTERRUPT_TOOL_NAME = "langgraph_interrupt"
LGOS_EXTENSION_KEY = "langgraph_openai_serve"
CLIENT_EVENTS_FEATURE = "client_events"
OPENAI_METADATA_VALUE_MAX_LENGTH = 512
RUNTIME_SETTINGS_METADATA_KEY = "langgraph_runtime_settings"
STREAM_EVENTS_METADATA_KEY = "langgraph_stream_events"
STREAM_EVENTS_METADATA_VALUE = "v1"
THREAD_METADATA_KEY = "langgraph_thread_id"
NO_CHOICES_MESSAGE = "LangGraph API returned no choices."
LIMITED_FUNCTIONALITY_MESSAGE = (
    "Limited functionality: the configured OpenAI endpoint did not return valid "
    "langgraph_openai_serve model metadata. Runtime settings, client events, and "
    "interrupts may be unavailable. Configure the proxy to pass LGOS /v1 requests "
    "and responses through unchanged."
)
PipeChunk = str | dict[str, Any]


class SettingsTransportError(ValueError):
    """A Chat Variable value cannot be represented in OpenAI metadata."""


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = Field(
            default="http://bifrost:8080/openai_passthrough/v1",
            description="OpenAI base URL used for listing, retrieval, and chat.",
        )
        OPENAI_API_KEY: str = Field(
            default="DUMMY",
            description="Bearer token sent to the configured OpenAI endpoint.",
            json_schema_extra={"input": {"type": "password"}},
        )
        OPENAI_API_TIMEOUT: float = Field(
            default=30,
            gt=0,
            description="OpenAI client timeout in seconds.",
        )
        OPENAI_API_MODEL_ROUTES: dict[str, dict[str, str]] = Field(
            default_factory=lambda: {
                "lgos-a": {"x-model-provider": "lgos-a"},
                "lgos-b": {"x-model-provider": "lgos-b"},
            },
            description=(
                "Synthetic model prefixes and request headers for a multiplexed "
                "endpoint. Leave empty for a standard OpenAI endpoint."
            ),
        )

        @field_validator("OPENAI_API_MODEL_ROUTES")
        @classmethod
        def validate_model_routes(
            cls,
            value: dict[str, dict[str, str]],
        ) -> dict[str, dict[str, str]]:
            if any(not route or "/" in route for route in value):
                raise ValueError("Model routes must be non-empty and contain no '/'.")
            return value

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def pipes(self) -> list[dict[str, str]]:
        """Expose every registered LangGraph model in Open WebUI's selector."""
        async with _client(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            timeout=self.valves.OPENAI_API_TIMEOUT,
        ) as client:
            model_ids = await _list_model_ids(
                client,
                self.valves.OPENAI_API_MODEL_ROUTES,
            )
            pipes = []
            for model_id in model_ids:
                model = await _retrieve_model(
                    client,
                    model_id,
                    self.valves.OPENAI_API_MODEL_ROUTES,
                )
                suffix = (
                    ""
                    if _model_extension(model) is not None
                    else " (Limited functionality)"
                )
                pipes.append(
                    {
                        "id": model_id,
                        "name": f"Generic / {model_id}{suffix}",
                    }
                )

        return pipes

    async def pipe(
        self,
        body: dict[str, Any],
        __event_call__: Any = None,
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
    ) -> AsyncIterator[PipeChunk]:
        openwebui_metadata = __metadata__ or {}
        thread_id = _thread_id(openwebui_metadata)
        messages = cast(list[ChatCompletionMessageParam], body.get("messages") or [])
        forward_annotations = body.get("stream") is True
        base_url = self.valves.OPENAI_API_BASE_URL
        api_key = self.valves.OPENAI_API_KEY
        timeout = self.valves.OPENAI_API_TIMEOUT
        model_routes = self.valves.OPENAI_API_MODEL_ROUTES

        try:
            model_id = _model_id(body)
            _model_request(model_id, model_routes)
        except ValueError as exc:
            yield str(exc)
            return

        try:
            async with _client(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            ) as client:
                model = await _retrieve_model(client, model_id, model_routes)
                extension = _model_extension(model)
                if extension is None:
                    await _emit_limited_functionality_warning(__event_emitter__)
                runtime_metadata = _runtime_settings_metadata(
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
                    thread_id=thread_id,
                    model_id=model_id,
                    runtime_metadata=runtime_metadata,
                    include_client_events=include_client_events,
                    model_routes=model_routes,
                ) as stream:
                    async for delta in _content_deltas(stream, __event_emitter__):
                        yield delta
                    response = await stream.get_final_completion()

                if not response.choices:
                    yield NO_CHOICES_MESSAGE
                    return

                assistant_message = response.choices[0].message
                for chunk in _completion_chunks(
                    response,
                    forward_annotations=forward_annotations,
                ):
                    yield chunk

                tool_call = _interrupt_tool_call(assistant_message)
                if tool_call is None:
                    return

                decision, approval_error = await _approval_decision(
                    tool_call,
                    __event_call__,
                )
                if approval_error is not None:
                    yield approval_error
                    return

                assert decision is not None
                resume_messages = [
                    *messages,
                    ChatCompletionAssistantMessageParam(
                        role=assistant_message.role,
                        content=assistant_message.content,
                        tool_calls=[
                            cast(
                                ChatCompletionMessageToolCallParam,
                                tool_call.model_dump(mode="json"),
                            )
                        ],
                    ),
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=tool_call.id,
                        content=json.dumps({"resume": decision}),
                    ),
                ]
                async with _chat(
                    client=client,
                    messages=resume_messages,
                    thread_id=thread_id,
                    model_id=model_id,
                    runtime_metadata=runtime_metadata,
                    include_client_events=include_client_events,
                    model_routes=model_routes,
                ) as stream:
                    async for delta in _content_deltas(stream, __event_emitter__):
                        yield delta
                    response = await stream.get_final_completion()

                for chunk in _completion_chunks(
                    response,
                    forward_annotations=forward_annotations,
                ):
                    yield chunk
        except SettingsTransportError as exc:
            yield str(exc)
        except OpenAIError as exc:
            yield f"Error calling LangGraph API: {exc}"


##################### UTILITY FUNCTIONS ###################

#### Chat API ####


@asynccontextmanager
async def _chat(
    *,
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    thread_id: str,
    model_id: str,
    model_routes: dict[str, dict[str, str]],
    runtime_metadata: dict[str, str] | None = None,
    include_client_events: bool = False,
) -> AsyncIterator[AsyncChatCompletionStream[Any]]:
    metadata = {
        THREAD_METADATA_KEY: thread_id,
        **(runtime_metadata or {}),
    }
    if include_client_events:
        metadata[STREAM_EVENTS_METADATA_KEY] = STREAM_EVENTS_METADATA_VALUE

    async with client.chat.completions.stream(
        **_model_request(model_id, model_routes),
        messages=messages,
        metadata=metadata,
    ) as stream:
        yield stream


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
    )


def _model_id(body: dict[str, Any]) -> str:
    qualified_model_id = body.get("model")
    if not isinstance(qualified_model_id, str):
        raise ValueError("Open WebUI did not provide a valid model ID.")

    _, separator, model_id = qualified_model_id.partition(".")
    if not separator or not model_id:
        raise ValueError("Open WebUI did not provide a valid model ID.")

    return model_id


def _thread_id(metadata: dict[str, Any]) -> str:
    value = metadata.get("chat_id") or metadata.get("session_id") or "default"
    return f"openwebui:function:{value}"


async def _list_model_ids(
    client: AsyncOpenAI,
    model_routes: dict[str, dict[str, str]],
) -> list[str]:
    if not model_routes:
        models = await client.models.list()
        return [model.id for model in models.data]

    model_ids = []
    for route, headers in model_routes.items():
        models = await client.models.list(extra_headers=headers)
        model_ids.extend(f"{route}/{model.id}" for model in models.data)
    return model_ids


def _model_request(
    model_id: str,
    model_routes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if not model_routes:
        return {"model": model_id}

    route, separator, upstream_model = model_id.partition("/")
    headers = model_routes.get(route)
    if not separator or not upstream_model or headers is None:
        raise ValueError(f"Unknown configured OpenAI model route in {model_id!r}.")

    request: dict[str, Any] = {"model": upstream_model}
    if headers:
        request["extra_headers"] = headers
    return request


#### Chat Settings ####


def _runtime_settings_metadata(
    *,
    model: Any,
    metadata: dict[str, Any],
) -> dict[str, str]:
    values = metadata.get("chat_variables")
    if not isinstance(values, dict) or not values:
        return {}

    defaults = _runtime_settings_defaults(model)
    if defaults is None:
        return {}

    changed = {}
    for name, default in defaults.items():
        if name not in values:
            continue
        value = values[name]
        if type(value) is type(default) and value == default:
            continue
        changed[name] = value

    if not changed:
        return {}
    try:
        encoded = json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SettingsTransportError(
            "The selected runtime settings cannot be encoded as JSON."
        ) from exc
    if len(encoded) > OPENAI_METADATA_VALUE_MAX_LENGTH:
        raise SettingsTransportError(
            "The selected runtime settings exceed the OpenAI metadata value limit."
        )
    return {RUNTIME_SETTINGS_METADATA_KEY: encoded}


def _runtime_settings_defaults(model: Any) -> dict[str, Any] | None:
    extension = _model_extension(model)
    if extension is None:
        return None
    settings = extension.get("client_settings")
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        return None
    defaults = settings.get("defaults")
    return defaults if isinstance(defaults, dict) else None


async def _retrieve_model(
    client: AsyncOpenAI,
    model_id: str,
    model_routes: dict[str, dict[str, str]],
) -> Any:
    """Return model details, or None when retrieval through this endpoint fails."""
    try:
        return await client.models.retrieve(**_model_request(model_id, model_routes))
    except OpenAIError:
        return None


def _model_extension(model: Any) -> dict[str, Any] | None:
    extension = (getattr(model, "model_extra", None) or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict) or extension.get("schema_version") != 1:
        return None
    features = extension.get("features")
    if not isinstance(features, list) or any(
        not isinstance(feature, str) for feature in features
    ):
        return None
    return extension


def _extension_supports(extension: dict[str, Any] | None, feature: str) -> bool:
    return extension is not None and feature in extension["features"]


async def _emit_limited_functionality_warning(event_emitter: Any) -> None:
    if event_emitter is None:
        return
    await event_emitter(
        {
            "type": "notification",
            "data": {
                "type": "warning",
                "content": LIMITED_FUNCTIONALITY_MESSAGE,
            },
        }
    )


#### Streaming ####


def _completion_chunks(
    response: ChatCompletion,
    *,
    forward_annotations: bool,
) -> list[PipeChunk]:
    """Return completion-level chunks that follow streamed text."""
    if not response.choices:
        return [NO_CHOICES_MESSAGE]
    annotations = response.choices[0].message.annotations
    if not forward_annotations or not annotations:
        return []

    return [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "annotations": [
                            annotation.model_dump(mode="json")
                            for annotation in annotations
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    ]


async def _content_deltas(
    stream: AsyncChatCompletionStream[Any],
    event_emitter: Any = None,
) -> AsyncIterator[str]:
    """Yield text and emit portable status updates."""
    async for event in stream:
        if isinstance(event, ContentDeltaEvent):
            yield event.delta
        elif isinstance(event, ChunkEvent) and event_emitter is not None:
            status = _status_event(event.chunk)
            if status is not None:
                await event_emitter(status)


def _status_event(
    chunk: ChatCompletionChunk,
) -> dict[str, Any] | None:
    extension = (chunk.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict) or extension.get("schema_version") != 1:
        return None

    event = extension.get("event")
    if not isinstance(event, dict) or event.get("type") != "status":
        return None
    data = event.get("data")
    if not isinstance(data, dict):
        return None

    description = data.get("description")
    done = data.get("done", False)
    hidden = data.get("hidden", False)
    if (
        not isinstance(description, str)
        or not description
        or not isinstance(done, bool)
        or not isinstance(hidden, bool)
    ):
        return None

    return {
        "type": "status",
        "data": {
            "description": description,
            "done": done,
            "hidden": hidden,
        },
    }


#### HITL ####


async def _approval_decision(
    tool_call: ChatCompletionMessageToolCall,
    event_call: Any,
) -> tuple[str | None, str | None]:
    if event_call is None:
        return None, "Open WebUI approval modal is unavailable for this request."

    approval = await event_call(_approval_event(tool_call))
    if isinstance(approval, dict) and approval.get("error"):
        return None, f"Open WebUI approval failed: {approval['error']}"

    return ("approve" if approval is True else "reject"), None


def _interrupt_tool_call(
    message: ChatCompletionMessage,
) -> ChatCompletionMessageToolCall | None:
    for tool_call in message.tool_calls or []:
        if (
            isinstance(tool_call, ChatCompletionMessageToolCall)
            and tool_call.function.name == INTERRUPT_TOOL_NAME
        ):
            return tool_call
    return None


def _approval_event(
    tool_call: ChatCompletionMessageToolCall,
) -> dict[str, Any]:
    payload = _interrupt_payload(tool_call) or {}
    question = str(payload.get("question") or "Approve this agent action?")
    request = str(payload.get("request") or json.dumps(payload, indent=2))
    return {
        "type": "confirmation",
        "data": {
            "title": question,
            "message": request,
        },
    }


def _interrupt_payload(
    tool_call: ChatCompletionMessageToolCall,
) -> dict[str, object] | None:
    try:
        arguments = json.loads(tool_call.function.arguments)
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(arguments, dict):
        return None

    payload = arguments.get("payload")
    return payload if isinstance(payload, dict) else None
