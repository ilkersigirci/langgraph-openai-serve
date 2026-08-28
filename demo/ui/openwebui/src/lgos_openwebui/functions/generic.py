"""
title: Generic

author: langgraph-openai-serve
version: 0.16
"""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from html import escape
from typing import Any, Literal, cast

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
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
SESSION_ID_METADATA_KEY = "session_id"
RUNTIME_SETTINGS_METADATA_KEY = "langgraph_runtime_settings"
STREAM_EVENTS_METADATA_KEY = "langgraph_stream_events"
STREAM_EVENTS_METADATA_VALUE = "v1"
LGOS_MODEL_OWNER = "langgraph-openai-serve"
NO_CHOICES_MESSAGE = "LangGraph API returned no choices."
LIMITED_FUNCTIONALITY_MESSAGE = (
    "Limited functionality: the configured OpenAI endpoint did not return valid "
    "langgraph_openai_serve model metadata. Runtime settings, client events, and "
    "interrupts may be unavailable. Configure the proxy to pass LGOS /v1 requests "
    "and responses through unchanged."
)
PipeChunk = str | dict[str, Any]
PipeResponse = AsyncIterator[PipeChunk] | dict[str, Any] | str


class SettingsTransportError(ValueError):
    """A Chat Variable value cannot be represented in OpenAI metadata."""


class PlotlyArtifact(BaseModel):
    """The supported LGOS Plotly artifact payload."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1]
    id: str = Field(min_length=1)
    kind: Literal["plotly"]
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    figure: dict[str, Any]


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
        __event_call__: Any = None,
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
    ) -> PipeResponse:
        """Use the same Chat Completions mode requested by Open WebUI."""
        if body.get("stream") is True:
            return self._stream(
                body,
                __event_call__=__event_call__,
                __event_emitter__=__event_emitter__,
                __metadata__=__metadata__,
                __user__=__user__,
            )
        return await self._complete(
            body,
            __event_call__=__event_call__,
            __event_emitter__=__event_emitter__,
            __metadata__=__metadata__,
            __user__=__user__,
        )

    async def _stream(
        self,
        body: dict[str, Any],
        __event_call__: Any = None,
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
    ) -> AsyncIterator[PipeChunk]:
        """
        Forward chat and complete each interrupt batch in one invocation.

        Open WebUI supplies the input ledger but this Function does not assume
        an undocumented API for persisting raw assistant tool calls. The exact
        assistant/tool exchange therefore remains local; cancellation leaves
        the durable run paused and sends no partial resume.

        Yields:
            PipeChunk objects for Open WebUI stream.
        """
        openwebui_metadata = __metadata__ or {}
        messages = cast(list[ChatCompletionMessageParam], body.get("messages") or [])
        base_url = self.valves.OPENAI_API_BASE_URL
        api_key = self.valves.OPENAI_API_KEY
        timeout = self.valves.OPENAI_API_TIMEOUT

        try:
            model_id = _model_id(body)
            _model_request(model_id)
        except ValueError as exc:
            yield str(exc)
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
                while True:
                    async with _chat(
                        client=client,
                        messages=messages,
                        model_id=model_id,
                        request_metadata=request_metadata,
                        include_client_events=include_client_events,
                        user_id=_user_id(__user__),
                    ) as stream:
                        async for delta in _content_deltas(
                            stream,
                            __event_emitter__,
                        ):
                            yield delta
                        response = await stream.get_final_completion()

                    if not response.choices:
                        yield NO_CHOICES_MESSAGE
                        return

                    for chunk in _completion_chunks(
                        response,
                    ):
                        yield chunk

                    resume_messages, error = await _resume_messages(
                        response,
                        __event_call__,
                    )
                    if error is not None:
                        yield error
                        return
                    if resume_messages is None:
                        return
                    messages = resume_messages
        except SettingsTransportError as exc:
            yield str(exc)
        except OpenAIError as exc:
            yield f"Error calling LangGraph API: {exc}"

    async def _complete(
        self,
        body: dict[str, Any],
        __event_call__: Any,
        __event_emitter__: Any,
        __metadata__: dict[str, Any] | None,
        __user__: dict[str, Any] | None,
    ) -> dict[str, Any] | str:
        """Return one complete OpenAI response without replaying stream events."""
        messages = cast(list[ChatCompletionMessageParam], body.get("messages") or [])
        try:
            model_id = _model_id(body)
            _model_request(model_id)
        except ValueError as exc:
            return str(exc)

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
                while True:
                    response = await _chat_completion(
                        client=client,
                        messages=messages,
                        model_id=model_id,
                        request_metadata=request_metadata,
                        user_id=_user_id(__user__),
                    )
                    if not response.choices:
                        return NO_CHOICES_MESSAGE

                    resume_messages, error = await _resume_messages(
                        response,
                        __event_call__,
                    )
                    if error is not None:
                        return error
                    if resume_messages is None:
                        await _emit_sources(response, __event_emitter__)
                        return response.model_dump(mode="json", exclude_none=True)
                    messages = resume_messages
        except SettingsTransportError as exc:
            return str(exc)
        except OpenAIError as exc:
            return f"Error calling LangGraph API: {exc}"


##################### UTILITY FUNCTIONS ###################

#### Chat API ####


@asynccontextmanager
async def _chat(
    *,
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    model_id: str,
    request_metadata: dict[str, str] | None = None,
    include_client_events: bool = False,
    user_id: str | None = None,
) -> AsyncIterator[AsyncChatCompletionStream[Any]]:
    metadata = dict(request_metadata or {})
    if include_client_events:
        metadata[STREAM_EVENTS_METADATA_KEY] = STREAM_EVENTS_METADATA_VALUE

    request = _chat_request(model_id, messages, metadata, user_id)

    async with client.chat.completions.stream(**request) as stream:
        yield stream


async def _chat_completion(
    *,
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    model_id: str,
    request_metadata: dict[str, str] | None = None,
    user_id: str | None = None,
) -> ChatCompletion:
    """Create one non-streaming Chat Completion."""
    request = _chat_request(model_id, messages, request_metadata, user_id)
    return await client.chat.completions.create(**request)


def _chat_request(
    model_id: str,
    messages: list[ChatCompletionMessageParam],
    metadata: dict[str, str] | None,
    user_id: str | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {**_model_request(model_id), "messages": messages}
    if user_id is not None:
        request["user"] = user_id
    if metadata:
        request["metadata"] = metadata
    return request


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
        raise TypeError(msg)

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


#### Chat Settings ####


def _request_metadata(
    *,
    model: Any,
    metadata: dict[str, Any],
) -> dict[str, str]:
    request_metadata = _runtime_settings_metadata(model=model, metadata=metadata)
    chat_id = metadata.get("chat_id")
    if isinstance(chat_id, str) and chat_id:
        request_metadata[SESSION_ID_METADATA_KEY] = chat_id
    return request_metadata


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
        msg = "The selected runtime settings cannot be encoded as JSON."
        raise SettingsTransportError(msg) from exc
    if len(encoded) > OPENAI_METADATA_VALUE_MAX_LENGTH:
        msg = "The selected runtime settings exceed the OpenAI metadata value limit."
        raise SettingsTransportError(msg)
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
) -> Any:
    """Return model details, or None when retrieval through this endpoint fails."""
    try:
        return await client.models.retrieve(**_model_request(model_id))
    except OpenAIError:
        return None


def _model_extension(model: Any) -> dict[str, Any] | None:
    extension = (getattr(model, "model_extra", None) or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict) or extension.get("schema_version") != 1:
        return None
    description = extension.get("description")
    features = extension.get("features")
    if (
        not isinstance(features, list)
        or any(not isinstance(feature, str) for feature in features)
        or not isinstance(description, str)
        or not description.strip()
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
) -> list[PipeChunk]:
    """Return completion-level chunks that follow streamed text."""
    if not response.choices:
        return [NO_CHOICES_MESSAGE]
    annotations = response.choices[0].message.annotations
    if not annotations:
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


async def _emit_sources(response: ChatCompletion, event_emitter: Any) -> None:
    """Emit final annotations in Open WebUI's native source format."""
    if event_emitter is None or not response.choices:
        return
    message = response.choices[0].message
    if not isinstance(message.content, str):
        return

    for annotation in message.annotations or []:
        citation = annotation.url_citation
        start = citation.start_index
        stop = citation.end_index + 1
        if not 0 <= start < stop <= len(message.content):
            continue
        await event_emitter(
            {
                "type": "source",
                "data": {
                    "source": {"name": citation.title, "url": citation.url},
                    "document": [message.content[start:stop]],
                    "metadata": [
                        {
                            "source": citation.url,
                            "name": citation.title,
                            "url": citation.url,
                        }
                    ],
                },
            }
        )


async def _resume_messages(
    response: ChatCompletion,
    event_call: Any,
) -> tuple[list[ChatCompletionMessageParam] | None, str | None]:
    """Build the next interrupt exchange, if the completion requested one."""
    if not response.choices:
        return None, NO_CHOICES_MESSAGE

    assistant_message = response.choices[0].message
    tool_calls = _interrupt_tool_calls(assistant_message)
    if tool_calls is None:
        return None, "Open WebUI received an unsupported tool-call batch."
    if not tool_calls:
        return None, None

    decisions = []
    for tool_call in tool_calls:
        decision, error = await _approval_decision(tool_call, event_call)
        if error is not None:
            return None, error
        if decision is None:
            msg = "Decision cannot be None"
            raise RuntimeError(msg)
        decisions.append(decision)

    messages: list[ChatCompletionMessageParam] = [
        _assistant_tool_call_message(assistant_message),
        *[
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call.id,
                content=json.dumps({"resume": decision}),
            )
            for tool_call, decision in zip(tool_calls, decisions, strict=True)
        ],
    ]
    return messages, None


async def _content_deltas(
    stream: AsyncChatCompletionStream[Any],
    event_emitter: Any = None,
) -> AsyncIterator[str]:
    """Yield text and emit supported portable UI events.

    Yields:
        String chunks for the response.
    """
    async for event in stream:
        if isinstance(event, ContentDeltaEvent):
            yield event.delta
        elif isinstance(event, ChunkEvent) and event_emitter is not None:
            ui_event = _status_event(event.chunk) or _plotly_embed_event(event.chunk)
            if ui_event is not None:
                await event_emitter(ui_event)


def _client_event_data(
    chunk: ChatCompletionChunk,
    event_type: str,
) -> dict[str, Any] | None:
    extension = (chunk.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict) or extension.get("schema_version") != 1:
        return None

    event = extension.get("event")
    if not isinstance(event, dict) or event.get("type") != event_type:
        return None
    data = event.get("data")
    return data if isinstance(data, dict) else None


def _status_event(
    chunk: ChatCompletionChunk,
) -> dict[str, Any] | None:
    data = _client_event_data(chunk, "status")
    if data is None:
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


#### PLOT ####


def _plotly_embed_event(chunk: ChatCompletionChunk) -> dict[str, Any] | None:
    data = _client_event_data(chunk, "artifact")
    if data is None:
        return None
    try:
        artifact = PlotlyArtifact.model_validate(data)
    except ValidationError:
        return None

    html = _plotly_html(artifact)
    if html is None:
        return None
    return {"type": "embeds", "data": {"embeds": [html]}}


def _plotly_html(artifact: PlotlyArtifact) -> str | None:
    try:
        figure = json.dumps(
            artifact.figure,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).replace("<", "\\u003c")
    except (TypeError, ValueError):
        return None

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-3.6.0.min.js" charset="utf-8"></script>
</head><body style="margin:0;padding:16px">
<h2>{escape(artifact.title)}</h2><p>{escape(artifact.summary)}</p>
<div id="plot"></div>
<script>
const figure = {figure};
Plotly.newPlot("plot", figure.data, figure.layout, {{responsive: true}});
</script></body></html>"""


def _user_id(user: dict[str, Any] | None) -> str | None:
    user_id = (user or {}).get("id")
    return user_id if isinstance(user_id, str) and user_id else None


#### HITL ####


async def _approval_decision(
    tool_call: ChatCompletionMessageToolCall,
    event_call: Any,
) -> tuple[str | None, str | None]:
    if event_call is None:
        return None, "Open WebUI approval modal is unavailable for this request."

    event = _approval_event(tool_call)
    if event is None:
        return None, "Open WebUI received an unsupported interrupt payload."

    try:
        approval = await event_call(event)
    except Exception as exc:
        detail = str(exc).strip()
        if not detail:
            detail = "the confirmation session disconnected or timed out"
        return None, f"Open WebUI approval failed: {detail}"
    if isinstance(approval, dict) and approval.get("error"):
        return None, f"Open WebUI approval failed: {approval['error']}"
    if approval is True:
        return "approve", None
    if approval is False:
        return "reject", None

    return None, "Open WebUI approval was cancelled or timed out."


def _interrupt_tool_calls(
    message: ChatCompletionMessage,
) -> list[ChatCompletionMessageToolCall] | None:
    tool_calls = list(message.tool_calls or [])
    if any(tool_call.function.name != INTERRUPT_TOOL_NAME for tool_call in tool_calls):  # ty: ignore[unresolved-attribute]
        return None
    return tool_calls  # ty: ignore[invalid-return-type]


def _assistant_tool_call_message(
    message: ChatCompletionMessage,
) -> ChatCompletionAssistantMessageParam:
    """Copy every tool call because their arguments are the resume cursor."""
    return ChatCompletionAssistantMessageParam(
        role=message.role,
        content=message.content,
        tool_calls=[
            cast(
                ChatCompletionMessageToolCallParam,
                tool_call.model_dump(mode="json"),
            )
            for tool_call in message.tool_calls or []
        ],
    )


def _approval_event(
    tool_call: ChatCompletionMessageToolCall,
) -> dict[str, Any] | None:
    try:
        payload = _interrupt_payload(tool_call)
    except ValueError:
        return None

    if isinstance(payload, dict):
        question = str(payload.get("question") or "Approve this agent action?")
        request = str(
            payload.get("request") or json.dumps(payload, ensure_ascii=False, indent=2)
        )
    else:
        question = "Approve this agent action?"
        request = _json_payload_text(payload)
    return {
        "type": "confirmation",
        "data": {
            "title": question,
            "message": request,
        },
    }


def _interrupt_payload(
    tool_call: ChatCompletionMessageToolCall,
) -> object:
    try:
        arguments = json.loads(tool_call.function.arguments)
    except (TypeError, ValueError) as exc:
        msg = "Interrupt tool arguments must be valid JSON."
        raise ValueError(msg) from exc

    if not isinstance(arguments, dict):
        msg = "Interrupt tool arguments must be a JSON object."
        raise ValueError(msg)

    if "payload" not in arguments:
        msg = "Interrupt tool arguments must contain a payload."
        raise ValueError(msg)

    return arguments["payload"]


def _json_payload_text(payload: object) -> str:
    if isinstance(payload, str) and payload:
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2)
