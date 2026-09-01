"""
title: Generic

author: langgraph-openai-serve
version: 0.25
"""

import base64
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from html import escape
from typing import Any, Literal, cast

from openai import AsyncOpenAI, AsyncStream, OpenAIError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
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
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/events.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/utils.py
INTERRUPT_TOOL_NAME = "langgraph_interrupt"
ASK_USER_TOOL_NAME = "ask_user"
ASK_USER_CALL_ID_PREFIX = "lgos_ask_"
ASK_USER_MAX_QUESTIONS = 3
ASK_USER_QUESTION_MAX_LENGTH = 500
ASK_USER_REJECTED_OUTPUT = "Error: tool call rejected by user."
INTERRUPT_CANCELLED_MESSAGE = "Interrupt cancelled."
LGOS_EXTENSION_KEY = "langgraph_openai_serve"
CLIENT_EVENTS_FEATURE = "client_events"
OPENAI_METADATA_VALUE_MAX_LENGTH = 512
SESSION_ID_METADATA_KEY = "session_id"
RUNTIME_SETTINGS_METADATA_KEY = "langgraph_runtime_settings"
STREAM_EVENTS_METADATA_KEY = "langgraph_stream_events"
STREAM_EVENTS_METADATA_VALUE = "v1"
LGOS_MODEL_OWNER = "langgraph-openai-serve"
NO_CHOICES_MESSAGE = "LangGraph API returned no choices."
CHAT_COMPLETION_REQUEST_FIELDS = (
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "tools",
    "tool_choice",
)
LIMITED_FUNCTIONALITY_MESSAGE = (
    "Limited functionality: the configured OpenAI endpoint did not return valid "
    "langgraph_openai_serve model metadata. Runtime settings, client events, and "
    "interrupts may be unavailable. Configure the proxy to pass LGOS /v1 requests "
    "and responses through unchanged."
)
PipeChunk = dict[str, Any]
PipeResponse = AsyncIterator[PipeChunk] | PipeChunk


class InterruptCancelled(Exception):
    """The user cancelled Open WebUI's native interrupt prompt."""


class ChartSeries(BaseModel):
    """One portable series in a chart artifact."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    name: str = Field(min_length=1)
    values: list[float]


class ChartArtifact(BaseModel):
    """The supported LGOS chart artifact payload."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1]
    id: str = Field(min_length=1)
    kind: Literal["chart"]
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    chart_type: Literal["bar", "line"]
    labels: list[str]
    series: list[ChartSeries]
    x_axis_title: str = Field(min_length=1)
    y_axis_title: str = Field(min_length=1)
    show_legend: bool


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
        raise ValueError(msg) from exc
    if len(encoded) > OPENAI_METADATA_VALUE_MAX_LENGTH:
        msg = "The selected runtime settings exceed the OpenAI metadata value limit."
        raise ValueError(msg)
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


def _request_messages(body: dict[str, Any]) -> list[ChatCompletionMessageParam]:
    """Translate a persisted Open WebUI answer into an LGOS resume."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []

    resume_messages = _ask_user_to_resume(messages)
    if resume_messages is not None:
        return resume_messages
    return cast(list[ChatCompletionMessageParam], messages)


#### HITL ####


def _ask_user_to_resume(messages: list[Any]) -> list[ChatCompletionMessageParam] | None:
    """Restore the canonical LGOS batch from Open WebUI's persisted answer."""
    if not messages:
        return None

    has_tool_result = (
        len(messages) >= 2
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "tool"
    )
    assistant = messages[-2] if has_tool_result else messages[-1]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        return None

    tool_calls = assistant.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        return None
    ask_call = tool_calls[0]
    if not isinstance(ask_call, dict) or not isinstance(ask_call.get("function"), dict):
        return None
    call_id = ask_call.get("id")
    if (
        ask_call["function"].get("name") != ASK_USER_TOOL_NAME
        or not isinstance(call_id, str)
        or not call_id.startswith(ASK_USER_CALL_ID_PREFIX)
    ):
        return None
    if not has_tool_result:
        msg = "Open WebUI returned an incomplete interrupt batch."
        raise ValueError(msg)

    tool_result = cast(dict[str, Any], messages[-1])
    if tool_result.get("tool_call_id") != call_id:
        msg = "Open WebUI returned an incomplete interrupt batch."
        raise ValueError(msg)

    try:
        encoded = call_id.removeprefix(ASK_USER_CALL_ID_PREFIX)
        padding = "=" * (-len(encoded) % 4)
        interrupt_calls = json.loads(base64.urlsafe_b64decode(encoded + padding))
    except (TypeError, ValueError) as exc:
        msg = "Open WebUI returned an invalid interrupt cursor."
        raise ValueError(msg) from exc
    if (
        not isinstance(interrupt_calls, list)
        or not 1 <= len(interrupt_calls) <= ASK_USER_MAX_QUESTIONS
    ):
        msg = "Open WebUI returned an invalid interrupt cursor."
        raise ValueError(msg)

    content = tool_result.get("content")
    if content == ASK_USER_REJECTED_OUTPUT:
        raise InterruptCancelled
    try:
        answer = json.loads(content) if isinstance(content, str) else None
    except ValueError as exc:
        msg = "Open WebUI returned an invalid interrupt answer."
        raise ValueError(msg) from exc
    if isinstance(answer, dict) and answer.get("status") == "cancelled":
        raise InterruptCancelled
    answers = answer.get("answers") if isinstance(answer, dict) else None
    if (
        not isinstance(answer, dict)
        or answer.get("status") != "answered"
        or not isinstance(answers, dict)
    ):
        msg = "Open WebUI returned an invalid interrupt answer."
        raise ValueError(msg)

    replay: list[dict[str, Any]] = [
        {"role": "assistant", "content": None, "tool_calls": interrupt_calls}
    ]
    for index, interrupt_call in enumerate(interrupt_calls):
        if (
            not isinstance(interrupt_call, dict)
            or not isinstance(interrupt_call.get("id"), str)
            or not isinstance(interrupt_call.get("function"), dict)
            or interrupt_call["function"].get("name") != INTERRUPT_TOOL_NAME
        ):
            msg = "Open WebUI returned an invalid interrupt cursor."
            raise ValueError(msg)
        payload = _interrupt_payload(interrupt_call)
        replay.append(
            {
                "role": "tool",
                "tool_call_id": interrupt_call["id"],
                "content": json.dumps(
                    {
                        "resume": _resume_value(
                            answers.get(f"resume_{index}"),
                            payload,
                        )
                    }
                ),
            }
        )
    return cast(list[ChatCompletionMessageParam], replay)


def _openwebui_chunk(chunk: ChatCompletionChunk) -> PipeChunk:
    value = chunk.model_dump(mode="json", exclude_none=True)
    for choice in value.get("choices", []):
        delta = choice.get("delta")
        if isinstance(delta, dict):
            _rewrite_interrupts(delta, streaming=True)
    return value


def _openwebui_completion(completion: ChatCompletion) -> PipeChunk:
    value = completion.model_dump(mode="json", exclude_none=True)
    for choice in value.get("choices", []):
        message = choice.get("message")
        if isinstance(message, dict):
            _rewrite_interrupts(message)
    return value


def _rewrite_interrupts(message: dict[str, Any], *, streaming: bool = False) -> None:
    tool_calls = message.get("tool_calls")
    if not (
        isinstance(tool_calls, list)
        and tool_calls
        and all(
            isinstance(call, dict)
            and isinstance(call.get("function"), dict)
            and call["function"].get("name") == INTERRUPT_TOOL_NAME
            for call in tool_calls
        )
    ):
        return
    message["tool_calls"] = [
        _interrupts_to_ask_user(
            cast(list[dict[str, Any]], tool_calls),
            streaming=streaming,
        )
    ]


def _interrupts_to_ask_user(
    calls: list[dict[str, Any]],
    *,
    streaming: bool = False,
) -> dict[str, Any]:
    """Present one atomic LGOS interrupt batch as one native question card."""
    if len(calls) > ASK_USER_MAX_QUESTIONS:
        msg = f"Open WebUI supports at most {ASK_USER_MAX_QUESTIONS} interrupts per batch."
        raise ValueError(msg)

    interrupt_calls = []
    questions = []
    for index, call in enumerate(calls):
        function = call.get("function")
        call_id = call.get("id")
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(function, dict)
            or not isinstance(function.get("arguments"), str)
        ):
            msg = "LangGraph API returned invalid interrupt tool arguments."
            raise ValueError(msg)
        interrupt_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": INTERRUPT_TOOL_NAME,
                "arguments": function["arguments"],
            },
        }
        interrupt_calls.append(interrupt_call)
        questions.append(_interrupt_question(_interrupt_payload(interrupt_call), index))

    cursor = json.dumps(
        interrupt_calls,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    result = {
        "id": ASK_USER_CALL_ID_PREFIX
        + base64.urlsafe_b64encode(cursor).decode().rstrip("="),
        "type": "function",
        "function": {
            "name": ASK_USER_TOOL_NAME,
            "arguments": json.dumps(
                {
                    "questions": questions,
                    "allow_other": any(
                        question["allow_other"] for question in questions
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }
    if streaming:
        result["index"] = 0
    return result


def _interrupt_payload(tool_call: dict[str, Any]) -> object:
    try:
        arguments = json.loads(tool_call["function"]["arguments"])
    except (KeyError, TypeError, ValueError) as exc:
        msg = "LangGraph API returned invalid interrupt tool arguments."
        raise ValueError(msg) from exc
    if not isinstance(arguments, dict) or "payload" not in arguments:
        msg = "LangGraph API returned invalid interrupt tool arguments."
        raise ValueError(msg)
    return arguments["payload"]


def _interrupt_question(payload: object, index: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        msg = "Open WebUI requires an object interrupt payload."
        raise ValueError(msg)
    question = payload.get("question")
    choices = payload.get("choices")
    allow_other = payload.get("allow_other", False)
    if not isinstance(question, str) or not question.strip():
        msg = "Open WebUI interrupt payload requires a question."
        raise ValueError(msg)
    if (
        not isinstance(choices, list)
        or not 2 <= len(choices) <= 3
        or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
        or len(set(choices)) != len(choices)
        or not isinstance(allow_other, bool)
    ):
        msg = "Open WebUI interrupts require 2-3 unique string choices."
        raise ValueError(msg)

    details = {
        key: value
        for key, value in payload.items()
        if key not in {"question", "choices", "allow_other"}
    }
    prompt = question.strip()
    if details:
        prompt = f"{prompt}\n\n{json.dumps(details, ensure_ascii=False, indent=2)}"
    if len(prompt) > ASK_USER_QUESTION_MAX_LENGTH:
        msg = (
            "Open WebUI interrupt question exceeds "
            f"{ASK_USER_QUESTION_MAX_LENGTH} characters."
        )
        raise ValueError(msg)
    return {
        "id": f"resume_{index}",
        "header": "Human input",
        "question": prompt,
        "options": [
            {
                "label": choice,
                "description": f"Resume with {choice!r}.",
            }
            for choice in choices
        ],
        "allow_other": allow_other,
    }


def _resume_value(answer: object, payload: object) -> str:
    if not isinstance(answer, dict) or not isinstance(payload, dict):
        msg = "Open WebUI returned an invalid interrupt answer."
        raise ValueError(msg)
    if answer.get("type") == "option":
        index = answer.get("option_index")
        choices = payload.get("choices")
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and isinstance(choices, list)
            and 0 <= index < len(choices)
            and isinstance(choices[index], str)
        ):
            return choices[index]
    elif answer.get("type") == "other" and payload.get("allow_other") is True:
        text = answer.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    msg = "Open WebUI returned an invalid interrupt answer."
    raise ValueError(msg)


def _interrupt_cancelled_response(
    model_id: str,
    *,
    streaming: bool,
) -> dict[str, Any]:
    message = {"role": "assistant", "content": INTERRUPT_CANCELLED_MESSAGE}
    return {
        "id": "chatcmpl-lgos-interrupt-cancelled",
        "object": "chat.completion.chunk" if streaming else "chat.completion",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta" if streaming else "message": message,
                "finish_reason": "stop",
            }
        ],
    }


#### CUSTOM EVENTS ####


def _client_event(chunk: ChatCompletionChunk) -> dict[str, Any] | None:
    extension = (chunk.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict) or extension.get("schema_version") != 1:
        return None

    event = extension.get("event")
    return event if isinstance(event, dict) else None


def _client_event_data(
    event: dict[str, Any],
    event_type: str,
) -> dict[str, Any] | None:
    if event.get("type") != event_type:
        return None
    data = event.get("data")
    return data if isinstance(data, dict) else None


def _status_event(
    event: dict[str, Any],
) -> dict[str, Any] | None:
    data = _client_event_data(event, "status")
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


#### CHART ####


def _chart_embed_event(event: dict[str, Any]) -> dict[str, Any] | None:
    data = _client_event_data(event, "artifact")
    if data is None:
        return None
    try:
        artifact = ChartArtifact.model_validate(data)
    except ValidationError:
        return None

    html = _chart_html(artifact)
    if html is None:
        return None
    return {"type": "embeds", "data": {"embeds": [html], "replace": True}}


def _chart_html(artifact: ChartArtifact) -> str | None:
    trace_type = "bar" if artifact.chart_type == "bar" else "scatter"
    data: list[dict[str, object]] = []
    for series in artifact.series:
        trace: dict[str, object] = {
            "type": trace_type,
            "name": series.name,
            "x": artifact.labels,
            "y": series.values,
            "showlegend": artifact.show_legend,
        }
        if artifact.chart_type == "line":
            trace["mode"] = "lines+markers"
        data.append(trace)
    try:
        figure = json.dumps(
            {
                "data": data,
                "layout": {
                    "title": {"text": artifact.title},
                    "xaxis": {"title": {"text": artifact.x_axis_title}},
                    "yaxis": {"title": {"text": artifact.y_axis_title}},
                    "showlegend": artifact.show_legend,
                },
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).replace("<", "\\u003c")
    except (TypeError, ValueError):
        return None

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-4.0.0.min.js" charset="utf-8"></script>
</head><body style="margin:0;padding:16px">
<h2>{escape(artifact.title)}</h2><p>{escape(artifact.summary)}</p>
<div id="plot"></div>
<script>
const figure = {figure};
function reportHeight() {{
  const height = document.documentElement.scrollHeight;
  parent.postMessage({{type: "iframe:height", height}}, "*");
}}
window.addEventListener("load", reportHeight);
new ResizeObserver(reportHeight).observe(document.body);
Plotly.newPlot("plot", figure.data, figure.layout, {{responsive: true}});
</script></body></html>"""


def _user_id(user: dict[str, Any] | None) -> str | None:
    user_id = (user or {}).get("id")
    return user_id if isinstance(user_id, str) and user_id else None


def _error(detail: str) -> dict[str, Any]:
    return {"error": {"detail": detail}}
