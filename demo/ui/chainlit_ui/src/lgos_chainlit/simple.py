"""Simple Chainlit UI for the demo OpenAI-compatible LangGraph server."""

import asyncio
import contextlib
import logging
from typing import cast

import chainlit as cl
from chainlit.types import ThreadDict
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from pydantic import ValidationError

from lgos_chainlit.auth import authenticated_user_identifier
from lgos_chainlit.client_settings import (
    model_client_settings,
    settings_metadata,
    settings_widgets,
)
from lgos_chainlit.history import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    text_only_chat_messages,
)
from lgos_chainlit.lgos_protocol import (
    LGOS_EXTENSION_KEY,
    STREAM_EVENTS_METADATA_KEY,
    STREAM_EVENTS_METADATA_VALUE,
    ClientEventExtension,
    ModelClientSettings,
)
from lgos_chainlit.settings import settings

client = AsyncOpenAI(
    base_url=settings.INFERENCE.base_url,
    api_key=settings.INFERENCE.api_key,
)
discovery_client = AsyncOpenAI(
    base_url=settings.chainlit_discovery_endpoint.base_url,
    api_key=settings.chainlit_discovery_endpoint.api_key,
)
logger = logging.getLogger(__name__)


def _client_event(chunk: ChatCompletionChunk) -> dict[str, object] | None:
    extension = (chunk.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict):
        return None

    try:
        parsed = ClientEventExtension.model_validate(extension)
    except ValidationError:
        return None
    return parsed.event.model_dump(mode="json")


class _ClientEventRenderer:
    """Render one live-updating custom element for a completion."""

    def __init__(self) -> None:
        self._events: list[dict[str, object]] = []
        self._element: cl.CustomElement | None = None

    async def render(self, chunk: ChatCompletionChunk) -> None:
        event = _client_event(chunk)
        if event is None:
            return

        self._events.append(event)
        props = {"events": [*self._events]}
        if self._element is None:
            self._element = cl.CustomElement(
                name="ClientEventTimeline",
                props=props,
            )
            message = cl.Message(content="", elements=[self._element])
            mark_model_context_excluded(message)
            await message.send()
            return

        self._element.props = props
        await self._element.update()


@cl.set_chat_profiles
async def set_chat_profiles(
    _current_user: cl.User | None = None,
) -> list[cl.ChatProfile]:
    models = await discovery_client.models.list()

    return [
        cl.ChatProfile(
            name=model.id,
            markdown_description=f"Talk to `{model.id}` from the demo backend.",
        )
        for model in models.data
    ]


@cl.set_starters
async def set_starters(_current_user: cl.User | None = None) -> list[cl.Starter]:
    return [
        cl.Starter(
            label="About",
            message="Tell me about yourself.",
            icon="",
        ),
        cl.Starter(
            label="History",
            message="Remember that my favorite color is green.",
            icon="",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    await configure_chat_settings()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    mark_persisted_errors_excluded(thread)
    saved = cl.user_session.get("chat_settings")
    await configure_chat_settings(saved if isinstance(saved, dict) else None)


@cl.on_message
async def on_message(_message: cl.Message) -> None:
    """Reply from chat context; Chainlit adds the user message before this hook."""
    model = cast(str, cl.user_session.get("chat_profile"))
    messages = text_only_chat_messages()
    assistant_message = cl.Message(content="")
    client_events = _ClientEventRenderer()
    stream = None

    try:
        metadata = settings_metadata(
            current_client_settings(),
            cl.user_session.get("chat_settings"),
        )
        metadata[STREAM_EVENTS_METADATA_KEY] = STREAM_EVENTS_METADATA_VALUE
        stream = await client.chat.completions.create(
            model=settings.chainlit_inference_model(model),
            messages=messages,
            stream=True,
            user=authenticated_user_identifier(),
            metadata=metadata,
        )

        async for chunk in stream:
            await client_events.render(chunk)
            token = chunk.choices[0].delta.content or ""
            if token:
                await assistant_message.stream_token(token)

        await assistant_message.update()
    except asyncio.CancelledError:
        if assistant_message.content:
            mark_model_context_excluded(assistant_message)
            await assistant_message.update()
        raise
    except Exception as exc:
        error = f"Chat completion failed: {exc}"
        if assistant_message.content:
            assistant_message.content = f"{assistant_message.content}\n\n{error}"
            mark_model_context_excluded(assistant_message)
            await assistant_message.update()
        else:
            error_message = cl.Message(content=error)
            mark_model_context_excluded(error_message)
            await error_message.send()
    finally:
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.close()


async def configure_chat_settings(
    saved: dict[str, object] | None = None,
) -> dict[str, object]:
    """Retrieve the selected model and publish its supported settings."""
    model_id = cast(str, cl.user_session.get("chat_profile"))
    try:
        model = await discovery_client.models.retrieve(model_id)
    except Exception:
        logger.warning(
            "Runtime settings discovery failed for %s; settings are inactive",
            model_id,
            exc_info=True,
        )
        await cl.ChatSettings([]).refresh()
        _store_client_settings(None)
        committed = cl.user_session.get("chat_settings")
        return committed if isinstance(committed, dict) else {}

    settings = model_client_settings(model)
    if settings is None:
        await cl.ChatSettings([]).send()
        _store_client_settings(None)
        return {}

    widgets, values = settings_widgets(settings, saved)
    if widgets:
        values = await cl.ChatSettings(widgets).send()
    else:
        await cl.ChatSettings([]).send()
    _store_client_settings(settings)
    return values


def current_client_settings() -> ModelClientSettings | None:
    """Parse the JSON-safe descriptor stored in the Chainlit user session."""
    value = cl.user_session.get("model_client_settings")
    if not isinstance(value, dict):
        return None
    try:
        return ModelClientSettings.model_validate(value)
    except ValidationError:
        logger.warning("Ignoring invalid saved runtime settings", exc_info=True)
        return None


def _store_client_settings(settings: ModelClientSettings | None) -> None:
    cl.user_session.set(
        "model_client_settings",
        settings.model_dump(mode="json") if settings is not None else None,
    )
