"""Simple Chainlit UI for the demo OpenAI-compatible LangGraph server."""

import asyncio
import contextlib
from typing import cast

import chainlit as cl
from chainlit.types import ThreadDict
from chainlit_utils.auth import authenticated_user_identifier
from chainlit_utils.chat import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    send_ui_message,
    text_only_chat_messages,
)
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from lgos_chainlit.auth import register_auth_callback
from lgos_chainlit.lgos_protocol import (
    STREAM_EVENTS_METADATA_KEY,
    STREAM_EVENTS_METADATA_VALUE,
    GraphFeature,
    model_description,
)
from lgos_chainlit.utils.chat import LIMITED_FUNCTIONALITY_MESSAGE, session_metadata
from lgos_chainlit.utils.chat_settings import (
    chat_settings_metadata,
    configure_chat_settings,
    model_feature_enabled,
    streaming_enabled,
)
from lgos_chainlit.utils.client_events import ClientEventRenderer
from lgos_chainlit.utils.clients import (
    list_models,
    model_request,
    openai_client,
)
from lgos_chainlit.utils.files import file_upload_overrides, with_file_parts

register_auth_callback()


@cl.set_chat_profiles
async def set_chat_profiles(
    _current_user: cl.User | None = None,
) -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name=model.id,
            markdown_description=(
                model_description(model) or LIMITED_FUNCTIONALITY_MESSAGE
            ),
            config_overrides=file_upload_overrides(model),
        )
        for model in await list_models()
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
    await configure_chat_settings()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Reply from chat context; Chainlit adds the user message before this hook."""
    model = cl.user_session.get("chat_profile")
    if not isinstance(model, str) or not model:
        await send_ui_message("Chat completion failed: no model profile is selected.")
        return

    assistant_message = cl.Message(content="")
    client_events = ClientEventRenderer()
    stream = None

    try:
        messages = await with_file_parts(text_only_chat_messages(), message)
        streaming = streaming_enabled()
        metadata = chat_settings_metadata()
        metadata.update(session_metadata())
        if streaming and model_feature_enabled(GraphFeature.CLIENT_EVENTS):
            metadata[STREAM_EVENTS_METADATA_KEY] = STREAM_EVENTS_METADATA_VALUE
        response = await openai_client.chat.completions.create(
            **model_request(model),
            messages=messages,
            stream=streaming,
            user=authenticated_user_identifier(),
            metadata=metadata,
        )

        if not streaming:
            completion = cast("ChatCompletion", response)
            assistant_message.content = completion.choices[0].message.content or ""
            await assistant_message.send()
            return

        stream = cast("AsyncStream[ChatCompletionChunk]", response)
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
            await send_ui_message(error)
    finally:
        try:
            await client_events.close()
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.close()
