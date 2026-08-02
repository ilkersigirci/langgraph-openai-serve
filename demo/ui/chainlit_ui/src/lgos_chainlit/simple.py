"""Simple Chainlit UI for the demo OpenAI-compatible LangGraph server."""

import asyncio
import contextlib

import chainlit as cl
from chainlit.types import ThreadDict
from openai import OpenAIError

from lgos_chainlit.auth import authenticated_user_identifier
from lgos_chainlit.lgos_protocol import (
    STREAM_EVENTS_METADATA_KEY,
    STREAM_EVENTS_METADATA_VALUE,
    GraphFeature,
    model_extension,
)
from lgos_chainlit.utils.chat import (
    LIMITED_FUNCTIONALITY_MESSAGE,
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    send_ui_message,
    text_only_chat_messages,
)
from lgos_chainlit.utils.chat_settings import (
    chat_settings_metadata,
    configure_chat_settings,
    model_feature_enabled,
)
from lgos_chainlit.utils.client_events import ClientEventRenderer
from lgos_chainlit.utils.clients import (
    list_model_ids,
    model_request,
    openai_client,
    retrieve_model,
)


@cl.set_chat_profiles
async def set_chat_profiles(
    _current_user: cl.User | None = None,
) -> list[cl.ChatProfile]:
    profiles = []
    for model_id in await list_model_ids():
        try:
            model = await retrieve_model(model_id)
        except OpenAIError:
            model = None
        description = (
            f"Talk to `{model_id}` from the demo backend."
            if model is not None and model_extension(model) is not None
            else LIMITED_FUNCTIONALITY_MESSAGE
        )
        profiles.append(
            cl.ChatProfile(
                name=model_id,
                markdown_description=description,
            )
        )
    return profiles


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
async def on_message(_message: cl.Message) -> None:
    """Reply from chat context; Chainlit adds the user message before this hook."""
    model = cl.user_session.get("chat_profile")
    if not isinstance(model, str) or not model:
        await send_ui_message("Chat completion failed: no model profile is selected.")
        return

    messages = text_only_chat_messages()
    assistant_message = cl.Message(content="")
    client_events = ClientEventRenderer()
    stream = None

    try:
        metadata = chat_settings_metadata()
        if model_feature_enabled(GraphFeature.CLIENT_EVENTS):
            metadata[STREAM_EVENTS_METADATA_KEY] = STREAM_EVENTS_METADATA_VALUE
        stream = await openai_client.chat.completions.create(
            **model_request(model),
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
            await send_ui_message(error)
    finally:
        try:
            await client_events.close()
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.close()
