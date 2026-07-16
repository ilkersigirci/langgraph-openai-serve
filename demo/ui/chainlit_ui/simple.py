"""Simple Chainlit UI for the demo OpenAI-compatible LangGraph server."""

import asyncio
import contextlib
from typing import cast

import chainlit as cl
from chainlit.types import ThreadDict
from demo.api.settings import settings
from demo.ui.chainlit_ui.auth import authenticated_user_identifier
from demo.ui.chainlit_ui.history import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    text_only_chat_messages,
)
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=settings.CHAINLIT_OPENAI_BASE_URL,
    api_key="DUMMY",
)


@cl.set_chat_profiles
async def set_chat_profiles(
    _current_user: cl.User | None = None,
) -> list[cl.ChatProfile]:
    models = await client.models.list()

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


@cl.on_chat_resume
def on_chat_resume(thread: ThreadDict) -> None:
    mark_persisted_errors_excluded(thread)


@cl.on_message
async def on_message(_message: cl.Message) -> None:
    """Reply from chat context; Chainlit adds the user message before this hook."""
    model = cast(str, cl.user_session.get("chat_profile"))
    messages = text_only_chat_messages()

    assistant_message = cl.Message(content="")
    stream = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            user=authenticated_user_identifier(),
        )

        async for chunk in stream:
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
