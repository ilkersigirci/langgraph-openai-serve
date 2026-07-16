"""Simple Chainlit UI for the demo OpenAI-compatible LangGraph server."""

import asyncio
import contextlib
import logging
from typing import cast

import chainlit as cl
from chainlit.types import ThreadDict
from demo.api.settings import settings
from demo.ui.chainlit_ui.auth import authenticated_user_identifier
from demo.ui.chainlit_ui.client_config import (
    model_client_config,
    settings_metadata,
    settings_widgets,
)
from demo.ui.chainlit_ui.history import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    text_only_chat_messages,
)
from openai import AsyncOpenAI
from pydantic import ValidationError

from langgraph_openai_serve.api.models.schemas import ModelClientConfig

client = AsyncOpenAI(
    base_url=settings.CHAINLIT_INFERENCE.base_url,
    api_key=settings.CHAINLIT_INFERENCE.api_key,
)
discovery_client = AsyncOpenAI(
    base_url=settings.chainlit_discovery_endpoint.base_url,
    api_key=settings.chainlit_discovery_endpoint.api_key,
)
logger = logging.getLogger(__name__)


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
    stream = None

    try:
        metadata = settings_metadata(
            current_client_config(),
            cl.user_session.get("chat_settings"),
        )
        stream = await client.chat.completions.create(
            model=settings.chainlit_inference_model(model),
            messages=messages,
            stream=True,
            user=authenticated_user_identifier(),
            metadata=metadata or None,
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


async def configure_chat_settings(
    saved: dict[str, object] | None = None,
) -> dict[str, object]:
    """Retrieve the selected model and publish its supported settings."""
    model_id = cast(str, cl.user_session.get("chat_profile"))
    try:
        model = await discovery_client.models.retrieve(model_id)
    except Exception:
        logger.warning(
            "Model configuration discovery failed for %s; settings are inactive",
            model_id,
            exc_info=True,
        )
        await cl.ChatSettings([]).refresh()
        _store_client_config(None)
        committed = cl.user_session.get("chat_settings")
        return committed if isinstance(committed, dict) else {}

    config = model_client_config(model)
    if config is None:
        await cl.ChatSettings([]).send()
        _store_client_config(None)
        return {}

    widgets, values = settings_widgets(config, saved)
    if widgets:
        values = await cl.ChatSettings(widgets).send()
    else:
        await cl.ChatSettings([]).send()
    _store_client_config(config)
    return values


def current_client_config() -> ModelClientConfig | None:
    """Parse the JSON-safe descriptor stored in the Chainlit user session."""
    value = cl.user_session.get("model_client_config")
    if not isinstance(value, dict):
        return None
    try:
        return ModelClientConfig.model_validate(value)
    except ValidationError:
        logger.warning("Ignoring invalid saved model configuration", exc_info=True)
        return None


def _store_client_config(config: ModelClientConfig | None) -> None:
    cl.user_session.set(
        "model_client_config",
        config.model_dump(mode="json") if config is not None else None,
    )
