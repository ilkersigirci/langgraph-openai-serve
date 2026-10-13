"""Translate LGOS model metadata into Chainlit chat settings."""

import logging

import chainlit as cl
from chainlit.input_widget import InputWidget, Switch
from chainlit_utils.chat.settings import serialize_settings, settings_widgets
from openai import OpenAIError
from openai.types.responses import ToolParam

from lgos_chainlit.clients import retrieve_model
from lgos_chainlit.display_files import DISPLAY_FILE_TOOL, DISPLAY_FILE_TOOL_NAME
from lgos_chainlit.lgos_protocol import (
    BACKGROUND_FEATURE,
    LGOS_EXTENSION_KEY,
    MCP_TOOLS_FEATURE,
    OPENAI_METADATA_VALUE_MAX_LENGTH,
    SETTINGS_METADATA_KEY,
    LangGraphModelExtension,
    model_extension,
)
from lgos_chainlit.mcp import mcp_tools

logger = logging.getLogger(__name__)
LIMITED_FUNCTIONALITY_MESSAGE = (
    "Limited functionality: The configured OpenAI endpoint did not return valid "
    f"{LGOS_EXTENSION_KEY} model metadata. Runtime settings, file uploads, "
    "and gateway tools may be unavailable."
)
MODEL_EXTENSION_SESSION_KEY = "lgos_model_extension"
STREAMING_SETTING_ID = "lgos_chainlit_stream"
BACKGROUND_SETTING_ID = "lgos_chainlit_background"
# OpenAI's built-in tool; every other advertised server tool is a custom tool.
WEB_SEARCH_TOOL_NAME = "web_search"


async def configure_chat_settings() -> None:
    """Retrieve the selected model and publish its supported settings."""
    model_id = cl.user_session.get("chat_profile")
    saved = cl.user_session.get("chat_settings")
    extension = None
    if model_id:
        try:
            extension = model_extension(await retrieve_model(model_id))
        except OpenAIError:
            logger.warning(
                "Model retrieval failed for %s; runtime settings are inactive",
                model_id,
                exc_info=True,
            )
    cl.user_session.set(MODEL_EXTENSION_SESSION_KEY, extension)

    widgets: list[InputWidget] = [
        Switch(
            id=STREAMING_SETTING_ID,
            label="Stream response",
            description="Show the answer as it is generated.",
            initial=saved.get(STREAMING_SETTING_ID) is not False,
        )
    ]
    if extension is not None:
        widgets.extend(
            Switch(
                id=name,
                label=name.replace("_", " ").capitalize(),
                description="Let the graph run this server tool.",
                initial=saved.get(name) is True,
            )
            for name in extension.server_tools
        )
    if extension is not None and BACKGROUND_FEATURE in extension.features:
        widgets.append(
            Switch(
                id=BACKGROUND_SETTING_ID,
                label="Run in background",
                description="Submit this response to the background worker and poll it.",
                initial=saved.get(BACKGROUND_SETTING_ID) is True,
            )
        )
    if extension is not None and extension.client_settings is not None:
        widgets.extend(
            settings_widgets(
                extension.client_settings.json_schema,
                extension.client_settings.defaults,
                saved,
            )
        )

    chat_settings = cl.ChatSettings(widgets)
    if extension is not None:
        # send() makes the session match the form, dropping stale settings.
        await chat_settings.send()
        return
    # Without trusted metadata, refresh() shows the form but keeps every saved
    # selection, so they return intact once the metadata does.
    await chat_settings.refresh()
    if model_id:
        await cl.context.emitter.send_toast(
            LIMITED_FUNCTIONALITY_MESSAGE,
            type="warning",
        )


def response_tools() -> list[ToolParam]:
    """Return the tools the selected graph advertises and the user enabled."""
    extension = _model_extension()
    if extension is None:
        return []
    selected = cl.user_session.get("chat_settings")
    tools: list[ToolParam] = []
    if DISPLAY_FILE_TOOL_NAME in extension.client_tools:
        tools.append(DISPLAY_FILE_TOOL)
    if MCP_TOOLS_FEATURE in extension.features:
        tools.extend(mcp_tools.response_tools())
    tools.extend(
        {"type": "web_search"}
        if name == WEB_SEARCH_TOOL_NAME
        else {"type": "custom", "name": name}
        for name in extension.server_tools
        if selected.get(name) is True
    )
    return tools


def streaming_enabled() -> bool:
    """Return the Chainlit response-delivery preference."""
    return cl.user_session.get("chat_settings").get(STREAMING_SETTING_ID) is not False


def background_enabled() -> bool:
    """Return whether this turn should use background execution."""
    return (
        _model_supports(BACKGROUND_FEATURE)
        and cl.user_session.get("chat_settings").get(BACKGROUND_SETTING_ID) is True
    )


def chat_settings_metadata() -> dict[str, str]:
    """Encode the current settings relative to their discovered defaults."""
    extension = _model_extension()
    client_settings = extension.client_settings if extension is not None else None
    encoded = serialize_settings(
        client_settings.defaults if client_settings is not None else None,
        cl.user_session.get("chat_settings"),
        max_length=OPENAI_METADATA_VALUE_MAX_LENGTH,
    )
    return {SETTINGS_METADATA_KEY: encoded} if encoded is not None else {}


def _model_extension() -> LangGraphModelExtension | None:
    return cl.user_session.get(MODEL_EXTENSION_SESSION_KEY)


def _model_supports(feature: str) -> bool:
    extension = _model_extension()
    return extension is not None and feature in extension.features
