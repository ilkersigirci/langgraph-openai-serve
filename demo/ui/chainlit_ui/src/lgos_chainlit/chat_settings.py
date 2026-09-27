"""Translate LGOS model metadata into Chainlit chat settings."""

import logging

import chainlit as cl
from chainlit.input_widget import InputWidget, Switch
from chainlit_utils.chat.settings import serialize_settings, settings_widgets
from openai import OpenAIError
from openai.types.responses import CustomToolParam, ToolParam

from lgos_chainlit.clients import retrieve_model
from lgos_chainlit.display_files import DISPLAY_FILE_TOOL
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
PACKAGE_VERSION_SETTING_ID = "lgos_package_version"
WEB_SEARCH_SETTING_ID = "web_search"
PACKAGE_VERSION_PROFILES = {"server-tool"}
WEB_SEARCH_PROFILES = {"advanced-graph", "server-tool"}
DISPLAY_FILE_PROFILES = {"persistent-plot-agent"}
PACKAGE_VERSION_TOOL: CustomToolParam = {
    "type": "custom",
    "name": PACKAGE_VERSION_SETTING_ID,
}


async def configure_chat_settings() -> None:
    """Retrieve the selected model and publish its supported settings."""
    model_id = cl.user_session.get("chat_profile")
    saved = cl.user_session.get("chat_settings")
    graph = _graph_name()
    extension = None
    retrieval_failed = False
    if model_id:
        try:
            extension = model_extension(await retrieve_model(model_id))
        except OpenAIError:
            logger.warning(
                "Model retrieval failed for %s; runtime settings are inactive",
                model_id,
                exc_info=True,
            )
            retrieval_failed = True
    cl.user_session.set(MODEL_EXTENSION_SESSION_KEY, extension)

    widgets: list[InputWidget] = [
        Switch(
            id=STREAMING_SETTING_ID,
            label="Stream response",
            description="Show the answer as it is generated.",
            initial=saved.get(STREAMING_SETTING_ID) is not False,
        )
    ]
    if graph in PACKAGE_VERSION_PROFILES:
        widgets.append(
            Switch(
                id=PACKAGE_VERSION_SETTING_ID,
                label="Package version",
                description="Let LGOS inspect selected server package versions.",
                initial=saved.get(PACKAGE_VERSION_SETTING_ID) is True,
            )
        )
    if graph in WEB_SEARCH_PROFILES:
        widgets.append(
            Switch(
                id=WEB_SEARCH_SETTING_ID,
                label="Web search",
                description="Let LGOS use its configured web-search backend.",
                initial=saved.get(WEB_SEARCH_SETTING_ID) is True,
            )
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
    if retrieval_failed:
        # Unlike send(), refresh() keeps the saved selections in the session.
        await chat_settings.refresh()
    else:
        await chat_settings.send()
    if model_id and extension is None:
        await cl.context.emitter.send_toast(
            LIMITED_FUNCTIONALITY_MESSAGE,
            type="warning",
        )


def response_tools() -> list[ToolParam]:
    """Return the tools available to the selected graph."""
    graph = _graph_name()
    selected = cl.user_session.get("chat_settings")
    tools: list[ToolParam] = []
    if graph in DISPLAY_FILE_PROFILES:
        tools.append(DISPLAY_FILE_TOOL)
    if _model_supports(MCP_TOOLS_FEATURE):
        tools.extend(mcp_tools.response_tools())
    if (
        graph in PACKAGE_VERSION_PROFILES
        and selected.get(PACKAGE_VERSION_SETTING_ID) is True
    ):
        tools.append(PACKAGE_VERSION_TOOL)
    if graph in WEB_SEARCH_PROFILES and selected.get(WEB_SEARCH_SETTING_ID) is True:
        tools.append({"type": "web_search"})
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


def _graph_name() -> str | None:
    model_id = cl.user_session.get("chat_profile")
    return model_id.rsplit("/", 1)[-1] if model_id else None
