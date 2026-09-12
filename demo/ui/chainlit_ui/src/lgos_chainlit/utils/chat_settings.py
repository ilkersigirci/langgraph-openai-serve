"""Translate LGOS model metadata into Chainlit chat settings."""

import logging
from collections.abc import Mapping

import chainlit as cl
from chainlit.input_widget import Switch
from chainlit_utils.chat_settings import (
    serialize_settings,
    settings_widgets,
)
from openai import OpenAIError
from openai.types.responses import ToolParam

from lgos_chainlit.lgos_protocol import (
    OPENAI_METADATA_VALUE_MAX_LENGTH,
    SETTINGS_METADATA_KEY,
    GraphFeature,
    model_client_settings,
    model_extension,
)
from lgos_chainlit.utils.chat import send_limited_functionality_warning
from lgos_chainlit.utils.clients import retrieve_model
from lgos_chainlit.utils.responses import DISPLAY_FILE_TOOL

logger = logging.getLogger(__name__)
RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY = "lgos_runtime_settings_defaults"
MODEL_FEATURES_SESSION_KEY = "lgos_model_features"
STREAMING_SETTING_ID = "lgos_chainlit_stream"
CLOCK_SETTING_ID = "lgos_current_time"
WEB_SEARCH_SETTING_ID = "web_search"


async def configure_chat_settings() -> None:
    """Retrieve the selected model and publish its supported settings."""
    model_id = cl.user_session.get("chat_profile")
    saved = cl.user_session.get("chat_settings")
    candidates = dict(saved) if isinstance(saved, dict) else None
    streaming = candidates.get(STREAMING_SETTING_ID) if candidates else None
    widgets = [
        Switch(
            id=STREAMING_SETTING_ID,
            label="Stream response",
            description="Show the answer as it is generated.",
            initial=streaming if type(streaming) is bool else True,
        )
    ]
    if _is_hosted_tool_profile(model_id):
        widgets.extend(
            [
                Switch(
                    id=CLOCK_SETTING_ID,
                    label="Current time",
                    description="Let LGOS read its server clock.",
                    initial=_selected(candidates, CLOCK_SETTING_ID),
                ),
                Switch(
                    id=WEB_SEARCH_SETTING_ID,
                    label="Web search",
                    description="Let LGOS use its configured web-search backend.",
                    initial=_selected(candidates, WEB_SEARCH_SETTING_ID),
                ),
            ]
        )
    _store_runtime_settings_defaults(None)
    _store_model_features(None)
    if not isinstance(model_id, str) or not model_id:
        await cl.ChatSettings(widgets).send()
        return

    try:
        model = await retrieve_model(model_id)
    except OpenAIError:
        logger.warning(
            "Model retrieval failed for %s; runtime settings are inactive",
            model_id,
            exc_info=True,
        )
        await cl.ChatSettings(widgets).refresh()
        await send_limited_functionality_warning()
        return

    extension = model_extension(model)
    if extension is None:
        await cl.ChatSettings(widgets).send()
        await send_limited_functionality_warning()
        return

    _store_model_features(extension.features)
    client_settings = model_client_settings(model)
    if client_settings is None:
        await cl.ChatSettings(widgets).send()
        return

    widgets.extend(
        settings_widgets(
            client_settings.json_schema,
            client_settings.defaults,
            candidates,
        )
    )
    await cl.ChatSettings(widgets).send()
    _store_runtime_settings_defaults(client_settings.defaults)


def response_tools() -> list[ToolParam]:
    """Return the tools available to the selected graph."""
    if not _is_hosted_tool_profile(cl.user_session.get("chat_profile")):
        return [DISPLAY_FILE_TOOL]
    selected = cl.user_session.get("chat_settings")
    if not isinstance(selected, dict):
        return []
    tools: list[ToolParam] = []
    if selected.get(CLOCK_SETTING_ID) is True:
        tools.append({"type": "custom", "name": CLOCK_SETTING_ID})
    if selected.get(WEB_SEARCH_SETTING_ID) is True:
        tools.append({"type": "web_search"})
    return tools


def model_feature_enabled(feature: GraphFeature) -> bool:
    """Return whether the selected model advertised a feature."""
    features = cl.user_session.get(MODEL_FEATURES_SESSION_KEY)
    return isinstance(features, list) and feature.value in features


def streaming_enabled() -> bool:
    """Return the Chainlit response-delivery preference."""
    selected = cl.user_session.get("chat_settings")
    value = selected.get(STREAMING_SETTING_ID) if isinstance(selected, dict) else None
    return value if type(value) is bool else True


def chat_settings_metadata() -> dict[str, str]:
    """Encode the current settings relative to their discovered defaults."""
    defaults = cl.user_session.get(RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY)
    selected = cl.user_session.get("chat_settings")
    encoded = serialize_settings(
        defaults if isinstance(defaults, dict) else None,
        selected if isinstance(selected, dict) else None,
        max_length=OPENAI_METADATA_VALUE_MAX_LENGTH,
    )
    return {SETTINGS_METADATA_KEY: encoded} if encoded is not None else {}


def _store_runtime_settings_defaults(
    defaults: Mapping[str, object] | None,
) -> None:
    """Store the baseline needed to encode settings on later requests."""
    cl.user_session.set(
        RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY,
        dict(defaults) if defaults is not None else None,
    )


def _store_model_features(features: list[str] | None) -> None:
    """Store model capabilities returned by the OpenAI endpoint."""
    cl.user_session.set(MODEL_FEATURES_SESSION_KEY, features)


def _selected(settings: dict[str, object] | None, key: str) -> bool:
    return settings is not None and settings.get(key) is True


def _is_hosted_tool_profile(model_id: object) -> bool:
    return isinstance(model_id, str) and model_id.rsplit("/", 1)[-1] == "hosted-tool"
