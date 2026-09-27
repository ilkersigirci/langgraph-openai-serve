"""Chat profiles, settings, and tools derived from LGOS model metadata."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import chainlit as cl
import httpx2
import pytest
from mcp.types import Tool

from lgos_chainlit import chat
from lgos_chainlit.chat_settings import (
    BACKGROUND_SETTING_ID,
    LIMITED_FUNCTIONALITY_MESSAGE,
    PACKAGE_VERSION_SETTING_ID,
    PACKAGE_VERSION_TOOL,
    STREAMING_SETTING_ID,
    WEB_SEARCH_SETTING_ID,
    chat_settings_metadata,
    configure_chat_settings,
    response_tools,
)
from lgos_chainlit.display_files import DISPLAY_FILE_TOOL
from lgos_chainlit.mcp import MCP_GATEWAY_NAME, mcp_tools
from tests.support import (
    message,
    model_info,
    reply,
    response,
    select_profile,
    transcript,
    user_message,
)

RUNTIME_SETTINGS = {
    "json_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "use_history": {
                "type": "boolean",
                "title": "Use conversation history",
                "default": True,
            },
            "mode": {
                "type": "string",
                "title": "Mode",
                "enum": ["brief", "detailed"],
                "default": "brief",
            },
            "assistant_name": {
                "type": "string",
                "title": "Assistant name",
                "minLength": 1,
                "default": "Helper",
            },
        },
    },
    "defaults": {"use_history": True, "mode": "brief", "assistant_name": "Helper"},
}
GATEWAY_TOOL = {
    "type": "function",
    "name": "database_report",
    "parameters": {"type": "object"},
    "strict": False,
}


async def test_chat_profiles_use_list_metadata_for_descriptions_and_uploads(
    fake_gateway,
) -> None:
    fake_gateway.replies.append(
        model_info(
            {
                "lgos-a/file-input": {
                    "description": "Files",
                    "features": ["file_inputs"],
                },
                "lgos-a/stripped": {"features": []},
            }
        )
    )

    profiles = await chat.set_chat_profiles(None)

    assert [
        (
            profile.name,
            profile.markdown_description,
            profile.config_overrides.features.spontaneous_file_upload.enabled,
        )
        for profile in profiles
    ] == [
        ("lgos-a/file-input", "Files", True),
        ("lgos-a/stripped", LIMITED_FUNCTIONALITY_MESSAGE, False),
    ]


@pytest.mark.parametrize(
    ("profile", "features", "saved", "offered", "tools"),
    [
        (
            "lgos-a/background-report",
            ["background"],
            {BACKGROUND_SETTING_ID: True},
            {STREAMING_SETTING_ID: True, BACKGROUND_SETTING_ID: True},
            [],
        ),
        (
            "provider/server-tool",
            [],
            {PACKAGE_VERSION_SETTING_ID: True, WEB_SEARCH_SETTING_ID: True},
            {
                STREAMING_SETTING_ID: True,
                PACKAGE_VERSION_SETTING_ID: True,
                WEB_SEARCH_SETTING_ID: True,
            },
            [PACKAGE_VERSION_TOOL, {"type": "web_search"}],
        ),
        (
            "lgos-a/advanced-graph",
            ["mcp_tools"],
            {WEB_SEARCH_SETTING_ID: True},
            {STREAMING_SETTING_ID: True, WEB_SEARCH_SETTING_ID: True},
            [GATEWAY_TOOL, {"type": "web_search"}],
        ),
        (
            "provider/persistent-plot-agent",
            [],
            {WEB_SEARCH_SETTING_ID: True},
            {STREAMING_SETTING_ID: True},
            [DISPLAY_FILE_TOOL],
        ),
    ],
    ids=["background", "server-tool", "advanced-graph", "plot-agent"],
)
async def test_profile_settings_select_the_offered_tools(
    chainlit_context,
    fake_gateway,
    profile: str,
    features: list[str],
    saved: dict[str, object],
    offered: dict[str, object],
    tools: list[dict[str, object]],
) -> None:
    # The gateway MCP session is connected, but only mcp_tools graphs get it.
    discovered = SimpleNamespace(
        tools=[Tool(name="database_report", inputSchema={"type": "object"})]
    )
    await mcp_tools.connect(
        SimpleNamespace(name=MCP_GATEWAY_NAME),
        SimpleNamespace(list_tools=AsyncMock(return_value=discovered)),
    )
    chainlit_context.session.chat_settings = saved

    await select_profile(fake_gateway, profile, features=features)

    assert cl.user_session.get("chat_settings") == offered
    assert response_tools() == tools


async def test_selected_settings_reach_the_responses_request(
    chainlit_context,
    fake_gateway,
) -> None:
    chainlit_context.session.chat_settings = {
        STREAMING_SETTING_ID: False,
        "use_history": False,
        "mode": "detailed",
        "assistant_name": "Guide",
    }
    await select_profile(
        fake_gateway, "lgos-a/simple-graph", client_settings=RUNTIME_SETTINGS
    )
    fake_gateway.replies.append(reply(response(message("Complete answer"))))

    await chat.on_message(user_message("Hello"))

    assert fake_gateway.bodies("/v1/responses") == [
        {
            "model": "lgos-a/simple-graph",
            "input": [{"role": "user", "content": "Hello"}],
            "tools": [],
            "user": "demo-user",
            "metadata": {
                "lgos_settings": (
                    '{"use_history":false,"mode":"detailed","assistant_name":"Guide"}'
                ),
                "conversation_id": chainlit_context.session.thread_id,
            },
            "store": False,
        }
    ]
    assert transcript() == ["Hello", "Complete answer"]


@pytest.mark.parametrize(
    ("model_reply", "kept_settings"),
    [
        (httpx2.Response(503, json={"error": "unavailable"}), {"mode": "detailed"}),
        (
            model_info({"lgos-a/simple-graph": {"features": []}}),
            {STREAMING_SETTING_ID: True},
        ),
    ],
    ids=["retrieval-failed", "invalid-metadata"],
)
async def test_limited_metadata_disables_runtime_settings_with_a_warning(
    chainlit_context,
    fake_gateway,
    monkeypatch: pytest.MonkeyPatch,
    model_reply: httpx2.Response,
    kept_settings: dict[str, object],
) -> None:
    send_toast = AsyncMock()
    monkeypatch.setattr(chainlit_context.emitter, "send_toast", send_toast)
    chainlit_context.session.chat_profile = "lgos-a/simple-graph"
    chainlit_context.session.chat_settings = {"mode": "detailed"}
    fake_gateway.replies.append(model_reply)

    await configure_chat_settings()

    send_toast.assert_awaited_once_with(LIMITED_FUNCTIONALITY_MESSAGE, type="warning")
    assert cl.user_session.get("chat_settings") == kept_settings
    assert chat_settings_metadata() == {}


async def test_malformed_runtime_settings_keep_the_model_features(
    chainlit_context,
    fake_gateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send_toast = AsyncMock()
    monkeypatch.setattr(chainlit_context.emitter, "send_toast", send_toast)

    await select_profile(
        fake_gateway,
        "lgos-a/simple-graph",
        features=["background"],
        client_settings={"json_schema": {}},
    )

    send_toast.assert_not_awaited()
    assert cl.user_session.get("chat_settings") == {
        STREAMING_SETTING_ID: True,
        BACKGROUND_SETTING_ID: False,
    }


async def test_missing_profile_offers_streaming_only_and_rejects_messages(
    chainlit_context,
    fake_gateway,
) -> None:
    await configure_chat_settings()
    await chat.on_message(user_message("Hello"))

    assert fake_gateway.requests == []
    assert cl.user_session.get("chat_settings") == {STREAMING_SETTING_ID: True}
    assert transcript() == ["Hello", "Response failed: no model profile is selected."]
