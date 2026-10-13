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
    STREAMING_SETTING_ID,
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
                "lgos/file-input": {
                    "description": "Files",
                    "features": ["file_inputs"],
                    "server_tools": [],
                    "client_tools": [],
                },
                "lgos/stripped": {"features": []},
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
        ("lgos/file-input", "Files", True),
        ("lgos/stripped", LIMITED_FUNCTIONALITY_MESSAGE, False),
    ]


@pytest.mark.parametrize(
    ("lgos", "saved", "offered", "tools"),
    [
        (
            {"features": ["background"]},
            {BACKGROUND_SETTING_ID: True},
            {STREAMING_SETTING_ID: True, BACKGROUND_SETTING_ID: True},
            [],
        ),
        (
            {"server_tools": ["package_version", "web_search"]},
            {"package_version": True, "web_search": True},
            {
                STREAMING_SETTING_ID: True,
                "package_version": True,
                "web_search": True,
            },
            [
                {"type": "custom", "name": "package_version"},
                {"type": "web_search"},
            ],
        ),
        (
            {"features": ["mcp_tools"], "server_tools": ["web_search"]},
            {"web_search": True},
            {STREAMING_SETTING_ID: True, "web_search": True},
            [GATEWAY_TOOL, {"type": "web_search"}],
        ),
        (
            # Only advertised tools are offered, whatever the saved settings say.
            {"client_tools": ["display_file"]},
            {"web_search": True},
            {STREAMING_SETTING_ID: True},
            [DISPLAY_FILE_TOOL],
        ),
    ],
    ids=["background", "server-tools", "mcp-and-search", "client-tool"],
)
async def test_advertised_metadata_selects_the_offered_settings_and_tools(
    chainlit_context,
    fake_gateway,
    lgos: dict[str, object],
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

    await select_profile(fake_gateway, "lgos/any-graph", **lgos)

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
        fake_gateway, "lgos/simple-graph", client_settings=RUNTIME_SETTINGS
    )
    fake_gateway.replies.append(reply(response(message("Complete answer"))))

    await chat.on_message(user_message("Hello"))

    assert fake_gateway.bodies("/v1/responses") == [
        {
            "model": "lgos/simple-graph",
            "input": [{"role": "user", "content": "Hello"}],
            "tools": [],
            "safety_identifier": "demo-user",
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
    "model_reply",
    [
        httpx2.Response(503, json={"error": "unavailable"}),
        model_info({"lgos/simple-graph": {"features": []}}),
    ],
    ids=["retrieval-failed", "invalid-metadata"],
)
async def test_limited_metadata_keeps_saved_settings_with_a_warning(
    chainlit_context,
    fake_gateway,
    monkeypatch: pytest.MonkeyPatch,
    model_reply: httpx2.Response,
) -> None:
    send_toast = AsyncMock()
    monkeypatch.setattr(chainlit_context.emitter, "send_toast", send_toast)
    chainlit_context.session.chat_profile = "lgos/simple-graph"
    chainlit_context.session.chat_settings = {"mode": "detailed"}
    fake_gateway.replies.append(model_reply)

    await configure_chat_settings()

    send_toast.assert_awaited_once_with(LIMITED_FUNCTIONALITY_MESSAGE, type="warning")
    assert cl.user_session.get("chat_settings") == {"mode": "detailed"}
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
        "lgos/simple-graph",
        features=["background"],
        client_settings={"json_schema": {}},
    )

    send_toast.assert_not_awaited()
    assert cl.user_session.get("chat_settings") == {
        STREAMING_SETTING_ID: True,
        BACKGROUND_SETTING_ID: False,
    }


async def test_missing_profile_rejects_messages(
    chainlit_context,
    fake_gateway,
) -> None:
    await configure_chat_settings()
    await chat.on_message(user_message("Hello"))

    assert fake_gateway.requests == []
    assert transcript() == ["Hello", "Response failed: no model profile is selected."]
