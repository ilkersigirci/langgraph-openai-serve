"""Open WebUI model entries and model info for gateway graphs."""

import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types import Model
from pydantic import JsonValue

from lgos_openwebui.connection import without_lgos_models
from lgos_openwebui.functions.generic import pipe as generic_pipe
from lgos_openwebui.functions.generic.catalog import (
    LIMITED_FUNCTIONALITY_DESCRIPTION,
    openwebui_models,
)

pytestmark = pytest.mark.usefixtures("gateway_environment")

SETTINGS_FEATURES = ["background", "file_inputs", "mcp_tools"]


def graph(model_id: str, **extension: Any) -> Model:
    return Model.model_validate(
        {
            "id": model_id,
            "object": "model",
            "created": 1,
            "owned_by": "langgraph-openai-serve",
            "lgos": {
                "description": "Test graph",
                "features": [],
                "server_tools": [],
                "client_tools": [],
                **extension,
            },
        }
    )


def settings(
    properties: dict[str, JsonValue], defaults: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    return {"json_schema": {"properties": properties}, "defaults": defaults}


def serve_catalog(
    monkeypatch: pytest.MonkeyPatch, handle: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Answer the Pipe's catalog reads with ``handle``."""

    @asynccontextmanager
    async def catalog_client(**kwargs: Any) -> AsyncGenerator[AsyncOpenAI, None]:
        async with AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as client:
            yield client

    monkeypatch.setattr(generic_pipe, "_client", catalog_client)


def meta(model: Model) -> dict[str, Any]:
    (listed,) = [item for item in openwebui_models([model]) if item["id"] == model.id]
    return listed["info"]["meta"]


@pytest.mark.parametrize(
    ("auth_type", "credential"),
    [("bearer", "test-api-key"), ("system_oauth", "user-access-token")],
)
@pytest.mark.parametrize("gateway_type", ["bifrost", "litellm"])
async def test_pipe_lists_gateway_graphs_with_their_openwebui_model_info(
    monkeypatch: pytest.MonkeyPatch,
    openwebui_connection: Callable[..., None],
    gateway_type: str,
    auth_type: str,
    credential: str,
) -> None:
    extension = {
        "description": "  Simple graph  ",
        "features": SETTINGS_FEATURES,
        "server_tools": [],
        "client_tools": [],
        "client_settings": settings(
            {"enabled": {"type": "boolean", "title": "Enabled"}}, {"enabled": False}
        ),
    }
    if gateway_type == "bifrost":
        path = "/v1/models"
        catalog = {
            "object": "list",
            "data": [
                {
                    "id": "lgos/simple-graph",
                    "object": "model",
                    "created": 1,
                    "owned_by": "langgraph-openai-serve",
                    "additional_attributes": {"lgos": json.dumps(extension)},
                },
                {"id": "openai/gpt-5", "object": "model", "owned_by": "openai"},
            ],
        }
    else:
        path = "/model/info"
        catalog = {
            "data": [
                {"model_name": "lgos/simple-graph", "model_info": {"lgos": extension}},
                {"model_name": "openai/gpt-5", "model_info": {}},
            ]
        }

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == path
        assert request.headers["Authorization"] == f"Bearer {credential}"
        return httpx.Response(200, json=catalog)

    monkeypatch.setenv("OPENAI_GATEWAY_TYPE", gateway_type)
    openwebui_connection(auth_type)
    serve_catalog(monkeypatch, handle)
    shared_meta = {
        "capabilities": {"file_upload": True, "file_context": False},
        "builtinTools": {"files": False},
        "toolIds": ["server:mcp:lgos-gateway"],
        "lgos": {"server_tools": [], "client_tools": []},
    }

    assert await generic_pipe.Pipe().pipes(
        __oauth_token__={"access_token": "user-access-token"}
    ) == [
        {
            "id": "lgos/simple-graph",
            "name": "lgos/simple-graph",
            "info": {
                "meta": {
                    **shared_meta,
                    "description": "Simple graph",
                    "chat_variables_schema": {
                        "fields": [
                            {
                                "key": "enabled",
                                "type": "checkbox",
                                "label": "Enabled",
                                "default": False,
                            },
                            {
                                "key": "lgos_background",
                                "type": "checkbox",
                                "label": "Run in background",
                                "default": False,
                            },
                        ]
                    },
                }
            },
        },
        # The UserValves example takes its settings from its Filter.
        {
            "id": "uservalves_simple",
            "name": "UserValves Simple / simple-graph",
            "info": {
                "meta": {
                    **shared_meta,
                    "description": "Static per-user history and audience settings.",
                    "filterIds": ["uservalves_simple"],
                }
            },
        },
    ]


@pytest.mark.parametrize("oauth_token", [None, {}, {"access_token": ""}])
async def test_sso_catalog_reads_without_an_access_token_never_use_the_shared_key(
    monkeypatch: pytest.MonkeyPatch,
    openwebui_connection: Callable[..., None],
    oauth_token: dict[str, str] | None,
) -> None:
    def client(**_: object) -> None:
        pytest.fail("The Pipe contacted the gateway without the user's token.")

    monkeypatch.setattr(generic_pipe, "_client", client)
    openwebui_connection("system_oauth")

    with pytest.raises(RuntimeError, match="Sign in again with SSO"):
        await generic_pipe.Pipe().pipes(__oauth_token__=oauth_token)


def test_settings_become_typed_chat_variables() -> None:
    model = graph(
        "lgos/settings-graph",
        client_settings=settings(
            {
                "use_history": {"type": "boolean", "title": "Use conversation history"},
                "mode": {"type": "string", "title": "Mode", "enum": ["brief", "deep"]},
                "assistant_name": {"type": "string", "title": "Assistant name"},
                "retries": {"type": "integer"},
                "delay": {"type": "integer", "minimum": 0, "maximum": 300},
            },
            {
                "use_history": False,
                "mode": "brief",
                "assistant_name": "Helper",
                "retries": 3,
                "delay": 5,
            },
        ),
    )

    assert meta(model)["chat_variables_schema"] == {
        "fields": [
            {
                "key": "use_history",
                "type": "checkbox",
                "label": "Use conversation history",
                "default": False,
            },
            {
                "key": "mode",
                "type": "select",
                "label": "Mode",
                "options": ["brief", "deep"],
                "default": "brief",
            },
            {
                "key": "assistant_name",
                "type": "text",
                "label": "Assistant name",
                "default": "Helper",
            },
            {
                "key": "retries",
                "type": "number",
                "label": "Retries",
                "default": 3,
                "step": 1,
            },
            {
                "key": "delay",
                "type": "number",
                "label": "Delay",
                "default": 5,
                "step": 1,
                "min": 0,
                "max": 300,
            },
        ]
    }


@pytest.mark.parametrize(
    ("name", "schema", "default"),
    [
        ("bad-key", {"type": "string"}, "value"),
        ("invalid", {"type": "boolean"}, "false"),
        ("invalid", {"type": "boolean"}, 0),
        ("invalid", {"type": "string"}, False),
        ("invalid", {"type": "string", "enum": []}, "value"),
        ("invalid", {"type": "string", "enum": ["a", "a"]}, "a"),
        ("invalid", {"type": "string", "enum": ["a", {}]}, "a"),
        ("invalid", {"type": "string", "enum": ["a"]}, "b"),
        ("invalid", {"type": "object"}, {}),
        ("invalid", {"type": "integer"}, "3"),
        ("invalid", {"type": "integer"}, True),
        ("invalid", None, "value"),
    ],
)
def test_unsupported_settings_are_omitted_without_losing_valid_ones(
    name: str, schema: JsonValue, default: JsonValue
) -> None:
    model = graph(
        "lgos/settings-graph",
        client_settings=settings(
            {"enabled": {"type": "boolean"}, name: schema},
            {"enabled": False, name: default},
        ),
    )

    assert meta(model)["chat_variables_schema"]["fields"] == [
        {"key": "enabled", "label": "Enabled", "default": False, "type": "checkbox"},
    ]


@pytest.mark.parametrize(
    ("extension", "controls"),
    [
        (
            {"server_tools": ["package_version", "web_search"]},
            [
                ("package_version", "Package version"),
                ("web_search", "Web search"),
            ],
        ),
        ({"features": ["background"]}, [("lgos_background", "Run in background")]),
        # A name Open WebUI cannot use as a Chat Variable key gets no control.
        ({"server_tools": ["web-search"], "client_tools": ["display_file"]}, []),
    ],
    ids=["server-tools", "background", "no-control"],
)
def test_controls_follow_the_advertised_tools_and_features(
    extension: dict[str, Any], controls: list[tuple[str, str]]
) -> None:
    model = graph("lgos/any-graph", **extension)
    model_meta = meta(model)

    fields = model_meta.get("chat_variables_schema", {"fields": []})["fields"]
    assert [(field["key"], field["label"], field["default"]) for field in fields] == [
        (key, label, False) for key, label in controls
    ]
    assert model_meta["lgos"] == {
        "server_tools": extension.get("server_tools", []),
        "client_tools": extension.get("client_tools", []),
    }


def test_models_without_lgos_metadata_stay_visible_with_limited_functionality() -> None:
    model = Model.model_validate(
        {
            "id": "lgos/simple-graph",
            "object": "model",
            "created": 1,
            "owned_by": "langgraph-openai-serve",
            "lgos": {"features": ["file_inputs"]},
        }
    )

    assert openwebui_models([model]) == [
        {
            "id": "lgos/simple-graph",
            "name": "lgos/simple-graph (Limited functionality)",
            "info": {
                "meta": {
                    "description": LIMITED_FUNCTIONALITY_DESCRIPTION,
                    "capabilities": {"file_upload": False, "file_context": False},
                    "builtinTools": {"files": False},
                }
            },
        }
    ]


async def test_the_gateway_connection_never_lists_lgos_models() -> None:
    # The Generic Pipe serves every lgos/ model, even while it is disabled or
    # its listing fails.
    async def fetch_openai_models(
        request: object, user: object = None
    ) -> list[dict[str, Any]]:
        assert (request, user) == ("request", "user")
        return [
            {"id": "lgos/simple-graph", "name": "lgos/simple-graph"},
            {"id": "openai/gpt-4.1-mini", "name": "openai/gpt-4.1-mini"},
        ]

    fetch = without_lgos_models(fetch_openai_models)

    assert await fetch("request", "user") == [
        {"id": "openai/gpt-4.1-mini", "name": "openai/gpt-4.1-mini"}
    ]
