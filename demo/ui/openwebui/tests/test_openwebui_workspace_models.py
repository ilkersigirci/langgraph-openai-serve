from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest
from openai import OpenAI
from openai.types import Model

from lgos_openwebui.functions.generic.gateway import gateway_config
from lgos_openwebui.workspace_models import (
    PUBLIC_READ_GRANT,
    WorkspaceModelSpec,
    chat_variable_fields,
    discover_workspace_model_specs,
    sync_workspace_models,
)


def _response(data: object) -> httpx.Response:
    return httpx.Response(
        200,
        json=data,
        request=httpx.Request("GET", "http://open-webui.test/api"),
    )


def _client(exported: object, base_models: object = ()) -> Mock:
    client = Mock()

    def get(path: str) -> httpx.Response:
        responses = {
            "/api/v1/models/export": exported,
            "/api/v1/models/base": base_models,
        }
        if path not in responses:
            msg = f"Unexpected Open WebUI read: {path}"
            raise AssertionError(msg)
        return _response(responses[path])

    client.get.side_effect = get
    client.post.return_value = _response({})
    return client


def _assert_workspace_reads(client: Mock) -> None:
    client.get.assert_has_calls(
        [
            call("/api/v1/models/export"),
            call("/api/v1/models/base"),
        ],
        any_order=True,
    )
    assert client.get.call_count == 2


def test_chat_variable_fields_reuses_the_chainlit_scalar_subset() -> None:
    model = SimpleNamespace(
        model_extra={
            "lgos": {
                "schema_version": 1,
                "description": "DUMMY",
                "features": [],
                "client_settings": {
                    "schema_version": 1,
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "use_history": {
                                "type": "boolean",
                                "title": "Use conversation history",
                            },
                            "mode": {
                                "type": "string",
                                "title": "Mode",
                                "enum": ["brief", "detailed"],
                            },
                            "assistant_name": {
                                "type": "string",
                                "title": "Assistant name",
                            },
                            "retries": {"type": "integer"},
                        },
                    },
                    "defaults": {
                        "use_history": False,
                        "mode": "brief",
                        "assistant_name": "Helper",
                        "retries": 3,
                    },
                },
            }
        }
    )

    assert chat_variable_fields(model) == (
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
            "options": ["brief", "detailed"],
            "default": "brief",
        },
        {
            "key": "assistant_name",
            "type": "text",
            "label": "Assistant name",
            "default": "Helper",
        },
    )
    assert chat_variable_fields(SimpleNamespace(model_extra={})) is None


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
        ("invalid", None, "value"),
    ],
)
def test_chat_variable_fields_omits_invalid_fields_without_losing_valid_ones(
    name: str, schema: object, default: object
) -> None:
    model = Model.model_validate(
        {
            "id": "test-graph",
            "object": "model",
            "created": 1,
            "owned_by": "langgraph-openai-serve",
            "lgos": {
                "schema_version": 1,
                "description": "Test graph",
                "features": [],
                "client_settings": {
                    "schema_version": 1,
                    "json_schema": {
                        "properties": {"enabled": {"type": "boolean"}, name: schema}
                    },
                    "defaults": {"enabled": False, name: default},
                },
            },
        }
    )

    assert chat_variable_fields(model) == (
        {"key": "enabled", "label": "Enabled", "default": False, "type": "checkbox"},
    )


@pytest.mark.parametrize("provider_routing", [True, False], ids=["bifrost", "litellm"])
def test_discovery_projects_settings_from_gateway_model_details(
    provider_routing: bool,
) -> None:
    graph = Model(
        id="simple-graph",
        object="model",
        created=1,
        owned_by="langgraph-openai-serve",
        lgos={
            "schema_version": 1,
            "description": "  Simple graph  ",
            "features": ["file_inputs"],
            "client_settings": {
                "schema_version": 1,
                "json_schema": {"properties": {"enabled": {"type": "boolean"}}},
                "defaults": {"enabled": False},
            },
        },
    )
    other_model = Model(id="gpt-5", object="model", created=1, owned_by="openai")
    providers = ("lgos-a", "lgos-future")
    responses: dict[tuple[str, str | None], object] = {}
    deployments = []
    for provider in providers:
        detail = graph.model_dump()
        if provider == "lgos-future":
            detail["lgos"]["client_settings"] = {
                "schema_version": 1,
                "json_schema": {
                    "properties": {
                        "audience": {"type": "string", "enum": ["general", "expert"]}
                    }
                },
                "defaults": {"audience": "general"},
            }
        if provider_routing:
            responses[("/openai_passthrough/v1/models/simple-graph", provider)] = detail
        else:
            deployments.append(
                {
                    "model_name": f"{provider}/simple-graph",
                    "model_info": {"lgos": detail["lgos"]},
                }
            )
    responses[("/model/info", None)] = {
        "data": [
            *deployments,
            {"model_name": "gpt-5", "model_info": {}},
        ]
    }
    if provider_routing:
        responses[("/v1/models", None)] = {
            "object": "list",
            "data": [
                {
                    **graph.model_dump(exclude={"lgos"}),
                    "id": "lgos-future/simple-graph",
                },
                {**graph.model_dump(exclude={"lgos"}), "id": "lgos-a/simple-graph"},
                other_model.model_dump(),
            ],
        }

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        key = (request.url.path, request.headers.get("x-model-provider"))
        return httpx.Response(200, json=responses[key])

    with OpenAI(
        base_url="https://gateway.example/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        specs = discover_workspace_model_specs(
            client,
            gateway=gateway_config(
                "bifrost" if provider_routing else "litellm", "https://gateway.example"
            ),
        )

    assert [spec.id for spec in specs] == [
        "lgos-a/simple-graph",
        "lgos-future/simple-graph",
    ]
    for spec in specs:
        assert spec.description == "Simple graph"
        assert spec.supports_file_inputs is True
    assert specs[0].fields == (
        {"key": "enabled", "type": "checkbox", "label": "Enabled", "default": False},
    )
    assert specs[1].fields == (
        {
            "key": "audience",
            "type": "select",
            "label": "Audience",
            "options": ["general", "expert"],
            "default": "general",
        },
    )


def test_discover_workspace_models_keeps_limited_models_visible() -> None:
    with OpenAI(
        base_url="https://gateway.example/v1",
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "model_name": "lgos-a/proxy-model",
                                "model_info": {
                                    "lgos": {"schema_version": 1, "features": []}
                                },
                            }
                        ]
                    },
                )
            )
        ),
    ) as client:
        specs = discover_workspace_model_specs(
            client, gateway=gateway_config("litellm", "https://gateway.example")
        )

    assert specs == (WorkspaceModelSpec(id="lgos-a/proxy-model", fields=()),)


def test_workspace_model_spec_rejects_oversized_openwebui_ids() -> None:
    WorkspaceModelSpec(id="x" * 248, fields=())

    with pytest.raises(ValueError, match="too long for Open WebUI"):
        WorkspaceModelSpec(id="x" * 249, fields=())


def test_sync_workspace_models_removes_generated_models_for_an_empty_catalog() -> None:
    client = _client(
        [
            {
                "id": "lgos.old-graph",
                "base_model_id": "generic.old-graph",
            },
            {
                "id": "user-model",
                "base_model_id": None,
            },
        ],
        [
            {
                "id": "generic.old-graph",
                "base_model_id": None,
            }
        ],
    )

    sync_workspace_models(client, ())

    _assert_workspace_reads(client)
    assert client.post.call_args_list == [
        call(
            "/api/v1/models/model/delete",
            json={"id": "lgos.old-graph"},
        ),
        call(
            "/api/v1/models/model/delete",
            json={"id": "generic.old-graph"},
        ),
    ]


def test_sync_workspace_models_keeps_unrelated_models() -> None:
    client = _client(
        [
            {
                "id": "preset",
                "base_model_id": "openai.gpt-5",
            },
        ],
        [
            {
                "id": "user-model",
                "base_model_id": None,
            }
        ],
    )

    sync_workspace_models(client, ())

    client.post.assert_not_called()


def test_sync_workspace_models_imports_hidden_base_and_new_wrapper() -> None:
    client = _client([])
    spec = WorkspaceModelSpec(
        id="simple-graph",
        description="DUMMY",
        supports_file_inputs=True,
        fields=(
            {
                "key": "use_history",
                "type": "checkbox",
                "label": "Use history",
                "default": False,
            },
        ),
    )

    sync_workspace_models(client, (spec,))

    _assert_workspace_reads(client)
    client.post.assert_called_once()
    base, wrapper = client.post.call_args.kwargs["json"]["models"]
    assert client.post.call_args.args == ("/api/v1/models/import",)
    assert base == {
        "id": "generic.simple-graph",
        "base_model_id": None,
        "name": "Generic / simple-graph",
        "meta": {"hidden": True},
        "params": {},
        "access_grants": [PUBLIC_READ_GRANT],
        "is_active": True,
    }
    assert wrapper["id"] == "lgos.simple-graph"
    assert wrapper["base_model_id"] == base["id"]
    assert wrapper["access_grants"] == [PUBLIC_READ_GRANT]
    assert wrapper["meta"]["description"] == "DUMMY"
    assert wrapper["meta"]["chat_variables_schema"] == {"fields": list(spec.fields)}
    assert wrapper["meta"]["capabilities"] == {
        "file_upload": True,
        "file_context": False,
    }
    assert wrapper["meta"]["builtinTools"] == {"files": False}


def test_hosted_tool_workspace_model_has_fixed_chat_controls() -> None:
    client = _client([])
    spec = WorkspaceModelSpec(
        id="lgos-a/hosted-tool",
        description="Hosted tools",
        fields=(),
    )

    sync_workspace_models(client, (spec,))

    _, wrapper = client.post.call_args.kwargs["json"]["models"]
    assert wrapper["meta"]["chat_variables_schema"]["fields"] == [
        {
            "key": "lgos_current_time",
            "type": "checkbox",
            "label": "Current time",
            "default": False,
        },
        {
            "key": "web_search",
            "type": "checkbox",
            "label": "Web search",
            "default": False,
        },
    ]


def test_limited_workspace_model_has_a_warning_and_description_fallback() -> None:
    client = _client([])
    spec = WorkspaceModelSpec(id="proxy-model", fields=())

    sync_workspace_models(client, (spec,))

    _, wrapper = client.post.call_args.kwargs["json"]["models"]
    assert "Limited functionality" in wrapper["name"]
    assert "Limited functionality" in wrapper["meta"]["description"]
    assert wrapper["meta"]["capabilities"]["file_upload"] is False


@pytest.mark.parametrize("existing", [False, True])
def test_simple_uservalves_model_reuses_pipe_without_chat_variable_controls(
    existing: bool,
) -> None:
    client = _client([{"id": "lgos.uservalves_simple"}] if existing else [])
    spec = WorkspaceModelSpec(
        id="lgos-a/simple-graph",
        description="Simple graph",
        fields=({"key": "use_history", "type": "checkbox", "default": False},),
    )

    sync_workspace_models(client, (spec,))

    base, dynamic, static = client.post.call_args.kwargs["json"]["models"]
    assert static["id"] == "lgos.uservalves_simple"
    assert static["base_model_id"] == dynamic["base_model_id"] == base["id"]
    assert static["meta"]["filterIds"] == ["uservalves_simple"]
    assert static["meta"]["chat_variables_schema"] == {"fields": []}
    assert "filterIds" not in dynamic["meta"]
    assert dynamic["meta"]["chat_variables_schema"]["fields"] == list(spec.fields)
    assert ("access_grants" in static) is not existing
    assert "is_active" not in static


def test_sync_workspace_models_leaves_existing_wrapper_state_to_openwebui() -> None:
    client = _client(
        [
            {"id": "lgos.plain"},
        ],
        [
            {
                "id": "generic.plain",
                "base_model_id": None,
                "is_active": False,
            },
        ],
    )

    sync_workspace_models(
        client,
        (WorkspaceModelSpec(id="plain", fields=()),),
    )

    base, wrapper = client.post.call_args.kwargs["json"]["models"]
    assert base["meta"] == {"hidden": True}
    assert base["access_grants"] == [PUBLIC_READ_GRANT]
    assert base["is_active"] is True
    assert "Limited functionality" in wrapper["meta"]["description"]
    assert "access_grants" not in wrapper
    assert "is_active" not in wrapper
