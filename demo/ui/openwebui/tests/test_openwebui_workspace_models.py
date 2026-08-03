from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest

from lgos_openwebui.workspace_models import (
    WorkspaceModelSpec,
    chat_variable_fields,
    discover_workspace_model_specs,
    sync_workspace_models,
)

PUBLIC_READ_GRANT = {
    "principal_type": "user",
    "principal_id": "*",
    "permission": "read",
}


def _response(data: object) -> httpx.Response:
    return httpx.Response(
        200,
        json=data,
        request=httpx.Request("GET", "http://open-webui.test/api"),
    )


def _client(exported: object) -> Mock:
    client = Mock()
    client.get.return_value = _response(exported)
    client.post.return_value = _response({})
    return client


def test_chat_variable_fields_reuses_the_chainlit_scalar_subset() -> None:
    model = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
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


def test_discover_workspace_models_uses_one_openai_client() -> None:
    configured = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "description": "  DUMMY  ",
                "features": [],
            }
        }
    )
    client = Mock()
    client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="plain"),
            SimpleNamespace(id="simple-graph"),
        ]
    )
    client.models.retrieve.return_value = configured

    specs = discover_workspace_model_specs(client)

    assert specs == (
        WorkspaceModelSpec(
            id="plain",
            fields=(),
            description="DUMMY",
        ),
        WorkspaceModelSpec(
            id="simple-graph",
            fields=(),
            description="DUMMY",
        ),
    )
    assert client.models.retrieve.call_args_list == [
        call(model="plain"),
        call(model="simple-graph"),
    ]


def test_discover_workspace_models_uses_explicit_routes() -> None:
    configured = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "description": "DUMMY",
                "features": [],
            }
        }
    )
    client = Mock()
    client.models.list.side_effect = [
        SimpleNamespace(data=[SimpleNamespace(id="graph-a")]),
        SimpleNamespace(data=[SimpleNamespace(id="graph-b")]),
    ]
    client.models.retrieve.return_value = configured

    specs = discover_workspace_model_specs(
        client,
        {
            "lgos-a": {"x-route": "a"},
            "lgos-b": {"x-route": "b"},
        },
    )

    assert specs == (
        WorkspaceModelSpec(
            id="lgos-a/graph-a",
            fields=(),
            description="DUMMY",
        ),
        WorkspaceModelSpec(
            id="lgos-b/graph-b",
            fields=(),
            description="DUMMY",
        ),
    )
    assert client.models.list.call_args_list == [
        call(extra_headers={"x-route": "a"}),
        call(extra_headers={"x-route": "b"}),
    ]
    assert client.models.retrieve.call_args_list == [
        call(
            model="graph-a",
            extra_headers={"x-route": "a"},
        ),
        call(
            model="graph-b",
            extra_headers={"x-route": "b"},
        ),
    ]


def test_discover_workspace_models_keeps_limited_models_visible() -> None:
    client = Mock()
    client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id="proxy-model")]
    )
    client.models.retrieve.return_value = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "features": [],
            }
        }
    )

    specs = discover_workspace_model_specs(client)

    assert specs == (WorkspaceModelSpec(id="proxy-model", fields=()),)


def test_discover_workspace_models_preserves_standard_catalog_ids() -> None:
    client = Mock()
    catalog = SimpleNamespace(
        data=[
            SimpleNamespace(id="lgos-a/graph-a"),
            SimpleNamespace(id="lgos-b/graph-b"),
        ]
    )
    client.models.list.return_value = catalog
    client.models.retrieve.return_value = "unsupported model detail"

    specs = discover_workspace_model_specs(client)

    assert specs == (
        WorkspaceModelSpec(id="lgos-a/graph-a", fields=()),
        WorkspaceModelSpec(id="lgos-b/graph-b", fields=()),
    )
    assert client.models.retrieve.call_args_list == [
        call(model="lgos-a/graph-a"),
        call(model="lgos-b/graph-b"),
    ]


def test_workspace_model_spec_rejects_oversized_openwebui_ids() -> None:
    WorkspaceModelSpec(id="x" * 248, fields=())

    with pytest.raises(ValueError, match="too long for Open WebUI"):
        WorkspaceModelSpec(id="x" * 249, fields=())


def test_sync_workspace_models_skips_an_empty_catalog() -> None:
    client = Mock()

    sync_workspace_models(client, ())

    client.get.assert_not_called()
    client.post.assert_not_called()


def test_sync_workspace_models_bulk_imports_base_and_new_wrapper() -> None:
    client = _client([])
    spec = WorkspaceModelSpec(
        id="simple-graph",
        description="DUMMY",
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

    client.get.assert_called_once_with("/api/v1/models/export")
    client.post.assert_called_once()
    base, wrapper = client.post.call_args.kwargs["json"]["models"]
    assert client.post.call_args.args == ("/api/v1/models/import",)
    assert [base["id"], wrapper["id"]] == [
        "generic.simple-graph",
        "lgos.simple-graph",
    ]
    assert base["base_model_id"] is None
    assert base["access_grants"] == [PUBLIC_READ_GRANT]
    assert base["is_active"] is False
    assert wrapper["base_model_id"] == base["id"]
    assert wrapper["access_grants"] == [PUBLIC_READ_GRANT]
    assert wrapper["meta"]["description"] == "DUMMY"
    assert wrapper["meta"]["chat_variables_schema"] == {"fields": list(spec.fields)}


def test_limited_workspace_model_has_a_warning_and_description_fallback() -> None:
    client = _client([])
    spec = WorkspaceModelSpec(id="proxy-model", fields=())

    sync_workspace_models(client, (spec,))

    _, wrapper = client.post.call_args.kwargs["json"]["models"]
    assert "Limited functionality" in wrapper["name"]
    assert "Limited functionality" in wrapper["meta"]["description"]


def test_sync_workspace_models_leaves_existing_wrapper_state_to_openwebui() -> None:
    client = _client([{"id": "lgos.plain"}])

    sync_workspace_models(
        client,
        (WorkspaceModelSpec(id="plain", fields=()),),
    )

    base, wrapper = client.post.call_args.kwargs["json"]["models"]
    assert base["access_grants"] == [PUBLIC_READ_GRANT]
    assert base["is_active"] is False
    assert "Limited functionality" in wrapper["meta"]["description"]
    assert "access_grants" not in wrapper
    assert "is_active" not in wrapper
