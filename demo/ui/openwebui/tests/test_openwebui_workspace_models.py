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


def test_discover_workspace_models_retrieves_qualified_models() -> None:
    configured = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "client_settings": None,
            }
        }
    )
    catalog_client = Mock()
    catalog_client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="plain"),
            SimpleNamespace(id="lgos-a/simple-graph"),
        ]
    )
    inference_client = Mock()
    inference_client.models.retrieve.return_value = configured

    specs = discover_workspace_model_specs(catalog_client, inference_client)

    assert specs == (
        WorkspaceModelSpec(id="lgos-a/simple-graph", fields=()),
        WorkspaceModelSpec(id="plain", fields=()),
    )
    assert inference_client.models.retrieve.call_args_list == [
        call(model="plain"),
        call(
            model="simple-graph",
            extra_headers={"x-model-provider": "lgos-a"},
        ),
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
        id="lgos-a/simple-graph",
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
        "generic.lgos-a/simple-graph",
        "lgos.lgos-a/simple-graph",
    ]
    assert base["base_model_id"] is None
    assert base["access_grants"] == [PUBLIC_READ_GRANT]
    assert base["is_active"] is False
    assert wrapper["base_model_id"] == base["id"]
    assert wrapper["access_grants"] == [PUBLIC_READ_GRANT]
    assert wrapper["meta"]["chat_variables_schema"] == {"fields": list(spec.fields)}


def test_sync_workspace_models_leaves_existing_wrapper_state_to_openwebui() -> None:
    client = _client([{"id": "lgos.plain"}])

    sync_workspace_models(
        client,
        (WorkspaceModelSpec(id="plain", fields=()),),
    )

    base, wrapper = client.post.call_args.kwargs["json"]["models"]
    assert base["access_grants"] == [PUBLIC_READ_GRANT]
    assert base["is_active"] is False
    assert "access_grants" not in wrapper
    assert "is_active" not in wrapper
