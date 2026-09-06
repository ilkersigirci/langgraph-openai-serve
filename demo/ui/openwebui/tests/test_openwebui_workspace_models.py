from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest

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


def test_discover_workspace_models_uses_bifrost_catalog_and_native_api() -> None:
    configured = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "description": "  DUMMY  ",
                "features": ["file_inputs"],
            }
        }
    )
    catalog_client = Mock()
    catalog_client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(
                id="lgos-a/graph-a",
                owned_by="langgraph-openai-serve",
            ),
            SimpleNamespace(
                id="lgos-future/graph-b",
                owned_by="langgraph-openai-serve",
            ),
            SimpleNamespace(id="openai/gpt-5", owned_by="openai"),
        ]
    )
    api_client = Mock()
    api_client.models.retrieve.return_value = configured

    specs = discover_workspace_model_specs(
        catalog_client,
        api_client,
        provider_routing=True,
    )

    assert specs == (
        WorkspaceModelSpec(
            id="lgos-a/graph-a",
            fields=(),
            description="DUMMY",
            supports_file_inputs=True,
        ),
        WorkspaceModelSpec(
            id="lgos-future/graph-b",
            fields=(),
            description="DUMMY",
            supports_file_inputs=True,
        ),
    )
    catalog_client.models.list.assert_called_once_with()
    api_client.models.retrieve.assert_has_calls(
        [
            call(
                model="graph-a",
                extra_headers={"x-model-provider": "lgos-a"},
            ),
            call(
                model="graph-b",
                extra_headers={"x-model-provider": "lgos-future"},
            ),
        ],
        any_order=True,
    )
    assert api_client.models.retrieve.call_count == 2


def test_discover_workspace_models_keeps_limited_models_visible() -> None:
    catalog_client = Mock()
    catalog_client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(
                id="lgos-a/proxy-model",
                owned_by="langgraph-openai-serve",
            )
        ]
    )
    api_client = Mock()
    api_client.models.retrieve.return_value = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "features": [],
            }
        }
    )

    specs = discover_workspace_model_specs(
        catalog_client,
        api_client,
        provider_routing=True,
    )

    assert specs == (WorkspaceModelSpec(id="lgos-a/proxy-model", fields=()),)


def test_discover_workspace_models_merges_litellm_passthrough_catalogs() -> None:
    catalog_client = Mock()
    catalog_client.base_url = "https://gateway.example/v1/"
    catalog_clients = {}
    for model_prefix in ("lgos-a", "lgos-b"):
        model = SimpleNamespace(
            id="simple-graph",
            owned_by="langgraph-openai-serve",
            model_extra={
                "langgraph_openai_serve": {
                    "schema_version": 1,
                    "description": f"LiteLLM {model_prefix} graph",
                    "features": ["file_inputs"],
                }
            },
        )
        current_catalog = Mock()
        current_catalog.models.list.return_value = SimpleNamespace(data=[model])
        current_catalog.models.retrieve.return_value = model
        catalog_clients[f"https://gateway.example/v1/{model_prefix}"] = current_catalog
    catalog_client.with_options.side_effect = lambda *, base_url: catalog_clients[
        base_url
    ]
    api_client = Mock()

    specs = discover_workspace_model_specs(
        catalog_client,
        api_client,
        provider_routing=False,
        model_prefixes=("lgos-a", "lgos-b"),
    )

    assert specs == (
        WorkspaceModelSpec(
            id="lgos-a/simple-graph",
            fields=(),
            description="LiteLLM lgos-a graph",
            supports_file_inputs=True,
        ),
        WorkspaceModelSpec(
            id="lgos-b/simple-graph",
            fields=(),
            description="LiteLLM lgos-b graph",
            supports_file_inputs=True,
        ),
    )
    catalog_client.with_options.assert_has_calls(
        [
            call(base_url="https://gateway.example/v1/lgos-a"),
            call(base_url="https://gateway.example/v1/lgos-b"),
        ]
    )
    for current_catalog in catalog_clients.values():
        current_catalog.models.retrieve.assert_called_once_with(model="simple-graph")
    api_client.models.retrieve.assert_not_called()


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
