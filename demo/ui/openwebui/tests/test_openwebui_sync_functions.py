from pathlib import Path
from unittest.mock import MagicMock, Mock, call

import httpx

import lgos_openwebui.sync_functions as sync_functions_module
from lgos_openwebui.sync_functions import (
    FunctionSpec,
    discover_function_specs,
    sign_in,
    sync_functions,
)
from lgos_openwebui.workspace_models import WorkspaceModelSpec


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


def _spec() -> FunctionSpec:
    return FunctionSpec(
        id="demo_pipe",
        name="Demo Pipe",
        content="class Pipe:\n    pass\n",
    )


def test_discover_function_specs_uses_filename_and_frontmatter(tmp_path: Path) -> None:
    source = tmp_path / "demo_pipe.py"
    source.write_text(
        '"""\ntitle: Demo: Pipe\nauthor: demo\n"""\n\nclass Pipe:\n    pass\n'
    )
    (tmp_path / "__init__.py").write_text("")

    specs = discover_function_specs(tmp_path)

    assert specs == (
        FunctionSpec(
            id="demo_pipe",
            name="Demo: Pipe",
            content=source.read_text(),
        ),
    )


def test_sync_functions_updates_existing_function_and_preserves_state() -> None:
    existing = {
        "id": "demo_pipe",
        "name": "Old Demo Pipe",
        "content": "old content",
        "meta": {"description": "Existing description", "custom": True},
        "is_active": False,
        "valves": {"API_KEY": "secret"},
    }
    client = _client([existing])

    results = sync_functions(client, (_spec(),))

    assert results == {"demo_pipe": "updated"}
    client.post.assert_called_once_with(
        "/api/v1/functions/id/demo_pipe/update",
        json={
            "id": "demo_pipe",
            "name": "Demo Pipe",
            "content": "class Pipe:\n    pass\n",
            "meta": existing["meta"],
        },
    )


def test_sync_functions_creates_and_enables_missing_function() -> None:
    client = _client([])

    results = sync_functions(client, (_spec(),))

    assert results == {"demo_pipe": "created"}
    client.post.assert_has_calls(
        [
            call(
                "/api/v1/functions/create",
                json={
                    "id": "demo_pipe",
                    "name": "Demo Pipe",
                    "content": "class Pipe:\n    pass\n",
                    "meta": {},
                },
            ),
            call("/api/v1/functions/id/demo_pipe/toggle"),
        ]
    )


def test_sync_functions_skips_unchanged_function() -> None:
    client = _client(
        [
            {
                "id": "demo_pipe",
                "name": "Demo Pipe",
                "content": "class Pipe:\n    pass\n",
                "meta": {},
            }
        ]
    )

    results = sync_functions(client, (_spec(),))

    assert results == {"demo_pipe": "unchanged"}
    client.post.assert_not_called()


def test_openwebui_client_signs_in_with_admin_credentials() -> None:
    client = Mock()
    client.headers = {}
    client.post.return_value = _response({"token": "jwt-token"})

    sign_in(client, "admin@example.com", "password")

    assert client.headers["Authorization"] == "Bearer jwt-token"
    client.post.assert_called_once_with(
        "/api/v1/auths/signin",
        json={"email": "admin@example.com", "password": "password"},
    )


def test_main_reads_demo_openwebui_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEMO_OPENWEBUI_URL", "https://openwebui.example")
    monkeypatch.setenv("DEMO_OPENWEBUI_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("DEMO_OPENWEBUI_ADMIN_PASSWORD", "password")
    monkeypatch.setenv(
        "DEMO_OPENWEBUI_OPENAI_BASE_URL",
        "https://bifrost.example/openai_passthrough/v1",
    )
    monkeypatch.setenv(
        "DEMO_OPENWEBUI_OPENAI_CATALOG_BASE_URL",
        "https://bifrost.example/v1",
    )
    monkeypatch.setenv("DEMO_OPENWEBUI_API_KEY", "api-key")
    client = Mock()
    client_context = MagicMock()
    client_context.__enter__.return_value = client
    client_factory = Mock(return_value=client_context)
    catalog_client = Mock()
    catalog_context = MagicMock()
    catalog_context.__enter__.return_value = catalog_client
    passthrough_client = Mock()
    passthrough_context = MagicMock()
    passthrough_context.__enter__.return_value = passthrough_client
    openai_contexts = {
        "https://bifrost.example/v1": catalog_context,
        "https://bifrost.example/openai_passthrough/v1": passthrough_context,
    }
    openai_factory = Mock(
        side_effect=lambda *, base_url, **_: openai_contexts[base_url]
    )
    sign_in_mock = Mock()
    sync_functions_mock = Mock(return_value={})
    model_specs = (WorkspaceModelSpec(id="plain", fields=()),)
    discover_workspace_models_mock = Mock(return_value=model_specs)
    sync_workspace_models_mock = Mock()
    monkeypatch.setattr(sync_functions_module.httpx, "Client", client_factory)
    monkeypatch.setattr(sync_functions_module, "OpenAI", openai_factory)
    monkeypatch.setattr(sync_functions_module, "sign_in", sign_in_mock)
    monkeypatch.setattr(
        sync_functions_module,
        "sync_functions",
        sync_functions_mock,
    )
    monkeypatch.setattr(
        sync_functions_module,
        "discover_workspace_model_specs",
        discover_workspace_models_mock,
    )
    monkeypatch.setattr(
        sync_functions_module,
        "sync_workspace_models",
        sync_workspace_models_mock,
    )

    sync_functions_module.main()

    client_factory.assert_called_once_with(
        base_url="https://openwebui.example",
        timeout=10,
    )
    sign_in_mock.assert_called_once_with(client, "admin@example.com", "password")
    sync_functions_mock.assert_called_once_with(client)
    openai_factory.assert_has_calls(
        [
            call(
                base_url="https://bifrost.example/v1",
                api_key="api-key",
                timeout=10,
            ),
            call(
                base_url="https://bifrost.example/openai_passthrough/v1",
                api_key="api-key",
                timeout=10,
            ),
        ],
        any_order=True,
    )
    assert openai_factory.call_count == 2
    discover_workspace_models_mock.assert_called_once_with(
        catalog_client,
        passthrough_client,
    )
    sync_workspace_models_mock.assert_called_once_with(
        client,
        model_specs,
    )
