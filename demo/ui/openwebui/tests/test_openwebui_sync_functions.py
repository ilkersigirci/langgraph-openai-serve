from pathlib import Path
from unittest.mock import MagicMock, Mock, call

import httpx
import pytest

import lgos_openwebui.sync_functions as sync_functions_module
from lgos_openwebui.bundle import bundle_function
from lgos_openwebui.sync_functions import (
    FUNCTIONS_DIR,
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


def test_bundle_function_is_frontmatter_first_and_executable() -> None:
    content = bundle_function(FUNCTIONS_DIR / "generic")
    namespace: dict[str, object] = {}

    exec(compile(content, "<generic>", "exec"), namespace)

    assert content.startswith('"""\ntitle: Generic\n')
    assert "from .api import" not in content
    assert "# ===== BEGIN contracts.py =====" in content
    assert "# ===== BEGIN files.py =====" in content
    assert "# ===== BEGIN pipe.py =====" in content
    assert "Pipe" in namespace


def test_discover_function_specs_includes_directory_backed_functions() -> None:
    specs = discover_function_specs()

    generic = next(spec for spec in specs if spec.id == "generic")

    assert generic.name == "Generic"
    assert generic.content.startswith('"""\ntitle: Generic\n')
    assert "# ===== BEGIN pipe.py =====" in generic.content


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


def test_sync_functions_preserves_unrelated_functions() -> None:
    client = _client(
        [
            {"id": "uservalues_simple", "name": "Retired", "content": "old"},
            {"id": "unrelated", "name": "Keep", "content": "external"},
        ]
    )

    results = sync_functions(client, (_spec(),))

    assert results == {"demo_pipe": "created"}
    client.delete.assert_not_called()


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


@pytest.mark.parametrize("server_error", [None, "No module named 'plotly'"])
def test_main_reads_demo_openwebui_environment(
    monkeypatch,
    server_error: str | None,
) -> None:
    monkeypatch.setenv("DEMO_OPENWEBUI_URL", "https://openwebui.example")
    monkeypatch.setenv("DEMO_OPENWEBUI_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("DEMO_OPENWEBUI_ADMIN_PASSWORD", "password")
    monkeypatch.setenv("OPENAI_GATEWAY_TYPE", "bifrost")
    monkeypatch.setenv(
        "DEMO_OPENWEBUI_OPENAI_GATEWAY_BASE_URL",
        "https://bifrost.example",
    )
    monkeypatch.setenv("DEMO_OPENWEBUI_API_KEY", "api-key")
    client = Mock()
    client_context = MagicMock()
    client_context.__enter__.return_value = client
    client_factory = Mock(return_value=client_context)
    catalog_client = Mock()
    catalog_context = MagicMock()
    catalog_context.__enter__.return_value = catalog_client
    api_client = Mock()
    api_context = MagicMock()
    api_context.__enter__.return_value = api_client
    openai_contexts = {
        "https://bifrost.example/v1": catalog_context,
        "https://bifrost.example/openai_passthrough/v1": api_context,
    }
    openai_factory = Mock(
        side_effect=lambda *, base_url, **_: openai_contexts[base_url]
    )
    sign_in_mock = Mock()
    sync_functions_mock = Mock(return_value={})
    if server_error is not None:
        response = httpx.Response(
            400,
            json={"detail": server_error},
            request=httpx.Request(
                "POST", "https://openwebui.example/api/v1/functions/id/generic/update"
            ),
        )
        sync_functions_mock.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=response.request, response=response
        )
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

    if server_error is not None:
        with pytest.raises(SystemExit, match=server_error):
            sync_functions_module.main()
        return

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
        api_client,
        provider_routing=True,
        model_prefixes=(),
    )
    sync_workspace_models_mock.assert_called_once_with(
        client,
        model_specs,
    )
