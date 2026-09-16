import ast
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import httpx2
import pytest
from openai import OpenAI

import lgos_openwebui.sync_functions as sync_functions_module
from lgos_openwebui.bundle import GENERIC_BUNDLE, bundle_function
from lgos_openwebui.functions.generic.gateway import gateway_config
from lgos_openwebui.sync_functions import (
    FUNCTIONS_DIR,
    FunctionSpec,
    discover_function_specs,
    sign_in,
    sync_functions,
)
from lgos_openwebui.workspace_models import WorkspaceModelSpec


def _response(data: object) -> httpx2.Response:
    return httpx2.Response(
        200,
        json=data,
        request=httpx2.Request("GET", "http://open-webui.test/api"),
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


def test_generic_bundle_modules_have_unique_top_level_definitions() -> None:
    definitions: dict[str, str] = {}

    for module_name in GENERIC_BUNDLE:
        source = FUNCTIONS_DIR / "generic" / module_name
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in tree.body:
            if not isinstance(
                node,
                (
                    ast.Assign,
                    ast.AnnAssign,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.FunctionDef,
                ),
            ):
                continue
            if isinstance(node, ast.Assign):
                names = [
                    target.id for target in node.targets if isinstance(target, ast.Name)
                ]
            elif isinstance(node, ast.AnnAssign):
                names = [node.target.id] if isinstance(node.target, ast.Name) else []
            else:
                names = [node.name]

            for name in names:
                assert name not in definitions, (
                    f"{name!r} is defined by both {definitions[name]} and {module_name}"
                )
                definitions[name] = module_name


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


@pytest.mark.parametrize("catalog_status", [200, 503])
def test_catalog_failure_does_not_modify_openwebui(
    monkeypatch: pytest.MonkeyPatch, gateway_environment, catalog_status: int
) -> None:
    requests: list[str] = []

    def respond_openwebui(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        assert request.url.path == "/api/v1/auths/signin"
        return httpx2.Response(200, json={"token": "admin-token"})

    def respond_gateway(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        assert request.url.path == "/model/info"
        # Both unavailable and malformed catalogs must abort before writes.
        return httpx2.Response(catalog_status, json={})

    with (
        closing(
            httpx2.Client(
                base_url="http://openwebui.test",
                transport=httpx2.MockTransport(respond_openwebui),
            )
        ) as client,
        closing(
            OpenAI(
                api_key="test-api-key",
                http_client=httpx2.Client(
                    transport=httpx2.MockTransport(respond_gateway)
                ),
                max_retries=0,
            )
        ) as gateway,
    ):
        monkeypatch.setattr(
            sync_functions_module,
            "httpx2",
            SimpleNamespace(
                Client=lambda **_: client,
                HTTPError=httpx2.HTTPError,
                HTTPStatusError=httpx2.HTTPStatusError,
            ),
        )
        monkeypatch.setattr(sync_functions_module, "OpenAI", lambda **_: gateway)

        with pytest.raises(SystemExit, match="Open WebUI sync failed"):
            sync_functions_module.main()

    assert requests == ["/api/v1/auths/signin", "/model/info"]


@pytest.mark.parametrize("server_error", [None, "No module named 'plotly'"])
def test_main_reads_demo_openwebui_environment(
    monkeypatch,
    server_error: str | None,
) -> None:
    monkeypatch.setenv("DEMO_OPENWEBUI_URL", "https://openwebui.example")
    monkeypatch.setenv("DEMO_OPENWEBUI_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("DEMO_OPENWEBUI_ADMIN_PASSWORD", "password")
    monkeypatch.setenv("OPENAI_GATEWAY_TYPE", "bifrost")
    monkeypatch.setenv("OPENAI_GATEWAY_BASE_URL", "https://bifrost.example")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "api-key")
    client = Mock()
    client_context = MagicMock()
    client_context.__enter__.return_value = client
    client_factory = Mock(return_value=client_context)
    catalog_client = Mock()
    catalog_context = MagicMock()
    catalog_context.__enter__.return_value = catalog_client
    openai_contexts = {
        "https://bifrost.example/v1": catalog_context,
    }
    openai_factory = Mock(
        side_effect=lambda *, base_url, **_: openai_contexts[base_url]
    )
    sign_in_mock = Mock()
    sync_functions_mock = Mock(return_value={})
    sync_mcp_mock = Mock(return_value="created")
    if server_error is not None:
        response = httpx2.Response(
            400,
            json={"detail": server_error},
            request=httpx2.Request(
                "POST", "https://openwebui.example/api/v1/functions/id/generic/update"
            ),
        )
        sync_functions_mock.side_effect = httpx2.HTTPStatusError(
            "400 Bad Request", request=response.request, response=response
        )
    model_specs = (WorkspaceModelSpec(id="plain", fields=()),)
    discover_workspace_models_mock = Mock(return_value=model_specs)
    sync_workspace_models_mock = Mock()
    monkeypatch.setattr(sync_functions_module.httpx2, "Client", client_factory)
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
    monkeypatch.setattr(
        sync_functions_module,
        "sync_mcp_gateway",
        sync_mcp_mock,
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
    sync_mcp_mock.assert_called_once_with(
        client,
        gateway=gateway_config("bifrost", "https://bifrost.example"),
        api_key="api-key",
    )
    openai_factory.assert_called_once_with(
        base_url="https://bifrost.example/v1",
        api_key="api-key",
        timeout=10,
    )
    discover_workspace_models_mock.assert_called_once_with(
        catalog_client,
        gateway=gateway_config("bifrost", "https://bifrost.example"),
    )
    sync_workspace_models_mock.assert_called_once_with(
        client,
        model_specs,
    )
