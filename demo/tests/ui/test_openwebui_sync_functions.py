from pathlib import Path
from unittest.mock import Mock, call

import httpx
from demo.ui.openwebui.sync_functions import (
    FunctionSpec,
    discover_function_specs,
    sign_in,
    sync_functions,
)


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
