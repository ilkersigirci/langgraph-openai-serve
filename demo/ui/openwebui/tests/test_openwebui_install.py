"""Bundled Functions and their installation through Open WebUI's API."""

import ast

import httpx
import pytest

from lgos_openwebui.bundle import GENERIC_BUNDLE, bundle_function
from lgos_openwebui.install import FUNCTIONS_DIR, function_specs, install_functions
from tests.support import FakeOpenWebUI


def test_bundle_function_is_frontmatter_first_and_executable() -> None:
    content = bundle_function(FUNCTIONS_DIR / "generic")
    namespace: dict[str, object] = {}

    exec(compile(content, "<generic>", "exec"), namespace)  # ruff: ignore[exec-builtin] - Execute the repository-owned bundle to test deployed behavior.

    assert content.startswith('"""\ntitle: Generic\n')
    assert "from .gateway import" not in content
    assert "# ===== BEGIN contracts.py =====" in content
    assert "# ===== BEGIN catalog.py =====" in content
    assert "# ===== BEGIN pipe.py =====" in content
    assert "Pipe" in namespace


def test_generic_bundle_modules_have_unique_top_level_definitions() -> None:
    definitions: dict[str, str] = {}

    for module_name in GENERIC_BUNDLE:
        source = FUNCTIONS_DIR / "generic" / module_name
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [
                    target.id for target in node.targets if isinstance(target, ast.Name)
                ]
            elif isinstance(node, ast.AnnAssign):
                names = [node.target.id] if isinstance(node.target, ast.Name) else []
            elif isinstance(
                node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)
            ):
                names = [node.name]
            else:
                continue

            for name in names:
                assert name not in definitions, (
                    f"{name!r} is defined by both {definitions[name]} and {module_name}"
                )
                definitions[name] = module_name


def test_function_names_match_their_frontmatter_titles() -> None:
    for spec in function_specs("generic", include_demo_functions=True):
        assert spec.content.startswith(f'"""\ntitle: {spec.name}\n')


async def test_install_creates_missing_and_updates_outdated_functions() -> None:
    generic = {
        "id": "generic",
        "name": "Old Generic",
        "content": "old content",
        "meta": {"description": "Edited by an admin"},
        "is_active": False,
    }
    unrelated = {"id": "unrelated", "name": "Keep", "content": "external"}
    openwebui = FakeOpenWebUI([generic, unrelated])
    specs = function_specs("generic", include_demo_functions=True)

    async with openwebui.client() as client:
        results = await install_functions(client, specs)

    contents = {spec.id: spec.content for spec in specs}
    assert results == {"generic": "updated", "uservalves_simple": "created"}
    assert openwebui.functions == {
        # An update keeps the admin's metadata and disabled state.
        "generic": {**generic, "name": "Generic", "content": contents["generic"]},
        "unrelated": unrelated,
        "uservalves_simple": {
            "id": "uservalves_simple",
            "name": "UserValves Simple",
            "content": contents["uservalves_simple"],
            "meta": {},
            "is_active": True,
        },
    }


async def test_install_merges_its_valves_and_writes_nothing_when_current() -> None:
    (spec,) = function_specs("generic", valves={"MODEL_PREFIX": "acme"})
    openwebui = FakeOpenWebUI(
        [
            {
                "id": "generic",
                "name": spec.name,
                "content": spec.content,
                "meta": {},
                "is_active": True,
                "valves": {"OPENAI_API_TIMEOUT": 73, "MODEL_PREFIX": "lgos"},
            }
        ]
    )

    async with openwebui.client() as client:
        assert await install_functions(client, (spec,)) == {"generic": "updated"}
        openwebui.writes.clear()
        assert await install_functions(client, (spec,)) == {"generic": "unchanged"}

    assert openwebui.writes == []
    # An admin's own valve settings stay.
    assert openwebui.functions["generic"]["valves"] == {
        "OPENAI_API_TIMEOUT": 73,
        "MODEL_PREFIX": "acme",
    }


async def test_rerun_completes_an_interrupted_first_install() -> None:
    openwebui = FakeOpenWebUI([])
    specs = function_specs("generic", valves={"MODEL_PREFIX": "acme"})

    async def disconnect(request: httpx.Request) -> None:
        if request.url.path.endswith("/valves/update"):
            message = "Connection lost"
            raise httpx.ConnectError(message, request=request)

    async with openwebui.client() as client:
        client.event_hooks["request"].append(disconnect)
        with pytest.raises(httpx.ConnectError):
            await install_functions(client, specs)
        client.event_hooks["request"].clear()
        assert await install_functions(client, specs) == {"generic": "updated"}

    assert openwebui.functions["generic"]["is_active"] is True
    assert openwebui.functions["generic"]["valves"] == {"MODEL_PREFIX": "acme"}


async def test_install_uses_the_lowercase_id_open_webui_stores() -> None:
    openwebui = FakeOpenWebUI([])
    specs = function_specs("Acme_Pipe", valves={"MODEL_PREFIX": "acme"})

    async with openwebui.client() as client:
        assert await install_functions(client, specs) == {"acme_pipe": "created"}
        assert await install_functions(client, specs) == {"acme_pipe": "unchanged"}

    assert openwebui.functions["acme_pipe"]["is_active"] is True
