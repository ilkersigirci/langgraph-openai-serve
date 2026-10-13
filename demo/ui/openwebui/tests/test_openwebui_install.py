"""Bundled Functions and their installation through Open WebUI's API."""

import ast
from typing import Any

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from lgos_openwebui.bundle import GENERIC_BUNDLE, bundle_function
from lgos_openwebui.install import FUNCTIONS_DIR, function_specs, install_functions

ADMIN_TOKEN = "admin-token"


class FakeOpenWebUI(Starlette):
    """Open WebUI's Functions routes, served in-process."""

    def __init__(
        self, functions: list[dict[str, Any]], *, rejection: str | None = None
    ) -> None:
        self.functions = {function["id"]: function for function in functions}
        self.rejection = rejection
        self.writes: list[str] = []
        super().__init__(
            routes=[
                Route("/api/v1/functions/export", self.export),
                Route("/api/v1/functions/{action:path}", self.write, methods=["POST"]),
            ]
        )

    async def export(self, request: Request) -> JSONResponse:
        if unauthorized := self.unauthorized(request):
            return unauthorized
        return JSONResponse(list(self.functions.values()))

    async def write(self, request: Request) -> JSONResponse:
        if unauthorized := self.unauthorized(request):
            return unauthorized
        self.writes.append(request.url.path)
        if self.rejection is not None:
            return JSONResponse({"detail": self.rejection}, status_code=400)
        action = request.path_params["action"]
        if action == "create":
            payload = await request.json()
            # Open WebUI creates Functions disabled.
            self.functions[payload["id"]] = {**payload, "is_active": False}
        elif action.endswith("/toggle"):
            function = self.functions[action.split("/")[-2]]
            function["is_active"] = not function["is_active"]
        elif action.endswith("/update"):
            payload = await request.json()
            self.functions[payload["id"]].update(
                name=payload["name"], content=payload["content"], meta=payload["meta"]
            )
        return JSONResponse({})

    @staticmethod
    def unauthorized(request: Request) -> JSONResponse | None:
        if request.headers.get("authorization") == f"Bearer {ADMIN_TOKEN}":
            return None
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)


def test_bundle_function_is_frontmatter_first_and_executable() -> None:
    content = bundle_function(FUNCTIONS_DIR / "generic")
    namespace: dict[str, object] = {}

    exec(compile(content, "<generic>", "exec"), namespace)  # ruff: ignore[exec-builtin] - Execute the repository-owned bundle to test deployed behavior.

    assert content.startswith('"""\ntitle: Generic\n')
    assert "from .api import" not in content
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
    for spec in function_specs():
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

    results = await install_functions(openwebui, ADMIN_TOKEN)

    specs = {spec.id: spec for spec in function_specs()}
    assert results == {"generic": "updated", "uservalves_simple": "created"}
    assert openwebui.functions == {
        # An update keeps the admin's metadata and disabled state.
        "generic": {
            **generic,
            "name": "Generic",
            "content": specs["generic"].content,
        },
        "unrelated": unrelated,
        "uservalves_simple": {
            "id": "uservalves_simple",
            "name": "UserValves Simple",
            "content": specs["uservalves_simple"].content,
            "meta": {},
            "is_active": True,
        },
    }


async def test_install_writes_nothing_when_functions_are_current() -> None:
    openwebui = FakeOpenWebUI(
        [
            {"id": spec.id, "name": spec.name, "content": spec.content, "meta": {}}
            for spec in function_specs()
        ]
    )

    results = await install_functions(openwebui, ADMIN_TOKEN)

    assert results == {"generic": "unchanged", "uservalves_simple": "unchanged"}
    assert openwebui.writes == []


async def test_install_reports_why_open_webui_rejected_a_function() -> None:
    openwebui = FakeOpenWebUI([], rejection="No module named 'plotly'")

    with pytest.raises(RuntimeError, match="No module named 'plotly'"):
        await install_functions(openwebui, ADMIN_TOKEN)
