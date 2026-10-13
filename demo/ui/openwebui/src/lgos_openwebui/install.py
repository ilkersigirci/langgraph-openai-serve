"""Install the demo Functions through Open WebUI's own API."""

from dataclasses import dataclass
from pathlib import Path

import httpx
from starlette.types import ASGIApp

from .bundle import bundle_function

FUNCTIONS_DIR = Path(__file__).with_name("functions")


@dataclass(frozen=True)
class FunctionSpec:
    """Describe a bundled Open WebUI Function."""

    id: str
    name: str
    content: str


def function_specs() -> tuple[FunctionSpec, ...]:
    """Return the demo's Open WebUI Functions."""
    return (
        FunctionSpec(
            id="generic",
            name="Generic",
            content=bundle_function(FUNCTIONS_DIR / "generic"),
        ),
        FunctionSpec(
            id="uservalves_simple",
            name="UserValves Simple",
            content=(FUNCTIONS_DIR / "uservalves_simple.py").read_text(
                encoding="utf-8"
            ),
        ),
    )


async def install_functions(app: ASGIApp, token: str) -> dict[str, str]:
    """
    Create or update the Functions in a started Open WebUI app.

    Requests run in-process with an admin's ``token``, before Open WebUI accepts
    connections. Unrelated Functions and the state of existing ones are preserved.
    """
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://open-webui",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        try:
            return await _install(client, function_specs())
        except httpx.HTTPStatusError as exc:
            # Open WebUI explains failed imports, such as a missing package, in
            # the response body.
            msg = f"Installing the Open WebUI Functions failed: {exc}\n{exc.response.text}"
            raise RuntimeError(msg) from exc


async def _install(
    client: httpx.AsyncClient, specs: tuple[FunctionSpec, ...]
) -> dict[str, str]:
    exported = await client.get("/api/v1/functions/export")
    existing_functions = {
        function["id"]: function for function in exported.raise_for_status().json()
    }

    results: dict[str, str] = {}
    for spec in specs:
        existing = existing_functions.get(spec.id)
        # Open WebUI recomputes meta.manifest from the frontmatter.
        payload = {
            "id": spec.id,
            "name": spec.name,
            "content": spec.content,
            "meta": existing["meta"] if existing is not None else {},
        }
        if existing is None:
            (
                await client.post("/api/v1/functions/create", json=payload)
            ).raise_for_status()
            (
                await client.post(f"/api/v1/functions/id/{spec.id}/toggle")
            ).raise_for_status()
            results[spec.id] = "created"
        elif existing["content"] != spec.content or existing["name"] != spec.name:
            (
                await client.post(
                    f"/api/v1/functions/id/{spec.id}/update", json=payload
                )
            ).raise_for_status()
            results[spec.id] = "updated"
        else:
            results[spec.id] = "unchanged"
    return results
