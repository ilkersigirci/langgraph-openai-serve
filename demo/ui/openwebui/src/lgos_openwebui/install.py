"""Install bundled Functions through Open WebUI's API."""

from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from .bundle import bundle_function

FUNCTIONS_DIR = Path(__file__).with_name("functions")


@dataclass(frozen=True)
class FunctionSpec:
    """Describe a bundled Open WebUI Function."""

    id: str
    name: str
    content: str
    valves: dict[str, JsonValue] = field(default_factory=dict)


def function_specs(
    function_id: str,
    *,
    valves: dict[str, JsonValue] | None = None,
    include_demo_functions: bool = False,
) -> tuple[FunctionSpec, ...]:
    """Bundle the Generic Pipe and, optionally, the UserValves example."""
    specs = [
        FunctionSpec(
            # Open WebUI stores Function IDs lowercased; later calls use that ID.
            id=function_id.lower(),
            name="Generic",
            content=bundle_function(FUNCTIONS_DIR / "generic"),
            valves=valves or {},
        ),
    ]
    if include_demo_functions:
        specs.append(
            FunctionSpec(
                id="uservalves_simple",
                name="UserValves Simple",
                content=(FUNCTIONS_DIR / "uservalves_simple.py").read_text(
                    encoding="utf-8"
                ),
            )
        )
    return tuple(specs)


class _StoredFunction(BaseModel):
    id: str
    name: str
    content: str
    meta: dict[str, JsonValue] = Field(default_factory=dict)


async def install_functions(
    client: httpx.AsyncClient, specs: tuple[FunctionSpec, ...]
) -> dict[str, str]:
    """
    Create or update the Functions with an admin's ``client``.

    New Functions are enabled. Existing ones keep their enabled state, metadata,
    and the valves their spec does not set; unrelated Functions are untouched.
    """
    exported = await client.get("/api/v1/functions/export")
    existing_functions = {
        function.id: function
        for function in TypeAdapter(list[_StoredFunction]).validate_python(
            exported.raise_for_status().json()
        )
    }

    results: dict[str, str] = {}
    for spec in specs:
        existing = existing_functions.get(spec.id)
        # Open WebUI recomputes meta.manifest from the frontmatter.
        payload = {
            "id": spec.id,
            "name": spec.name,
            "content": spec.content,
            "meta": existing.meta if existing is not None else {},
        }
        if existing is None:
            (
                await client.post("/api/v1/functions/create", json=payload)
            ).raise_for_status()
            (
                await client.post(f"/api/v1/functions/id/{spec.id}/toggle")
            ).raise_for_status()
            results[spec.id] = "created"
        elif existing.content != spec.content or existing.name != spec.name:
            (
                await client.post(
                    f"/api/v1/functions/id/{spec.id}/update", json=payload
                )
            ).raise_for_status()
            results[spec.id] = "updated"
        else:
            results[spec.id] = "unchanged"
        # Every run reconciles the valves, so a rerun also completes a first
        # install that was interrupted after enabling the Function.
        if await _update_valves(client, spec) and results[spec.id] == "unchanged":
            results[spec.id] = "updated"
    return results


async def _update_valves(client: httpx.AsyncClient, spec: FunctionSpec) -> bool:
    """Merge the spec's valves into the stored ones; skip unchanged writes."""
    if not spec.valves:
        return False
    path = f"/api/v1/functions/id/{spec.id}/valves"
    stored = (await client.get(path)).raise_for_status().json() or {}
    merged = {**stored, **spec.valves}
    if merged == stored:
        return False
    (await client.post(f"{path}/update", json=merged)).raise_for_status()
    return True
