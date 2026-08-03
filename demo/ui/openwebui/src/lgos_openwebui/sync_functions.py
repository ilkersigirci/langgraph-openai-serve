"""Synchronize the demo integration with a running Open WebUI instance."""

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import OpenAI, OpenAIError
from pydantic import TypeAdapter, ValidationError

from .workspace_models import (
    discover_workspace_model_specs,
    sync_workspace_models,
)


@dataclass(frozen=True)
class FunctionSpec:
    """Describe a bundled Open WebUI Function."""

    id: str
    name: str
    content: str


FUNCTIONS_DIR = Path(__file__).with_name("functions")
MODEL_ROUTES_ADAPTER = TypeAdapter(dict[str, dict[str, str]])
DEFAULT_MODEL_ROUTES = (
    '{"lgos-a":{"x-model-provider":"lgos-a"},"lgos-b":{"x-model-provider":"lgos-b"}}'
)


def _frontmatter_title(content: str) -> str | None:
    lines = content.splitlines()
    if not lines or lines[0].strip() != '"""':
        return None

    title = None
    for line in lines[1:]:
        if '"""' in line:
            break
        key, separator, value = line.lstrip().partition(":")
        if separator and key == "title":
            title = value.strip()
    return title


def discover_function_specs(
    functions_dir: Path = FUNCTIONS_DIR,
) -> tuple[FunctionSpec, ...]:
    """Build Function specs from Python filenames and Open WebUI frontmatter."""
    specs: list[FunctionSpec] = []
    sources = sorted(
        source
        for source in functions_dir.glob("*.py")
        if not source.name.startswith("_")
    )

    for source in sources:
        function_id = source.stem
        if not function_id.isidentifier() or function_id != function_id.lower():
            raise ValueError(
                f"Open WebUI Function filename must be a lowercase Python identifier: "
                f"{source.name}"
            )

        content = source.read_text(encoding="utf-8")
        title = _frontmatter_title(content)
        if not title:
            raise ValueError(
                f"Open WebUI Function is missing a frontmatter title: {source}"
            )

        specs.append(
            FunctionSpec(
                id=function_id,
                name=title,
                content=content,
            )
        )

    if not specs:
        raise ValueError(f"No Open WebUI Functions found in {functions_dir}")
    return tuple(specs)


def sign_in(client: httpx.Client, email: str, password: str) -> None:
    """Sign in and configure the client with the returned bearer token."""
    response = client.post(
        "/api/v1/auths/signin",
        json={"email": email, "password": password},
    ).raise_for_status()
    data = response.json()
    token = data.get("token") if isinstance(data, dict) else None
    if not isinstance(token, str) or not token:
        raise ValueError("Open WebUI sign-in response did not contain a token.")
    client.headers["Authorization"] = f"Bearer {token}"


def sync_functions(
    client: httpx.Client,
    specs: tuple[FunctionSpec, ...] | None = None,
) -> dict[str, str]:
    """Create or update bundled Functions without deleting others."""
    specs = discover_function_specs() if specs is None else specs
    exported = client.get("/api/v1/functions/export").raise_for_status().json()
    if not isinstance(exported, list):
        raise ValueError("Open WebUI Functions export returned invalid data.")
    existing_functions = {
        function["id"]: function
        for function in exported
        if isinstance(function, dict) and isinstance(function.get("id"), str)
    }
    results: dict[str, str] = {}

    for spec in specs:
        existing = existing_functions.get(spec.id)
        meta = (
            existing.get("meta")
            if existing is not None and isinstance(existing.get("meta"), dict)
            else {}
        )
        payload = {
            "id": spec.id,
            "name": spec.name,
            "content": spec.content,
            "meta": meta,
        }

        if existing is None:
            client.post(
                "/api/v1/functions/create",
                json=payload,
            ).raise_for_status()
            client.post(f"/api/v1/functions/id/{spec.id}/toggle").raise_for_status()
            results[spec.id] = "created"
        elif (
            existing.get("content") != spec.content or existing.get("name") != spec.name
        ):
            client.post(
                f"/api/v1/functions/id/{spec.id}/update",
                json=payload,
            ).raise_for_status()
            results[spec.id] = "updated"
        else:
            results[spec.id] = "unchanged"

    return results


def main() -> None:
    """Synchronize the bundled Function and generated Workspace Models."""
    base_url = os.environ.get("DEMO_OPENWEBUI_URL", "http://localhost:3003")
    email = os.environ.get("DEMO_OPENWEBUI_ADMIN_EMAIL", "lgos@example.com")
    password = os.environ.get("DEMO_OPENWEBUI_ADMIN_PASSWORD", "lgos")
    openai_base_url = os.environ.get(
        "DEMO_OPENWEBUI_OPENAI_BASE_URL",
        "http://localhost:3000/openai_passthrough/v1",
    )
    api_key = os.environ.get("DEMO_OPENWEBUI_API_KEY", "DUMMY")
    model_routes_json = os.environ.get(
        "DEMO_OPENWEBUI_MODEL_ROUTES",
        DEFAULT_MODEL_ROUTES,
    )

    try:
        model_routes = MODEL_ROUTES_ADAPTER.validate_json(model_routes_json)
        with (
            httpx.Client(base_url=base_url, timeout=10) as client,
            OpenAI(
                base_url=openai_base_url,
                api_key=api_key,
                timeout=10,
            ) as openai_client,
        ):
            sign_in(client, email, password)
            function_results = sync_functions(client)
            model_specs = discover_workspace_model_specs(
                openai_client,
                model_routes,
            )
            sync_workspace_models(client, model_specs)
    except (OSError, ValueError, ValidationError, httpx.HTTPError, OpenAIError) as exc:
        raise SystemExit(f"Open WebUI sync failed: {exc}") from exc

    for function_id, action in function_results.items():
        print(f"{action.capitalize()} Function: {function_id}")
    print(f"Synchronized Workspace Models: {len(model_specs)}")


if __name__ == "__main__":
    main()
