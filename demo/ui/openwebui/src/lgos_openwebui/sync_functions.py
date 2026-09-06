"""Synchronize the demo integration with a running Open WebUI instance."""

from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import OpenAI, OpenAIError

from .bundle import bundle_function
from .functions.generic.gateway import gateway_config
from .settings import Settings
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
    """Build Function specs from source files and modular Function directories."""
    specs: list[FunctionSpec] = []
    sources = sorted(
        source
        for source in functions_dir.iterdir()
        if not source.name.startswith("_")
        and (source.is_dir() or (source.is_file() and source.suffix == ".py"))
    )

    for source in sources:
        function_id = source.stem if source.is_file() else source.name
        if not function_id.isidentifier() or function_id != function_id.lower():
            msg = (
                f"Open WebUI Function filename must be a lowercase Python identifier: "
                f"{source.name}"
            )
            raise ValueError(msg)

        content = (
            bundle_function(source)
            if source.is_dir()
            else source.read_text(encoding="utf-8")
        )
        title = _frontmatter_title(content)
        if not title:
            msg = f"Open WebUI Function is missing a frontmatter title: {source}"
            raise ValueError(msg)

        specs.append(
            FunctionSpec(
                id=function_id,
                name=title,
                content=content,
            )
        )

    if not specs:
        msg = f"No Open WebUI Functions found in {functions_dir}"
        raise ValueError(msg)
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
        msg = "Open WebUI sign-in response did not contain a token."
        raise ValueError(msg)
    client.headers["Authorization"] = f"Bearer {token}"


def sync_functions(
    client: httpx.Client,
    specs: tuple[FunctionSpec, ...] | None = None,
) -> dict[str, str]:
    """Create/update maintained Functions while preserving unrelated Functions."""
    specs = discover_function_specs() if specs is None else specs
    exported = client.get("/api/v1/functions/export").raise_for_status().json()
    if not isinstance(exported, list):
        msg = "Open WebUI Functions export returned invalid data."
        raise TypeError(msg)
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
    try:
        settings = Settings()
        gateway = gateway_config(
            settings.OPENAI_GATEWAY_TYPE,
            settings.OPENAI_GATEWAY_BASE_URL,
            local=True,
        )
        with (
            httpx.Client(base_url=settings.URL, timeout=10) as client,
            OpenAI(
                base_url=gateway.catalog_base_url,
                api_key=settings.API_KEY,
                timeout=10,
            ) as catalog_client,
            OpenAI(
                base_url=gateway.catalog_detail_base_url,
                api_key=settings.API_KEY,
                timeout=10,
            ) as catalog_detail_client,
        ):
            sign_in(client, settings.ADMIN_EMAIL, settings.ADMIN_PASSWORD)
            function_results = sync_functions(client)
            model_specs = discover_workspace_model_specs(
                catalog_client,
                catalog_detail_client,
                provider_routing=gateway.provider_routing,
                model_prefixes=gateway.model_prefixes,
            )
            sync_workspace_models(client, model_specs)
    except httpx.HTTPStatusError as exc:
        msg = f"Open WebUI sync failed: {exc}\n{exc.response.text}"
        raise SystemExit(msg) from exc
    except (OSError, ValueError, httpx.HTTPError, OpenAIError) as exc:
        msg = f"Open WebUI sync failed: {exc}"
        raise SystemExit(msg) from exc

    for function_id, action in function_results.items():
        print(f"{action.capitalize()} Function: {function_id}")
    print(f"Synchronized Workspace Models: {len(model_specs)}")


if __name__ == "__main__":
    main()
