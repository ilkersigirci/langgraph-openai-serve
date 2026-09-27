"""Synchronize the demo integration with a running Open WebUI instance."""

from dataclasses import dataclass
from pathlib import Path

import httpx2
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


def sign_in(client: httpx2.Client, email: str, password: str) -> None:
    """Sign in and configure the client with the returned bearer token."""
    response = client.post(
        "/api/v1/auths/signin",
        json={"email": email, "password": password},
    ).raise_for_status()
    client.headers["Authorization"] = f"Bearer {response.json()['token']}"


def sync_functions(
    client: httpx2.Client,
    specs: tuple[FunctionSpec, ...] | None = None,
) -> dict[str, str]:
    """Create/update maintained Functions while preserving unrelated Functions."""
    specs = function_specs() if specs is None else specs
    exported = client.get("/api/v1/functions/export").raise_for_status().json()
    existing_functions = {function["id"]: function for function in exported}
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
            client.post(
                "/api/v1/functions/create",
                json=payload,
            ).raise_for_status()
            client.post(f"/api/v1/functions/id/{spec.id}/toggle").raise_for_status()
            results[spec.id] = "created"
        elif existing["content"] != spec.content or existing["name"] != spec.name:
            client.post(
                f"/api/v1/functions/id/{spec.id}/update",
                json=payload,
            ).raise_for_status()
            results[spec.id] = "updated"
        else:
            results[spec.id] = "unchanged"

    return results


def main() -> None:
    """Synchronize the bundled Functions and generated Workspace Models."""
    try:
        settings = Settings()
        gateway = gateway_config(
            settings.OPENAI_GATEWAY_TYPE,
            settings.DEMO_GATEWAY_HOST_URL or settings.OPENAI_GATEWAY_BASE_URL,
        )
        with (
            httpx2.Client(base_url=settings.URL, timeout=10) as client,
            OpenAI(
                base_url=f"{gateway.root_url}/v1",
                api_key=settings.OPENAI_GATEWAY_API_KEY,
                timeout=10,
            ) as openai_client,
        ):
            sign_in(client, settings.ADMIN_EMAIL, settings.ADMIN_PASSWORD)
            model_specs = discover_workspace_model_specs(
                openai_client,
                gateway=gateway,
            )
            function_results = sync_functions(client)
            sync_workspace_models(client, model_specs)
    except httpx2.HTTPStatusError as exc:
        msg = f"Open WebUI sync failed: {exc}\n{exc.response.text}"
        raise SystemExit(msg) from exc
    except (OSError, TypeError, ValueError, httpx2.HTTPError, OpenAIError) as exc:
        msg = f"Open WebUI sync failed: {exc}"
        raise SystemExit(msg) from exc

    for function_id, action in function_results.items():
        print(f"{action.capitalize()} Function: {function_id}")
    print(f"Synchronized Workspace Models: {len(model_specs)}")


if __name__ == "__main__":
    main()
