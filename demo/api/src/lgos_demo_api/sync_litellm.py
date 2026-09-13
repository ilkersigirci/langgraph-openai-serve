"""Copy LGOS catalog metadata into native, database-backed LiteLLM models."""

import argparse
import os
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

import httpx
from langgraph_openai_serve.api.models.schemas import ModelDetails, ModelList
from pydantic import BaseModel, JsonValue, ValidationError


class ModelInfo(BaseModel):
    id: str
    db_model: bool
    lgos: dict[str, JsonValue] | None = None
    lgos_sync: bool = False
    supports_native_streaming: bool | None = None


class Deployment(BaseModel):
    model_name: str
    model_info: ModelInfo


class Deployments(BaseModel):
    data: list[Deployment]


def _is_owned(deployment: Deployment, *, prefix: str) -> bool:
    """Identify models owned by this sync within the requested namespace."""
    return (
        deployment.model_name.startswith(f"{prefix}/")
        and deployment.model_info.db_model
        and deployment.model_info.lgos_sync
    )


def sync_models(
    source: httpx.Client,
    gateway: httpx.Client,
    *,
    prefix: str,
    api_key: str,
    api_base: str | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Reconcile namespaced LGOS models without changing other deployments."""
    if not prefix or any(char.isspace() or char in "/*" for char in prefix):
        msg = "Model namespace must be non-empty, without whitespace, / or *"
        raise ValueError(msg)

    response = source.get("models")
    response.raise_for_status()
    catalog = ModelList.model_validate(response.json())
    desired: dict[str, ModelDetails] = {}
    for summary in catalog.data:
        response = source.get(f"models/{quote(summary.id, safe='')}")
        response.raise_for_status()
        model = ModelDetails.model_validate(response.json())
        if model.id != summary.id or model.owned_by != "langgraph-openai-serve":
            msg = f"Invalid model detail for {summary.id}"
            raise ValueError(msg)
        name = f"{prefix}/{model.id}"
        if name in desired:
            msg = f"Duplicate upstream model: {model.id}"
            raise ValueError(msg)
        desired[name] = model

    response = gateway.get("model/info")
    response.raise_for_status()
    deployments = Deployments.model_validate(response.json()).data
    # Validate every desired name before writing. Ambiguous or independently
    # managed matches need an operator decision, not a guessed target.
    existing: dict[str, Deployment | None] = {}
    for name in desired:
        matches = [item for item in deployments if item.model_name == name]
        if len(matches) > 1 or (matches and not _is_owned(matches[0], prefix=prefix)):
            msg = f"{name}: conflicts with ambiguous or non-sync-owned deployment"
            raise ValueError(msg)
        existing[name] = matches[0] if matches else None

    results: dict[str, str] = {}
    for name, model in desired.items():
        current = existing[name]
        # LiteLLM's ModelInfo supplies a random ID and db_model=False when
        # omitted, even on PATCH. Preserve the deployment identity explicitly.
        extension = model.lgos.model_dump(mode="json")
        info: dict[str, JsonValue] = {
            "id": current.model_info.id
            if current is not None
            else str(uuid5(NAMESPACE_URL, f"lgos:{name}")),
            "db_model": True,
            "lgos": extension,
            "lgos_sync": True,
            "supports_native_streaming": True,
        }
        if current is None:
            results[name] = "created"
            if not dry_run:
                gateway.post(
                    "model/new",
                    json={
                        "model_name": name,
                        "litellm_params": {
                            "model": f"openai/{model.id}",
                            "api_base": api_base or str(source.base_url).rstrip("/"),
                            "api_key": api_key,
                            # LiteLLM does not assume custom models accept Chat's user.
                            "allowed_openai_params": ["user"],
                        },
                        "model_info": info,
                    },
                ).raise_for_status()
        elif (
            current.model_info.lgos != extension
            or current.model_info.supports_native_streaming is not True
        ):
            results[name] = "updated"
            if not dry_run:
                gateway.patch(
                    f"model/{quote(current.model_info.id, safe='')}/update",
                    json={"model_info": info},
                ).raise_for_status()
        else:
            results[name] = "unchanged"

    for deployment in sorted(
        deployments, key=lambda item: (item.model_name, item.model_info.id)
    ):
        name = deployment.model_name
        if name in desired or not _is_owned(deployment, prefix=prefix):
            continue
        results[name] = "deleted"
        if not dry_run:
            gateway.post(
                "model/delete", json={"id": deployment.model_info.id}
            ).raise_for_status()
    return results


def main() -> None:
    """Run one operator-requested catalog synchronization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", required=True, help="LGOS /v1 base URL")
    parser.add_argument(
        "--gateway-url", required=True, help="LiteLLM administrator API root URL"
    )
    parser.add_argument("--prefix", required=True, help="Public model namespace")
    parser.add_argument(
        "--api-base", help="LGOS /v1 URL reachable from LiteLLM (default: --source-url)"
    )
    parser.add_argument(
        "--api-key-env", help="Environment variable holding the upstream API key"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        api_key = os.environ[args.api_key_env] if args.api_key_env else "DUMMY"
        with (
            httpx.Client(
                base_url=args.source_url.rstrip("/") + "/",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            ) as source,
            httpx.Client(
                base_url=args.gateway_url.rstrip("/") + "/",
                headers={"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"},
                timeout=30,
            ) as gateway,
        ):
            results = sync_models(
                source,
                gateway,
                prefix=args.prefix,
                api_base=args.api_base,
                api_key=api_key,
                dry_run=args.dry_run,
            )
        for name, action in results.items():
            print(f"{name}: {'would be ' if args.dry_run else ''}{action}")
    except ValidationError:
        raise SystemExit(
            "LiteLLM model sync failed: invalid catalog response"
        ) from None
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        # Do not print response bodies: management errors can echo credentials.
        raise SystemExit(f"LiteLLM model sync failed: {exc}") from None


if __name__ == "__main__":
    main()
