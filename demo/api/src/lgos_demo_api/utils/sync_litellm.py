"""Copy LGOS catalog metadata into native, database-backed LiteLLM models."""

import argparse
import os
from collections.abc import Collection, Mapping
from contextlib import ExitStack
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

import httpx2
from pydantic import BaseModel, JsonValue, ValidationError

from lgos_demo_api.utils.model_catalog import read_model_catalogs


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


def _validate_namespace(prefix: str) -> None:
    if not prefix or any(char.isspace() or char in "/*" for char in prefix):
        msg = "Model namespace must be non-empty, without whitespace, / or *"
        raise ValueError(msg)


def _index_deployments(
    deployments: list[Deployment], desired: Collection[str], *, prefix: str
) -> tuple[dict[str, ModelInfo], list[Deployment]]:
    """Find managed deployments and reject conflicting names before any writes."""
    existing: dict[str, ModelInfo] = {}
    retired: list[Deployment] = []
    for deployment in deployments:
        name = deployment.model_name
        info = deployment.model_info
        owned = info.db_model and info.lgos_sync
        if name in desired:
            if not owned or name in existing:
                msg = f"{name}: conflicts with ambiguous or non-sync-owned deployment"
                raise ValueError(msg)
            existing[name] = info
        elif owned and name.startswith(f"{prefix}/"):
            retired.append(deployment)
    return existing, retired


def sync_models(
    sources: Mapping[str, httpx2.Client],
    gateway: httpx2.Client,
    *,
    prefix: str,
    api_key: str,
    dry_run: bool = False,
) -> dict[str, str]:
    """Reconcile all catalogs in a namespace; source keys are upstream API bases."""
    _validate_namespace(prefix)
    desired = {
        f"{prefix}/{name}": entry
        for name, entry in read_model_catalogs(sources).items()
    }

    response = gateway.get("model/info")
    response.raise_for_status()
    deployments = Deployments.model_validate(response.json()).data
    existing, retired = _index_deployments(deployments, desired, prefix=prefix)

    results: dict[str, str] = {}
    for name, (api_base, model) in desired.items():
        current = existing.get(name)
        # PATCH merges ModelInfo defaults too. Supplying the stored ID and
        # db_model flag keeps LiteLLM from replacing them with its defaults.
        info = ModelInfo(
            id=current.id
            if current is not None
            else str(uuid5(NAMESPACE_URL, f"lgos:{name}")),
            db_model=True,
            lgos=model.lgos.model_dump(mode="json"),
            lgos_sync=True,
            supports_native_streaming=True,
        )
        if current == info:
            results[name] = "unchanged"
            continue
        results[name] = "created" if current is None else "updated"
        if dry_run:
            continue
        if current is None:
            gateway.post(
                "model/new",
                json={
                    "model_name": name,
                    "litellm_params": {
                        "model": f"openai/{model.id}",
                        "api_base": api_base,
                        "api_key": api_key,
                        # LiteLLM forwards Chat's safety_identifier to any model,
                        # but its deprecated user only to OpenAI's own models.
                        "allowed_openai_params": ["user"],
                    },
                    "model_info": info.model_dump(mode="json"),
                },
            ).raise_for_status()
        else:
            gateway.patch(
                f"model/{quote(current.id, safe='')}/update",
                json={"model_info": info.model_dump(mode="json")},
            ).raise_for_status()

    for deployment in sorted(
        retired, key=lambda item: (item.model_name, item.model_info.id)
    ):
        name = deployment.model_name
        results[name] = "deleted"
        if not dry_run:
            gateway.post(
                "model/delete", json={"id": deployment.model_info.id}
            ).raise_for_status()
    return results


def main() -> None:
    """Run one operator-requested catalog synchronization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-url",
        action="append",
        required=True,
        help="LGOS /v1 base URL; repeat for every API in the namespace",
    )
    parser.add_argument(
        "--gateway-url", required=True, help="LiteLLM administrator API root URL"
    )
    parser.add_argument(
        "--prefix", default="lgos", help="Public model namespace (default: lgos)"
    )
    parser.add_argument(
        "--api-base",
        action="append",
        help="Gateway-reachable URL for each --source-url, in the same order (default: source URLs)",
    )
    parser.add_argument(
        "--api-key-env", help="Environment variable holding the upstream API key"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bases = [base.rstrip("/") for base in (args.api_base or args.source_url)]
    if len(bases) != len(args.source_url) or len(set(bases)) != len(bases):
        parser.error("Provide one unique API base for each source URL")
    try:
        api_key = os.environ[args.api_key_env] if args.api_key_env else "DUMMY"
        with (
            ExitStack() as stack,
            httpx2.Client(
                base_url=args.gateway_url.rstrip("/") + "/",
                headers={"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"},
                timeout=30,
            ) as gateway,
        ):
            sources = {
                base: stack.enter_context(
                    httpx2.Client(
                        base_url=url.rstrip("/") + "/",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=30,
                    )
                )
                for base, url in zip(bases, args.source_url, strict=True)
            }
            results = sync_models(
                sources,
                gateway,
                prefix=args.prefix,
                api_key=api_key,
                dry_run=args.dry_run,
            )
        for name, action in results.items():
            print(f"{name}: {'would be ' if args.dry_run else ''}{action}")
    except ValidationError:
        msg = "LiteLLM model sync failed: invalid catalog response"
        raise SystemExit(msg) from None
    except (httpx2.HTTPError, ValueError, KeyError) as exc:
        # Do not print response bodies: management errors can echo credentials.
        msg = f"LiteLLM model sync failed: {exc}"
        raise SystemExit(msg) from None


if __name__ == "__main__":
    main()
