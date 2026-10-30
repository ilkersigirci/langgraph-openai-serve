"""
Publish LGOS metadata through a running Bifrost's native model catalog.

Bifrost stores model attributes only on existing pricing rows, and only its
pricing datasheet creates those rows. The sync therefore writes a datasheet of
graph rows into the directory Bifrost loads it from, has Bifrost reload it and
the graph provider's model list, and then writes the attributes, signed in as
Bifrost's admin.
"""

import argparse
import json
import os
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

import httpx2
from pydantic import BaseModel, ValidationError

from lgos_demo_api.utils.model_catalog import read_model_catalogs

# The custom provider in docker/configs/bifrost/config.json that owns the
# public `lgos/` namespace.
PROVIDER = "lgos"


class ModelAttributes(BaseModel):
    """One entry of Bifrost's `PUT /api/models/catalog` request."""

    provider: str = PROVIDER
    model: str
    additional_attributes: dict[str, str]


def write_datasheet(
    sources: Mapping[str, httpx2.Client], directory: Path
) -> list[ModelAttributes]:
    """Write graph pricing rows and return the attributes to publish on them."""
    pricing: dict[str, dict[str, str | int]] = {}
    attributes: list[ModelAttributes] = []
    for name, (_, model) in read_model_catalogs(sources).items():
        # Zero prices only anchor the attributes: a graph's own model calls
        # are separate gateway requests. Limits and capabilities stay unset.
        pricing[f"{PROVIDER}/{name}"] = {
            "provider": PROVIDER,
            "mode": "responses",
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
        # Attribute values are strings, so the complete extension is JSON
        # encoded. Bifrost's model editor displays the description.
        attributes.append(
            ModelAttributes(
                model=name,
                additional_attributes={
                    "description": model.lgos.description,
                    "lgos": model.lgos.model_dump_json(),
                },
            )
        )

    directory.mkdir(parents=True, exist_ok=True)
    # The write can race Bifrost's periodic datasheet reload, so replace the
    # file atomically; the shared directory mount makes the rename visible.
    temporary = directory / "pricing.json.tmp"
    temporary.write_text(json.dumps(pricing), encoding="utf-8")
    temporary.replace(directory / "pricing.json")
    return attributes


def sync_catalog(gateway: httpx2.Client, attributes: list[ModelAttributes]) -> None:
    """Replace the attributes of every written graph row."""
    # Reload the rewritten datasheet and the provider's cached model list so
    # added graphs get rows and removed graphs leave /v1/models.
    gateway.post("api/pricing/force-sync").raise_for_status()
    gateway.post(f"api/providers/{PROVIDER}/refresh-models").raise_for_status()
    # Bifrost writes the batch in one transaction and rejects all of it when
    # any pricing row is missing.
    gateway.put(
        "api/models/catalog",
        json=[attribute.model_dump() for attribute in attributes],
    ).raise_for_status()


def _publish(
    directory: Path, gateway_url: str, source_urls: list[str]
) -> list[ModelAttributes]:
    admin = (
        os.environ["BIFROST_ADMIN_USERNAME"],
        os.environ["BIFROST_ADMIN_PASSWORD"],
    )
    with ExitStack() as stack:
        sources = {
            url: stack.enter_context(
                httpx2.Client(base_url=f"{url.rstrip('/')}/", timeout=30)
            )
            for url in source_urls
        }
        attributes = write_datasheet(sources, directory)
        gateway = stack.enter_context(
            httpx2.Client(
                base_url=f"{gateway_url.rstrip('/')}/",
                auth=admin,
                timeout=120,
            )
        )
        sync_catalog(gateway, attributes)
    return attributes


def main() -> None:
    """Run one catalog publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Directory Bifrost loads its pricing datasheet from",
    )
    parser.add_argument("--gateway-url", required=True, help="Bifrost root URL")
    parser.add_argument(
        "--source-url",
        action="append",
        required=True,
        help="LGOS /v1 base URL; repeat for every API in the namespace",
    )
    args = parser.parse_args()
    try:
        attributes = _publish(args.directory, args.gateway_url, args.source_url)
        print(f"Published metadata for {len(attributes)} LGOS models to Bifrost")
    except ValidationError:
        # Validation errors quote their input, which can hold complete settings.
        msg = "Bifrost model sync failed: invalid catalog data"
        raise SystemExit(msg) from None
    except (httpx2.HTTPError, OSError, ValueError, KeyError) as exc:
        msg = f"Bifrost model sync failed: {exc}"
        raise SystemExit(msg) from None


if __name__ == "__main__":
    main()
