"""
Publish LGOS metadata through Bifrost's native model catalog.

Bifrost stores model attributes only on existing pricing rows, and only its
pricing datasheet creates those rows. `prepare` therefore writes a datasheet
of graph rows before Bifrost starts: with an empty config store, Bifrost exits
at boot when that datasheet cannot load. `sync` writes the attributes once
Bifrost is healthy.
"""

import argparse
import json
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

import httpx2
from pydantic import BaseModel, TypeAdapter, ValidationError

from lgos_demo_api.utils.model_catalog import read_model_catalogs

# The custom provider in docker/configs/bifrost/config.json that owns the
# public `lgos/` namespace.
PROVIDER = "lgos"


class ModelAttributes(BaseModel):
    """One entry of Bifrost's `PUT /api/models/catalog` request."""

    provider: str = PROVIDER
    model: str
    additional_attributes: dict[str, str]


MODEL_ATTRIBUTES = TypeAdapter(list[ModelAttributes])


def prepare_catalog(
    sources: Mapping[str, httpx2.Client], directory: Path
) -> list[ModelAttributes]:
    """Write graph pricing rows and the attributes to publish on them."""
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
    # A rerun can race Bifrost's periodic datasheet reload. Replace each file
    # atomically; the shared directory mount makes the rename visible to it.
    for filename, content in (
        ("pricing.json", json.dumps(pricing).encode()),
        ("attributes.json", MODEL_ATTRIBUTES.dump_json(attributes)),
    ):
        temporary = directory / f"{filename}.tmp"
        temporary.write_bytes(content)
        temporary.replace(directory / filename)
    return attributes


def sync_catalog(gateway: httpx2.Client, attributes: list[ModelAttributes]) -> None:
    """Replace the attributes of every prepared graph row."""
    # Reload the rewritten datasheet and the provider's cached model list so
    # added graphs get rows and removed graphs leave /v1/models.
    gateway.post("api/pricing/force-sync").raise_for_status()
    gateway.post(f"api/providers/{PROVIDER}/refresh-models").raise_for_status()
    # Bifrost writes the batch in one transaction and rejects all of it when
    # any pricing row is missing.
    gateway.put(
        "api/models/catalog", json=MODEL_ATTRIBUTES.dump_python(attributes)
    ).raise_for_status()


def main() -> None:
    """Run one catalog preparation or publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, required=True, help="Directory shared with Bifrost"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Write the catalog before Bifrost starts"
    )
    prepare.add_argument(
        "--source-url",
        action="append",
        required=True,
        help="LGOS /v1 base URL; repeat for every API in the namespace",
    )
    sync = commands.add_parser("sync", help="Publish attributes to a healthy Bifrost")
    sync.add_argument("--gateway-url", required=True, help="Bifrost root URL")
    args = parser.parse_args()
    try:  # ruff: ignore[too-many-statements-in-try-clause] - The CLI reports failures from the whole sync operation consistently.
        if args.command == "prepare":
            with ExitStack() as stack:
                sources = {
                    url: stack.enter_context(
                        httpx2.Client(base_url=f"{url.rstrip('/')}/", timeout=30)
                    )
                    for url in args.source_url
                }
                attributes = prepare_catalog(sources, args.directory)
            print(f"Prepared {len(attributes)} LGOS models for Bifrost")
        else:
            attributes = MODEL_ATTRIBUTES.validate_json(
                (args.directory / "attributes.json").read_bytes()
            )
            with httpx2.Client(
                base_url=f"{args.gateway_url.rstrip('/')}/", timeout=120
            ) as gateway:
                sync_catalog(gateway, attributes)
            print(f"Published metadata for {len(attributes)} LGOS models to Bifrost")
    except ValidationError:
        # Validation errors quote their input, which can hold complete settings.
        msg = "Bifrost model sync failed: invalid catalog data"
        raise SystemExit(msg) from None
    except (httpx2.HTTPError, OSError, ValueError) as exc:
        msg = f"Bifrost model sync failed: {exc}"
        raise SystemExit(msg) from None


if __name__ == "__main__":
    main()
