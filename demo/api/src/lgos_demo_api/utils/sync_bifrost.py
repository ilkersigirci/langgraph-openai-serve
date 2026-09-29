"""Publish LGOS metadata through Bifrost's native model catalog.

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
from urllib.parse import quote

import httpx2
from pydantic import BaseModel, TypeAdapter, ValidationError

from lgos_demo_api.utils.model_catalog import read_model_catalog, validate_namespace


class ModelAttributes(BaseModel):
    """One entry of Bifrost's `PUT /api/models/catalog` request."""

    provider: str
    model: str
    additional_attributes: dict[str, str]


MODEL_ATTRIBUTES = TypeAdapter(list[ModelAttributes])


def prepare_catalog(
    sources: Mapping[str, httpx2.Client], directory: Path
) -> list[ModelAttributes]:
    """Write graph pricing rows and the attributes to publish on them."""
    pricing: dict[str, dict[str, str | int]] = {}
    attributes: list[ModelAttributes] = []
    for provider, source in sources.items():
        validate_namespace(provider)
        for name, model in read_model_catalog(source).items():
            # Zero prices only anchor the attributes: graphs pay for their LLM
            # calls outside Bifrost. Limits and capabilities stay unset.
            pricing[f"{provider}/{name}"] = {
                "provider": provider,
                "mode": "responses",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }
            # Attribute values are strings, so the complete extension is JSON
            # encoded. Bifrost's model editor displays the description.
            attributes.append(
                ModelAttributes(
                    provider=provider,
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
    # After graph changes, reload the rewritten datasheet and the providers'
    # cached model lists so new graphs have rows and appear in /v1/models.
    gateway.post("api/pricing/force-sync").raise_for_status()
    for provider in sorted({entry.provider for entry in attributes}):
        gateway.post(
            f"api/providers/{quote(provider, safe='')}/refresh-models"
        ).raise_for_status()
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
        "--source",
        action="append",
        required=True,
        metavar="PROVIDER=URL",
        help="Bifrost provider name and its LGOS /v1 base URL",
    )
    sync = commands.add_parser("sync", help="Publish attributes to a healthy Bifrost")
    sync.add_argument("--gateway-url", required=True, help="Bifrost root URL")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            with ExitStack() as stack:
                sources: dict[str, httpx2.Client] = {}
                for source in args.source:
                    provider, _, url = source.partition("=")
                    if not url or provider in sources:
                        msg = "Sources must use unique PROVIDER=URL pairs"
                        raise ValueError(msg)
                    sources[provider] = stack.enter_context(
                        httpx2.Client(base_url=f"{url.rstrip('/')}/", timeout=30)
                    )
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
        raise SystemExit("Bifrost model sync failed: invalid catalog data") from None
    except (httpx2.HTTPError, OSError, ValueError) as exc:
        raise SystemExit(f"Bifrost model sync failed: {exc}") from None


if __name__ == "__main__":
    main()
