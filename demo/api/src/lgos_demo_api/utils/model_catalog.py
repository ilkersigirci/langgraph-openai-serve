"""Read the complete public metadata of an LGOS deployment."""

from collections.abc import Mapping
from urllib.parse import quote

import httpx2
from langgraph_openai_serve.api.models.schemas import Model, ModelList


def read_model_catalog(source: httpx2.Client) -> dict[str, Model]:
    """Validate the whole catalog before a gateway sync changes anything."""
    response = source.get("models")
    response.raise_for_status()
    models: dict[str, Model] = {}
    for summary in ModelList.model_validate(response.json()).data:
        response = source.get(f"models/{quote(summary.id, safe='')}")
        response.raise_for_status()
        models[summary.id] = Model.model_validate(response.json())
    return models


def read_model_catalogs(
    sources: Mapping[str, httpx2.Client],
) -> dict[str, tuple[str, Model]]:
    """Combine complete catalogs, rejecting collisions before gateway writes."""
    models: dict[str, tuple[str, Model]] = {}
    for source_id, source in sources.items():
        for name, model in read_model_catalog(source).items():
            if name in models:
                msg = f"Duplicate upstream model across sources: {name}"
                raise ValueError(msg)
            models[name] = (source_id, model)
    return models
