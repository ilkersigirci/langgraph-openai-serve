"""Read the complete public metadata of an LGOS deployment."""

from collections.abc import Mapping
from urllib.parse import quote

import httpx2
from langgraph_openai_serve.api.models.schemas import ModelDetails, ModelList


def validate_namespace(prefix: str) -> None:
    if not prefix or any(char.isspace() or char in "/*" for char in prefix):
        msg = "Model namespace must be non-empty, without whitespace, / or *"
        raise ValueError(msg)


def read_model_catalog(source: httpx2.Client) -> dict[str, ModelDetails]:
    """Validate the whole catalog before a gateway sync changes anything."""
    response = source.get("models")
    response.raise_for_status()
    catalog = ModelList.model_validate(response.json())
    models: dict[str, ModelDetails] = {}
    for summary in catalog.data:
        response = source.get(f"models/{quote(summary.id, safe='')}")
        response.raise_for_status()
        model = ModelDetails.model_validate(response.json())
        if model.id != summary.id or model.owned_by != "langgraph-openai-serve":
            msg = f"Invalid model detail for {summary.id}"
            raise ValueError(msg)
        if model.id in models:
            msg = f"Duplicate upstream model: {model.id}"
            raise ValueError(msg)
        models[model.id] = model
    return models


def read_model_catalogs(
    sources: Mapping[str, httpx2.Client],
) -> dict[str, tuple[str, ModelDetails]]:
    """Combine complete catalogs, rejecting collisions before gateway writes."""
    models: dict[str, tuple[str, ModelDetails]] = {}
    for source_id, source in sources.items():
        for name, model in read_model_catalog(source).items():
            if name in models:
                msg = f"Duplicate upstream model across sources: {name}"
                raise ValueError(msg)
            models[name] = (source_id, model)
    return models
