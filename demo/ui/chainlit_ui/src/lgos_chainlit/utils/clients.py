"""OpenAI client shared by the Chainlit demo applications."""

from typing import Any

from openai import AsyncOpenAI
from openai.types import Model

from lgos_chainlit.settings import settings

openai_client = AsyncOpenAI(
    base_url=settings.OPENAI.base_url,
    api_key=settings.OPENAI.api_key,
)


async def retrieve_model(model_id: str) -> Model:
    """Retrieve LGOS model metadata through the configured endpoint."""
    return await openai_client.models.retrieve(**model_request(model_id))


async def list_model_ids() -> list[str]:
    """List standard or explicitly routed models through one OpenAI client."""
    routes = settings.OPENAI.model_routes
    if not routes:
        models = await openai_client.models.list()
        return [model.id for model in models.data]

    model_ids = []
    for route, headers in routes.items():
        models = await openai_client.models.list(extra_headers=headers)
        model_ids.extend(f"{route}/{model.id}" for model in models.data)
    return model_ids


def model_request(model_id: str) -> dict[str, Any]:
    """Resolve a selected UI model through explicit route configuration."""
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("OpenAI model ID is missing.")

    routes = settings.OPENAI.model_routes
    if not routes:
        return {"model": model_id}

    route, separator, upstream_model = model_id.partition("/")
    headers = routes.get(route)
    if not separator or not upstream_model or headers is None:
        raise ValueError(f"Unknown configured OpenAI model route in {model_id!r}.")

    request: dict[str, Any] = {"model": upstream_model}
    if headers:
        request["extra_headers"] = headers
    return request
