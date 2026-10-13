"""Keep Open WebUI's gateway connection from listing the Generic Pipe's models."""

from collections.abc import Awaitable, Callable
from typing import Any

from .functions.generic.contracts import LGOS_MODEL_NAMESPACE

# Open WebUI's model entries, request, and user stay opaque here.
ModelEntry = dict[str, Any]
FetchModels = Callable[..., Awaitable[list[ModelEntry]]]


def without_lgos_models(fetch_openai_models: FetchModels) -> FetchModels:
    """
    Drop the ``lgos/`` models from Open WebUI's OpenAI connection models.

    The connection reaches the same gateway as the Generic Pipe, which serves
    every ``lgos/`` model. Open WebUI's connection settings offer only an
    allowlist, so the namespace decides, whether or not the Pipe is listing.
    """

    async def fetch_openai_models_without_lgos(
        request: object, user: object = None
    ) -> list[ModelEntry]:
        models = await fetch_openai_models(request, user)
        return [
            model
            for model in models
            if not model["id"].startswith(LGOS_MODEL_NAMESPACE)
        ]

    return fetch_openai_models_without_lgos
