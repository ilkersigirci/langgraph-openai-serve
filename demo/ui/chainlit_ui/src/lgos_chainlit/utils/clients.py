"""OpenAI client shared by the Chainlit demo applications."""

from typing import Any

from openai import AsyncOpenAI, OpenAIError
from openai.types import Model

from lgos_chainlit.settings import settings

LGOS_MODEL_OWNER = "langgraph-openai-serve"

openai_client = AsyncOpenAI(
    base_url=settings.OPENAI.base_url,
    api_key=settings.OPENAI.api_key,
    max_retries=0,
    default_headers={"User-Agent": "lgos-chainlit"},
)
catalog_client = AsyncOpenAI(
    base_url=settings.OPENAI.catalog_base_url or settings.OPENAI.base_url,
    api_key=settings.OPENAI.api_key,
    max_retries=0,
)
files_client = AsyncOpenAI(
    base_url=settings.OPENAI.files_base_url,
    api_key=settings.OPENAI.api_key,
    max_retries=0,
)


def files_request() -> tuple[AsyncOpenAI, str | None]:
    """Return the configured Files client and optional gateway provider."""
    return files_client, settings.OPENAI.files_provider


async def retrieve_model(model_id: str) -> Model:
    """Retrieve LGOS model metadata through the configured endpoint."""
    model = await openai_client.models.retrieve(**model_request(model_id))
    if not isinstance(model, Model):
        msg = "The endpoint returned an invalid model response."
        raise OpenAIError(msg)
    return model


async def list_models() -> list[Model]:
    """List models through the configured OpenAI endpoint."""
    if settings.OPENAI.catalog_base_url is None:
        models = await openai_client.models.list()
        return list(models.data)

    catalog = await catalog_client.models.list()
    providers = sorted(
        {
            _bifrost_model(model.id)[0]
            for model in catalog.data
            if model.owned_by == LGOS_MODEL_OWNER
        }
    )
    models = []
    for provider in providers:
        provider_models = await openai_client.models.list(
            extra_headers={"x-model-provider": provider}
        )
        models.extend(
            model.model_copy(update={"id": f"{provider}/{model.id}"})
            for model in provider_models.data
        )
    return models


def model_request(model_id: str) -> dict[str, Any]:
    """Build a request for a standard endpoint or Bifrost pass-through."""
    if not isinstance(model_id, str) or not model_id:
        msg = "OpenAI model ID is missing."
        raise ValueError(msg)

    if settings.OPENAI.catalog_base_url is None:
        return {"model": model_id}

    provider, upstream_model = _bifrost_model(model_id)

    return {
        "model": upstream_model,
        "extra_headers": {"x-model-provider": provider},
    }


def _bifrost_model(model_id: str) -> tuple[str, str]:
    provider, separator, upstream_model = model_id.partition("/")
    if not provider or not separator or not upstream_model:
        msg = f"Bifrost model ID must use the provider/model format: {model_id!r}."
        raise ValueError(msg)
    return provider, upstream_model
