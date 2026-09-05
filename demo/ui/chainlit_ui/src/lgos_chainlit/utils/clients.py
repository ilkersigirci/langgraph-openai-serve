"""OpenAI client shared by the Chainlit demo applications."""

from typing import Any

from openai import AsyncOpenAI, OpenAIError
from openai.types import Model

from lgos_chainlit.gateway import gateway_config
from lgos_chainlit.settings import settings

LGOS_MODEL_OWNER = "langgraph-openai-serve"
gateway = gateway_config(
    settings.OPENAI_GATEWAY_TYPE,
    settings.OPENAI.gateway_base_url,
)

openai_client = AsyncOpenAI(
    base_url=gateway.responses_base_url,
    api_key=settings.OPENAI.api_key,
    max_retries=0,
    default_headers={"User-Agent": "lgos-chainlit"},
)
catalog_client = AsyncOpenAI(
    base_url=gateway.catalog_base_url,
    api_key=settings.OPENAI.api_key,
    max_retries=0,
)
catalog_detail_client = AsyncOpenAI(
    base_url=gateway.catalog_detail_base_url,
    api_key=settings.OPENAI.api_key,
    max_retries=0,
)
files_client = AsyncOpenAI(
    base_url=gateway.files_base_url,
    api_key=settings.OPENAI.api_key,
    max_retries=0,
)


def files_request() -> tuple[AsyncOpenAI, str]:
    """Return the configured Files client and gateway provider."""
    return files_client, gateway.files_provider


async def retrieve_model(model_id: str) -> Model:
    """Retrieve LGOS model metadata through the configured endpoint."""
    model_prefixes = gateway.model_prefixes
    if model_prefixes:
        model_prefix, catalog_model_id = _catalog_model(model_id, model_prefixes)
        model = await _catalog_client(model_prefix).models.retrieve(
            model=catalog_model_id
        )
    else:
        model = await catalog_detail_client.models.retrieve(
            **_provider_model_request(model_id)
        )
    if not isinstance(model, Model):
        msg = "The endpoint returned an invalid model response."
        raise OpenAIError(msg)
    return model


async def list_models() -> list[Model]:
    """List models through the configured OpenAI endpoint."""
    if model_prefixes := gateway.model_prefixes:
        models = []
        for model_prefix in model_prefixes:
            catalog = await _catalog_client(model_prefix).models.list()
            models.extend(
                model.model_copy(update={"id": f"{model_prefix}/{model.id}"})
                for model in catalog.data
                if model.owned_by == LGOS_MODEL_OWNER
            )
        return models

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
        provider_models = await catalog_detail_client.models.list(
            extra_headers={"x-model-provider": provider}
        )
        models.extend(
            model.model_copy(update={"id": f"{provider}/{model.id}"})
            for model in provider_models.data
        )
    return models


def model_request(model_id: str) -> dict[str, Any]:
    """Build a Responses request for the selected gateway's native route."""
    if not isinstance(model_id, str) or not model_id:
        msg = "OpenAI model ID is missing."
        raise ValueError(msg)

    if model_prefixes := gateway.model_prefixes:
        return {"model": _managed_model_id(model_id, model_prefixes)}

    return _provider_model_request(model_id)


def _bifrost_model(model_id: str) -> tuple[str, str]:
    provider, separator, upstream_model = model_id.partition("/")
    if not provider or not separator or not upstream_model:
        msg = f"Bifrost model ID must use the provider/model format: {model_id!r}."
        raise ValueError(msg)
    return provider, upstream_model


def _catalog_client(model_prefix: str) -> AsyncOpenAI:
    return catalog_detail_client.with_options(
        base_url=f"{gateway.catalog_detail_base_url}/{model_prefix}"
    )


def _provider_model_request(model_id: str) -> dict[str, Any]:
    provider, upstream_model = _bifrost_model(model_id)
    return {
        "model": upstream_model,
        "extra_headers": {"x-model-provider": provider},
    }


def _managed_model_id(model_id: str, model_prefixes: tuple[str, ...]) -> str:
    provider, separator, upstream_model = model_id.partition("/")
    if not separator:
        return f"{model_prefixes[0]}/{model_id}"
    if provider not in model_prefixes or not upstream_model:
        expected = ", ".join(f"{prefix}/model" for prefix in model_prefixes)
        msg = f"LiteLLM model ID must use one of [{expected}]: {model_id!r}."
        raise ValueError(msg)
    return model_id


def _catalog_model(
    model_id: str,
    model_prefixes: tuple[str, ...],
) -> tuple[str, str]:
    model_prefix, upstream_model = _managed_model_id(model_id, model_prefixes).split(
        "/", maxsplit=1
    )
    return model_prefix, upstream_model
