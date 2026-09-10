"""OpenAI client shared by the Chainlit demo applications."""

from typing import Any

from openai import AsyncOpenAI, DefaultAsyncHttpxClient, OpenAIError
from openai.types import Model

from lgos_chainlit.auth.chainlit import gateway_api_key
from lgos_chainlit.gateway import gateway_config, litellm_models
from lgos_chainlit.settings import settings

LGOS_MODEL_OWNER = "langgraph-openai-serve"
gateway = gateway_config(
    settings.OPENAI_GATEWAY_TYPE,
    settings.OPENAI_GATEWAY_BASE_URL,
)
gateway_http_client = DefaultAsyncHttpxClient()

openai_client = AsyncOpenAI(
    base_url=gateway.responses_base_url,
    api_key=gateway_api_key,
    http_client=gateway_http_client,
    max_retries=0,
    default_headers={"User-Agent": "lgos-chainlit"},
)
files_client = AsyncOpenAI(
    base_url=gateway.files_base_url,
    api_key=gateway_api_key,
    http_client=gateway_http_client,
    max_retries=0,
)


def files_request() -> tuple[AsyncOpenAI, str]:
    """Return the configured Files client and gateway provider."""
    return files_client, gateway.files_provider


async def retrieve_model(model_id: str) -> Model:
    """Retrieve LGOS model metadata through the configured endpoint."""
    if not gateway.provider_routing:
        for model in await list_models():
            if model.id == model_id:
                return model
        msg = f"Model {model_id!r} is not available in LiteLLM model info."
        raise OpenAIError(msg)

    model = await _bifrost_detail_client().models.retrieve(
        **_provider_model_request(model_id)
    )
    if not isinstance(model, Model):
        msg = "The endpoint returned an invalid model response."
        raise OpenAIError(msg)
    return model


async def list_models() -> list[Model]:
    """List models through the configured OpenAI endpoint."""
    if not gateway.provider_routing:
        payload = await openai_client.get(
            f"{gateway.root_url}/model/info", cast_to=object
        )
        return litellm_models(payload)

    catalog = await openai_client.with_options(
        base_url=f"{gateway.root_url}/v1"
    ).models.list()
    providers = sorted(
        {
            _bifrost_model(model.id)[0]
            for model in catalog.data
            if model.owned_by == LGOS_MODEL_OWNER
        }
    )
    models = []
    for provider in providers:
        provider_models = await _bifrost_detail_client().models.list(
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

    if not gateway.provider_routing:
        return {"model": model_id}

    return _provider_model_request(model_id)


def _bifrost_model(model_id: str) -> tuple[str, str]:
    provider, separator, upstream_model = model_id.partition("/")
    if not provider or not separator or not upstream_model:
        msg = f"Bifrost model ID must use the provider/model format: {model_id!r}."
        raise ValueError(msg)
    return provider, upstream_model


def _bifrost_detail_client() -> AsyncOpenAI:
    return openai_client.with_options(
        base_url=f"{gateway.root_url}/openai_passthrough/v1"
    )


def _provider_model_request(model_id: str) -> dict[str, Any]:
    provider, upstream_model = _bifrost_model(model_id)
    return {
        "model": upstream_model,
        "extra_headers": {"x-model-provider": provider},
    }
