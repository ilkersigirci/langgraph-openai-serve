"""OpenAI clients for the configured gateway."""

from openai import AsyncOpenAI, DefaultAsyncHttpx2Client, OpenAIError
from openai.types import Model

from lgos_chainlit.auth import gateway_credential
from lgos_chainlit.gateway import gateway_config, litellm_models
from lgos_chainlit.lgos_protocol import LGOS_MODEL_OWNER
from lgos_chainlit.settings import settings

gateway = gateway_config(
    settings.OPENAI_GATEWAY_TYPE,
    settings.OPENAI_GATEWAY_BASE_URL,
)
gateway_http_client = DefaultAsyncHttpx2Client()

# Both gateways serve Files, speech, and their OpenAI model catalog under the
# root's /v1 path, the same route Open WebUI's native audio settings use. Only
# the Responses route differs.
v1_client = AsyncOpenAI(
    base_url=f"{gateway.root_url}/v1",
    api_key=gateway_credential,
    http_client=gateway_http_client,
    max_retries=0,
    default_headers={"User-Agent": "lgos-chainlit"},
)
responses_client = v1_client.with_options(base_url=gateway.responses_base_url)


async def retrieve_model(model_id: str) -> Model:
    """Retrieve LGOS model metadata through the configured endpoint."""
    if gateway.type == "litellm":
        for model in await list_models():
            if model.id == model_id:
                return model
        msg = f"Model {model_id!r} is not available in LiteLLM model info."
        raise OpenAIError(msg)

    # Bifrost reads x-model-provider only on pass-through routes. Responses
    # select the provider from the catalog ID's prefix instead.
    provider, upstream_model = bifrost_model(model_id)
    model = await _bifrost_detail_client().models.retrieve(
        upstream_model, extra_headers={"x-model-provider": provider}
    )
    if not isinstance(model, Model):
        msg = "The endpoint returned an invalid model response."
        raise OpenAIError(msg)
    return model


async def list_models() -> list[Model]:
    """List models through the configured OpenAI endpoint."""
    if gateway.type == "litellm":
        payload = await v1_client.get(f"{gateway.root_url}/model/info", cast_to=object)
        return litellm_models(payload)

    catalog = await v1_client.models.list()
    allowed_models = {model.id for model in catalog.data}
    providers = sorted(
        {
            bifrost_model(model.id)[0]
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
            if f"{provider}/{model.id}" in allowed_models
        )
    return models


def bifrost_model(model_id: str) -> tuple[str, str]:
    """Split a Bifrost catalog ID into its provider and upstream model."""
    provider, separator, upstream_model = model_id.partition("/")
    if not provider or not separator or not upstream_model:
        msg = f"Bifrost model ID must use the provider/model format: {model_id!r}."
        raise ValueError(msg)
    return provider, upstream_model


def _bifrost_detail_client() -> AsyncOpenAI:
    return v1_client.with_options(base_url=f"{gateway.root_url}/openai_passthrough/v1")
