"""OpenAI clients for the configured gateway."""

from openai import AsyncOpenAI, DefaultAsyncHttpx2Client, OpenAIError
from openai.types import Model

from lgos_chainlit.auth import gateway_credential
from lgos_chainlit.gateway import bifrost_models, gateway_config, litellm_models
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
    """Find one LGOS model in the gateway catalog."""
    for model in await list_models():
        if model.id == model_id:
            return model
    msg = f"Model {model_id!r} is not available in the gateway catalog."
    raise OpenAIError(msg)


async def list_models() -> list[Model]:
    """List the gateway's LGOS models from its native metadata catalog."""
    if gateway.type == "litellm":
        payload = await v1_client.get(f"{gateway.root_url}/model/info", cast_to=object)
        return litellm_models(payload)

    catalog = await v1_client.models.list()
    return bifrost_models(catalog.data)
