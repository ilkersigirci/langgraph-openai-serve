"""OpenAI clients shared by the Chainlit demo applications."""

from typing import Any

from openai import AsyncOpenAI
from openai.types import Model

from lgos_chainlit.settings import settings

MODEL_PROVIDER_HEADER = "x-model-provider"

inference_client = AsyncOpenAI(
    base_url=settings.INFERENCE.base_url,
    api_key=settings.INFERENCE.api_key,
)
catalog_client = AsyncOpenAI(
    base_url=settings.chainlit_catalog_endpoint.base_url,
    api_key=settings.chainlit_catalog_endpoint.api_key,
)


def model_request(model_id: str) -> dict[str, Any]:
    """Route a Bifrost-prefixed model through its raw provider endpoint."""
    provider, separator, model = model_id.partition("/")
    if not separator or not provider or not model:
        return {"model": model_id}
    return {
        "model": model,
        "extra_headers": {MODEL_PROVIDER_HEADER: provider},
    }


async def retrieve_model(model_id: str) -> Model:
    """Retrieve raw LGOS model metadata through the inference endpoint."""
    return await inference_client.models.retrieve(**model_request(model_id))
