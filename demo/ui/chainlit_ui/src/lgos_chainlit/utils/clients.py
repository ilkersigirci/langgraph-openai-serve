"""OpenAI clients shared by the Chainlit demo applications."""

from openai import AsyncOpenAI

from lgos_chainlit.settings import settings

inference_client = AsyncOpenAI(
    base_url=settings.INFERENCE.base_url,
    api_key=settings.INFERENCE.api_key,
)
discovery_client = AsyncOpenAI(
    base_url=settings.chainlit_discovery_endpoint.base_url,
    api_key=settings.chainlit_discovery_endpoint.api_key,
)
