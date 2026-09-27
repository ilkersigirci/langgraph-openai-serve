"""OpenAI-compatible client and model-catalog helpers."""

from openai import AsyncOpenAI

from .contracts import LGOS_MODEL_OWNER


def _client(
    *,
    base_url: str,
    api_key: str,
    timeout: float,
) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
        default_headers={"User-Agent": "lgos-openwebui"},
    )


async def _list_model_ids(client: AsyncOpenAI) -> list[str]:
    models = await client.models.list()
    return [model.id for model in models.data if model.owned_by == LGOS_MODEL_OWNER]
