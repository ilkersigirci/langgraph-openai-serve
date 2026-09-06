"""OpenAI-compatible client and model-catalog helpers."""

from typing import Any

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


def _model_id(body: dict[str, Any]) -> str:
    qualified_model_id = body.get("model")
    if not isinstance(qualified_model_id, str):
        msg = "Open WebUI did not provide a valid model ID."
        raise ValueError(msg)

    _, separator, model_id = qualified_model_id.partition(".")
    if not separator or not model_id:
        msg = "Open WebUI did not provide a valid model ID."
        raise ValueError(msg)

    return model_id


async def _list_model_ids(
    client: AsyncOpenAI,
    *,
    model_prefix: str | None = None,
) -> list[str]:
    models = await client.models.list()
    model_ids = [
        model.id for model in models.data if model.owned_by == LGOS_MODEL_OWNER
    ]
    if model_prefix is None:
        return model_ids
    return [_managed_model_id(model_id, (model_prefix,)) for model_id in model_ids]


def _model_request(
    model_id: str,
    *,
    provider_routing: bool,
    model_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not model_id:
        msg = "OpenAI model ID is missing."
        raise ValueError(msg)
    if model_prefixes:
        return {"model": _managed_model_id(model_id, model_prefixes)}
    if not provider_routing:
        msg = "Gateway model routing is not configured."
        raise ValueError(msg)

    provider, separator, upstream_model = model_id.partition("/")
    if not provider or not separator or not upstream_model:
        msg = f"Bifrost model ID must use the provider/model format: {model_id!r}."
        raise ValueError(msg)

    return {
        "model": upstream_model,
        "extra_headers": {"x-model-provider": provider},
    }


def _catalog_base_url(catalog_root: str, model_prefix: str) -> str:
    return f"{catalog_root.rstrip('/')}/{model_prefix}"


def _managed_model_id(model_id: str, model_prefixes: tuple[str, ...]) -> str:
    provider, separator, upstream_model = model_id.partition("/")
    if not separator:
        return f"{model_prefixes[0]}/{model_id}"
    if provider not in model_prefixes or not upstream_model:
        expected = ", ".join(f"{prefix}/model" for prefix in model_prefixes)
        msg = f"LiteLLM model ID must use one of [{expected}]: {model_id!r}."
        raise ValueError(msg)
    return model_id
