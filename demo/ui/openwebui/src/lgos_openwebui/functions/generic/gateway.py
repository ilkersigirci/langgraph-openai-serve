"""Gateway settings, routes, and catalog decoding for the Generic Pipe."""

from dataclasses import dataclass
from importlib import import_module
from typing import Annotated, Any, Literal

from openai import AsyncOpenAI
from openai.types import Model
from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    Field,
    Json,
    JsonValue,
    PlainValidator,
    TypeAdapter,
    ValidationError,
)

from .contracts import LGOS_MODEL_OWNER

GatewayType = Literal["litellm", "bifrost"]
AnyHttpUrlAdapter = TypeAdapter(AnyHttpUrl)
GatewayRoot = Annotated[
    str,
    PlainValidator(AnyHttpUrlAdapter.validate_strings, json_schema_input_type=str),
    AfterValidator(lambda value: str(value).rstrip("/")),
]
GatewayRootAdapter = TypeAdapter(GatewayRoot)


class _LiteLLMDeployment(BaseModel):
    model_name: str = Field(min_length=1)
    model_info: dict[str, JsonValue]


class _LiteLLMCatalog(BaseModel):
    data: list[_LiteLLMDeployment]


class _BifrostAttributes(BaseModel):
    lgos: Json[dict[str, JsonValue]] | None = None


def bifrost_models(models: list[Model], model_prefix: str) -> list[Model]:
    """Decode native catalog attributes; incomplete metadata stays visible."""
    result = []
    for model in models:
        # Bifrost aliases omit owned_by; the public namespace identifies LGOS.
        if not model.id.startswith(f"{model_prefix}/"):
            continue
        attributes = (model.model_extra or {}).get("additional_attributes", {})
        try:
            extension = _BifrostAttributes.model_validate(attributes).lgos
        except ValidationError:
            extension = None
        result.append(model.model_copy(update={"lgos": extension}))
    return result


def litellm_models(payload: object, model_prefix: str) -> list[Model]:
    """Expose namespaced LiteLLM models once per public model name."""
    catalog = _LiteLLMCatalog.model_validate(payload)
    return list(
        {
            item.model_name: Model.model_validate(
                {
                    "id": item.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": LGOS_MODEL_OWNER,
                    "lgos": item.model_info.get("lgos"),
                }
            )
            for item in catalog.data
            if item.model_name.startswith(f"{model_prefix}/")
        }.values()
    )


@dataclass(frozen=True)
class GatewayConfig:
    """Resolved URLs and routing behavior for one supported gateway."""

    root_url: str
    responses_base_url: str
    provider_routing: bool
    files_base_url: str
    files_provider: str


def gateway_config(
    gateway_type: GatewayType,
    gateway_base_url: str,
) -> GatewayConfig:
    """Resolve gateway paths from an explicitly configured root."""
    root = gateway_base_url.rstrip("/")
    if gateway_type == "litellm":
        managed_base_url = f"{root}/v1"
        return GatewayConfig(
            root_url=root,
            responses_base_url=managed_base_url,
            provider_routing=False,
            files_base_url=managed_base_url,
            files_provider="litellm_proxy",
        )

    return GatewayConfig(
        root_url=root,
        responses_base_url=f"{root}/openai/v1",
        provider_routing=True,
        files_base_url=f"{root}/v1",
        files_provider="lgos-files",
    )


def _normalized_url(url: str) -> str | None:
    # The valves store their URL normalized, so connections compare the same way.
    try:
        return GatewayRootAdapter.validate_python(url)
    except ValidationError:
        return None


async def gateway_connection(root_url: str) -> tuple[str, str]:
    """Return the auth type and key of Open WebUI's OpenAI connection to the gateway."""
    # Open WebUI is installed only in its image, not in this project.
    openai_router: Any = import_module("open_webui.routers.openai")
    _, base_urls, keys, configs = await openai_router.get_openai_runtime_config()
    for index, url in enumerate(base_urls):
        if _normalized_url(url) == f"{root_url}/v1":
            # Open WebUI looks a connection's settings up by index, then by URL,
            # and sends the key when the connection sets no auth type.
            auth_type = configs.get(str(index), configs.get(url, {})).get("auth_type")
            return "bearer" if auth_type is None else auth_type, keys[index]
    msg = (
        f"Add an OpenAI connection to {root_url}/v1 in Open WebUI; "
        "the Generic Pipe authenticates like it."
    )
    raise RuntimeError(msg)


async def gateway_models(
    client: AsyncOpenAI, gateway: GatewayConfig, model_prefix: str
) -> list[Model]:
    """Read one namespace from the gateway's native metadata catalog."""
    if gateway.provider_routing:
        catalog = await client.models.list()
        return bifrost_models(catalog.data, model_prefix)
    payload = await client.get(f"{gateway.root_url}/model/info", cast_to=object)
    return litellm_models(payload, model_prefix)


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
