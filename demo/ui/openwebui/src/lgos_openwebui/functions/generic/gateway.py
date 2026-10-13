"""Gateway settings, routes, and catalog decoding for the Generic Pipe."""

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Annotated, Any, Literal

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

from .contracts import LGOS_MODEL_NAMESPACE, LGOS_MODEL_OWNER

GatewayType = Literal["litellm", "bifrost"]
# The TOOL_SERVER_CONNECTIONS entry in demo/docker/apps/openwebui.yml uses this ID.
MCP_GATEWAY_ID = "lgos-gateway"
AnyHttpUrlAdapter = TypeAdapter(AnyHttpUrl)
GatewayRoot = Annotated[
    str,
    PlainValidator(AnyHttpUrlAdapter.validate_strings, json_schema_input_type=str),
    AfterValidator(lambda value: str(value).rstrip("/")),
]


class GatewaySettings(BaseModel):
    """The gateway configured in Open WebUI's environment."""

    OPENAI_GATEWAY_TYPE: GatewayType
    OPENAI_GATEWAY_BASE_URL: GatewayRoot
    # Unused when Open WebUI's connection to the gateway sends user tokens.
    OPENAI_GATEWAY_API_KEY: str | None = None


def gateway_settings() -> GatewaySettings:
    """Read the gateway settings without echoing the environment on failure."""
    # An empty value counts as unset, as the SSO overlay clears the key.
    environment = {
        name: value
        for name in GatewaySettings.model_fields
        if (value := os.environ.get(name))
    }
    try:
        return GatewaySettings.model_validate(environment)
    except ValidationError as exc:
        # Validation messages quote their input, which includes the gateway key.
        names = ", ".join(sorted({str(error["loc"][0]) for error in exc.errors()}))
        msg = f"Set valid gateway settings in Open WebUI's environment: {names}."
        raise RuntimeError(msg) from None


async def connection_sends_user_token(root_url: str) -> bool:
    """Whether Open WebUI's OpenAI connection to the gateway uses system_oauth."""
    # Open WebUI is installed only in its image, not in this project.
    openai_router: Any = import_module("open_webui.routers.openai")
    _, base_urls, _, configs = await openai_router.get_openai_runtime_config()
    for index, url in enumerate(base_urls):
        if url.rstrip("/") == f"{root_url}/v1":
            config = configs.get(str(index), configs.get(url, {}))
            return config.get("auth_type") == "system_oauth"
    return False


class _LiteLLMDeployment(BaseModel):
    model_name: str = Field(min_length=1)
    model_info: dict[str, JsonValue]


class _LiteLLMCatalog(BaseModel):
    data: list[_LiteLLMDeployment]


class _BifrostAttributes(BaseModel):
    lgos: Json[dict[str, JsonValue]] | None = None


def bifrost_models(models: list[Model]) -> list[Model]:
    """Decode native catalog attributes; incomplete metadata stays visible."""
    result = []
    for model in models:
        # Bifrost aliases omit owned_by; the public namespace identifies LGOS.
        if not model.id.startswith(LGOS_MODEL_NAMESPACE):
            continue
        attributes = (model.model_extra or {}).get("additional_attributes", {})
        try:
            extension = _BifrostAttributes.model_validate(attributes).lgos
        except ValidationError:
            extension = None
        result.append(model.model_copy(update={"lgos": extension}))
    return result


def litellm_models(payload: object) -> list[Model]:
    """Expose LGOS metadata once per public LiteLLM model name."""
    catalog = _LiteLLMCatalog.model_validate(payload)
    return list(
        {
            item.model_name: Model.model_validate(
                {
                    "id": item.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": LGOS_MODEL_OWNER,
                    "lgos": item.model_info["lgos"],
                }
            )
            for item in catalog.data
            if item.model_info.get("lgos") is not None
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


__all__ = [
    "MCP_GATEWAY_ID",
    "GatewayConfig",
    "GatewaySettings",
    "bifrost_models",
    "connection_sends_user_token",
    "gateway_config",
    "gateway_settings",
    "litellm_models",
]
