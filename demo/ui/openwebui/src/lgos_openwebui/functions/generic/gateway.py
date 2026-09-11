"""Gateway-specific endpoint selection shared by sync and the bundled Pipe."""

from dataclasses import dataclass
from typing import Annotated, Literal

from openai.types import Model
from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    Field,
    JsonValue,
    PlainValidator,
    TypeAdapter,
)

from .contracts import LGOS_MODEL_OWNER

GatewayType = Literal["litellm", "bifrost"]
AnyHttpUrlAdapter = TypeAdapter(AnyHttpUrl)
GatewayRoot = Annotated[
    str,
    PlainValidator(AnyHttpUrlAdapter.validate_strings, json_schema_input_type=str),
    AfterValidator(lambda value: str(value).rstrip("/")),
]


class _LiteLLMDeployment(BaseModel):
    model_name: str = Field(min_length=1)
    model_info: dict[str, JsonValue]


class _LiteLLMCatalog(BaseModel):
    data: list[_LiteLLMDeployment]


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
    "GatewayConfig",
    "GatewayRoot",
    "GatewayType",
    "gateway_config",
    "litellm_models",
]
