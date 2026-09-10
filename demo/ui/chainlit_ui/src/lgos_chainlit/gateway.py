"""Gateway-specific OpenAI endpoint selection for the Chainlit demo."""

from dataclasses import dataclass
from typing import Literal

from openai.types import Model
from pydantic import BaseModel, Field, JsonValue

GatewayType = Literal["litellm", "bifrost"]


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
                    "owned_by": "langgraph-openai-serve",
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


__all__ = ["GatewayConfig", "GatewayType", "gateway_config", "litellm_models"]
