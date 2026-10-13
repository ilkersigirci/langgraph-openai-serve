"""Gateway-specific OpenAI endpoint selection for the Chainlit demo."""

from dataclasses import dataclass
from typing import Literal

from openai.types import Model
from pydantic import BaseModel, Field, Json, JsonValue, ValidationError

from lgos_chainlit.lgos_protocol import LGOS_MODEL_OWNER

GatewayType = Literal["litellm", "bifrost"]


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
        if not model.id.startswith("lgos/"):
            continue
        attributes = (model.model_extra or {}).get("additional_attributes", {})
        try:
            extension = _BifrostAttributes.model_validate(attributes).lgos
        except ValidationError:
            extension = None
        result.append(
            model.model_copy(update={"owned_by": LGOS_MODEL_OWNER, "lgos": extension})
        )
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
    """Resolved routes of one supported gateway."""

    type: GatewayType
    root_url: str
    responses_base_url: str
    files_provider: str


def gateway_config(
    gateway_type: GatewayType,
    gateway_base_url: str,
) -> GatewayConfig:
    """Resolve gateway paths from an explicitly configured root."""
    root = gateway_base_url.rstrip("/")
    if gateway_type == "litellm":
        return GatewayConfig(
            type=gateway_type,
            root_url=root,
            responses_base_url=f"{root}/v1",
            files_provider="litellm_proxy",
        )

    return GatewayConfig(
        type=gateway_type,
        root_url=root,
        responses_base_url=f"{root}/openai/v1",
        files_provider="lgos-files",
    )
