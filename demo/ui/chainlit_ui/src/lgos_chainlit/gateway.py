"""Gateway-specific OpenAI endpoint selection for the Chainlit demo."""

from dataclasses import dataclass
from typing import Literal

GatewayType = Literal["litellm", "bifrost"]
LITELLM_MODEL_PREFIXES = ("lgos-a", "lgos-b")
LOCAL_GATEWAY_URLS: dict[GatewayType, str] = {
    "litellm": "http://localhost:3007",
    "bifrost": "http://localhost:3000",
}


@dataclass(frozen=True)
class GatewayConfig:
    """Resolved URLs and routing behavior for one supported gateway."""

    responses_base_url: str
    catalog_base_url: str
    catalog_detail_base_url: str
    model_prefixes: tuple[str, ...]
    provider_routing: bool
    files_base_url: str
    files_provider: str


def gateway_config(
    gateway_type: GatewayType,
    gateway_base_url: str | None = None,
) -> GatewayConfig:
    """Resolve the selected gateway without leaking its paths into UI code."""
    root = (gateway_base_url or LOCAL_GATEWAY_URLS[gateway_type]).rstrip("/")
    if gateway_type == "litellm":
        managed_base_url = f"{root}/v1"
        return GatewayConfig(
            responses_base_url=managed_base_url,
            catalog_base_url=managed_base_url,
            catalog_detail_base_url=managed_base_url,
            model_prefixes=LITELLM_MODEL_PREFIXES,
            provider_routing=False,
            files_base_url=managed_base_url,
            files_provider="litellm_proxy",
        )

    return GatewayConfig(
        responses_base_url=f"{root}/openai/v1",
        catalog_base_url=f"{root}/v1",
        catalog_detail_base_url=f"{root}/openai_passthrough/v1",
        model_prefixes=(),
        provider_routing=True,
        files_base_url=f"{root}/v1",
        files_provider="lgos-files",
    )


__all__ = ["GatewayConfig", "GatewayType", "gateway_config"]
