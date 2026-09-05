"""Settings for the Open WebUI synchronization command."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from lgos_openwebui.functions.generic.gateway import GatewayType


class Settings(BaseSettings):
    """Configuration for the Open WebUI synchronization command."""

    model_config = SettingsConfigDict(env_prefix="DEMO_OPENWEBUI_")

    URL: str = Field(
        default="http://localhost:3003",
        description="Open WebUI API base URL used by the sync command.",
    )
    ADMIN_EMAIL: str = Field(
        default="lgos@example.com",
        description="Email for the Open WebUI account used by the sync command.",
    )
    ADMIN_PASSWORD: str = Field(
        default="lgos",
        description="Password for the Open WebUI account used by the sync command.",
    )
    OPENAI_GATEWAY_TYPE: GatewayType = Field(
        default="litellm",
        validation_alias="OPENAI_GATEWAY_TYPE",
        description="OpenAI gateway used for model sync and Function defaults.",
    )
    OPENAI_GATEWAY_BASE_URL: str | None = Field(
        default=None,
        description="Optional gateway root override for the host sync command.",
    )
    API_KEY: str = Field(
        default="sk-lgos-litellm-demo",
        description="API key sent to the configured OpenAI-compatible endpoints.",
    )
